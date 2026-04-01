"""Government data scrapers — Congress.gov, SEC EDGAR, CourtListener."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import structlog

from oracle.config import settings
from oracle.ingestion.entity_resolver import EntityResolver
from oracle.knowledge.neo4j_client import Neo4jClient
from oracle.knowledge.qdrant_client import QdrantManager

logger = structlog.get_logger()


class GovScraper:
    """Scraper for government data sources — bills, SEC filings, court opinions."""

    def __init__(
        self,
        neo4j: Neo4jClient | None = None,
        qdrant: QdrantManager | None = None,
        entity_resolver: EntityResolver | None = None,
        db_path: str = "oracle_gov.db",
    ) -> None:
        self._neo4j = neo4j
        self._qdrant = qdrant
        self._entity_resolver = entity_resolver or EntityResolver()
        self._db_path = db_path
        self._setup_db()

    def _setup_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gov_last_fetched (
                source TEXT PRIMARY KEY,
                last_fetched TEXT NOT NULL,
                doc_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gov_documents (
                doc_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT DEFAULT '',
                fetched_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _get_last_fetched(self, source: str) -> str | None:
        conn = sqlite3.connect(self._db_path)
        row = conn.execute(
            "SELECT last_fetched FROM gov_last_fetched WHERE source = ?", (source,)
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def _update_last_fetched(self, source: str, doc_count: int) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT OR REPLACE INTO gov_last_fetched (source, last_fetched, doc_count) VALUES (?, ?, ?)",
            (source, datetime.now(timezone.utc).isoformat(), doc_count),
        )
        conn.commit()
        conn.close()

    def _is_doc_seen(self, doc_id: str) -> bool:
        conn = sqlite3.connect(self._db_path)
        row = conn.execute("SELECT 1 FROM gov_documents WHERE doc_id = ?", (doc_id,)).fetchone()
        conn.close()
        return row is not None

    def _mark_doc_seen(self, doc_id: str, source: str, title: str = "") -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT OR IGNORE INTO gov_documents (doc_id, source, title, fetched_at) VALUES (?, ?, ?, ?)",
            (doc_id, source, title, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

    # ---- Congress.gov Bills ----

    async def fetch_congress_bills(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch latest bills from Congress.gov API."""
        if not settings.congress_api_key:
            logger.warning("gov.no_congress_key", msg="CONGRESS_API_KEY not configured")
            return []

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://api.congress.gov/v3/bill",
                params={
                    "api_key": settings.congress_api_key,
                    "limit": limit,
                    "sort": "updateDate+desc",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        bills = []
        for bill_data in data.get("bills", []):
            bill_id = f"{bill_data.get('type', '')}{bill_data.get('number', '')}-{bill_data.get('congress', '')}"
            if self._is_doc_seen(bill_id):
                continue

            bill = {
                "id": bill_id,
                "bill_number": f"{bill_data.get('type', '')} {bill_data.get('number', '')}",
                "title": bill_data.get("title", ""),
                "sponsor": bill_data.get("sponsors", [{}])[0].get("fullName", "") if bill_data.get("sponsors") else "",
                "status": bill_data.get("latestAction", {}).get("text", ""),
                "subjects": [],
                "summary": bill_data.get("title", ""),
                "url": bill_data.get("url", ""),
                "update_date": bill_data.get("updateDate", ""),
                "source": "congress_gov",
            }
            bills.append(bill)
            self._mark_doc_seen(bill_id, "congress_gov", bill["title"])

        logger.info("gov.congress_fetched", count=len(bills))
        return bills

    # ---- SEC EDGAR Filings ----

    async def fetch_sec_filings(
        self, filing_types: list[str] | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Fetch recent SEC EDGAR filings."""
        filing_types = filing_types or ["8-K", "10-K", "DEF 14A"]
        start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

        filings: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for ftype in filing_types:
                try:
                    resp = await client.get(
                        "https://efts.sec.gov/LATEST/search-index",
                        params={
                            "q": ftype,
                            "dateRange": "custom",
                            "startdt": start_date,
                            "forms": ftype,
                        },
                        headers={"User-Agent": "Oracle/1.0 research@oracle.dev"},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    for hit in data.get("hits", {}).get("hits", [])[:limit]:
                        source = hit.get("_source", {})
                        doc_id = hit.get("_id", "")
                        if self._is_doc_seen(doc_id):
                            continue

                        filing = {
                            "id": doc_id,
                            "company": source.get("display_names", [""])[0] if source.get("display_names") else "",
                            "filing_type": ftype,
                            "date": source.get("file_date", ""),
                            "description": source.get("display_description", "")[:500],
                            "url": f"https://www.sec.gov/Archives/edgar/data/{source.get('file_num', '')}",
                            "source": "sec_edgar",
                        }
                        filings.append(filing)
                        self._mark_doc_seen(doc_id, "sec_edgar", filing["company"])
                except httpx.HTTPError as e:
                    logger.warning("gov.sec_fetch_error", filing_type=ftype, error=str(e))

        logger.info("gov.sec_fetched", count=len(filings))
        return filings

    # ---- CourtListener Opinions ----

    async def fetch_court_opinions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Fetch recent court opinions from CourtListener API."""
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        headers: dict[str, str] = {}
        if settings.courtlistener_api_key:
            headers["Authorization"] = f"Token {settings.courtlistener_api_key}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    "https://www.courtlistener.com/api/rest/v3/opinions/",
                    params={
                        "date_created__gte": seven_days_ago,
                        "court__in": "scotus,ca1,ca2,ca3,ca4,ca5,ca6,ca7,ca8,ca9,ca10,ca11,cadc,cafc",
                        "order_by": "-date_created",
                        "page_size": limit,
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as e:
                logger.warning("gov.courtlistener_error", error=str(e))
                return []

        opinions = []
        for result in data.get("results", []):
            doc_id = str(result.get("id", ""))
            if self._is_doc_seen(doc_id):
                continue

            plain_text = result.get("plain_text", "") or ""
            opinion = {
                "id": doc_id,
                "case_name": result.get("case_name", "Unknown"),
                "court": result.get("court", ""),
                "date_filed": result.get("date_created", ""),
                "summary": plain_text[:500] if plain_text else "",
                "url": f"https://www.courtlistener.com{result.get('absolute_url', '')}",
                "source": "courtlistener",
            }
            opinions.append(opinion)
            self._mark_doc_seen(doc_id, "courtlistener", opinion["case_name"])

        logger.info("gov.court_opinions_fetched", count=len(opinions))
        return opinions

    # ---- Ingest All Sources ----

    async def ingest_all(self) -> dict[str, Any]:
        """Fetch and ingest from all government sources."""
        stats: dict[str, Any] = {"bills": 0, "sec_filings": 0, "court_opinions": 0, "entities": 0}

        # Fetch from all sources
        bills = await self.fetch_congress_bills()
        filings = await self.fetch_sec_filings()
        opinions = await self.fetch_court_opinions()

        all_docs = []

        # Process bills
        for bill in bills:
            all_docs.append({
                "text": f"{bill['bill_number']}: {bill['title']}\nSponsor: {bill['sponsor']}\nStatus: {bill['status']}",
                "metadata": bill,
                "neo4j_label": "Policy",
                "neo4j_props": {"name": bill["title"][:200], "status": bill["status"]},
            })
        stats["bills"] = len(bills)

        # Process SEC filings
        for filing in filings:
            all_docs.append({
                "text": f"{filing['filing_type']} Filing: {filing['company']}\n{filing['description']}",
                "metadata": filing,
                "neo4j_label": "Organization",
                "neo4j_props": {"name": filing["company"], "type": "corporation"},
            })
        stats["sec_filings"] = len(filings)

        # Process court opinions
        for opinion in opinions:
            all_docs.append({
                "text": f"Court Opinion: {opinion['case_name']}\nCourt: {opinion['court']}\n{opinion['summary']}",
                "metadata": opinion,
                "neo4j_label": "LegalCase",
                "neo4j_props": {"case_name": opinion["case_name"], "court": opinion["court"]},
            })
        stats["court_opinions"] = len(opinions)

        # Ingest to Qdrant
        if self._qdrant and all_docs:
            from oracle.knowledge.embeddings import EmbeddingService
            embedder = EmbeddingService.get_instance()

            texts = [d["text"] for d in all_docs]
            embeddings = embedder.embed(texts)
            ids = []
            vectors = []
            payloads = []

            for doc, embedding in zip(all_docs, embeddings):
                chunk_id = hashlib.md5(
                    f"gov:{doc['metadata'].get('source', '')}:{doc['metadata'].get('id', '')}".encode()
                ).hexdigest()
                ids.append(chunk_id)
                vectors.append(embedding)
                payloads.append({
                    "text": doc["text"],
                    "source_url": doc["metadata"].get("url", ""),
                    "publication_date": doc["metadata"].get("date", doc["metadata"].get("date_filed", "")),
                    "author": doc["metadata"].get("source", "government"),
                    "source_authority_score": 0.95,
                    "entity_ids": [],
                    "market_ids": [],
                    "source": doc["metadata"].get("source", "government"),
                })

            await self._qdrant.upsert_chunks(
                collection="official_documents",
                ids=ids,
                vectors=vectors,
                payloads=payloads,
            )

        # Merge entities to Neo4j
        if self._neo4j:
            for doc in all_docs:
                if doc["neo4j_props"].get("name") or doc["neo4j_props"].get("case_name"):
                    await self._neo4j.merge_entity(doc["neo4j_label"], doc["neo4j_props"])
                    stats["entities"] += 1

                # Also extract entities from text
                entities = await self._entity_resolver.extract_and_resolve(doc["text"])
                for entity in entities:
                    await self._neo4j.merge_entity(entity["label"], entity["properties"])
                    stats["entities"] += 1

        self._update_last_fetched("congress_gov", stats["bills"])
        self._update_last_fetched("sec_edgar", stats["sec_filings"])
        self._update_last_fetched("courtlistener", stats["court_opinions"])

        logger.info("gov.ingestion_complete", **stats)
        return stats
