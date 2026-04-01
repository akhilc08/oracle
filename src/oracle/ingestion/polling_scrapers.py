"""Polling aggregator scrapers — FiveThirtyEight, RealClearPolitics."""

from __future__ import annotations

import csv
import hashlib
import io
import sqlite3
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from oracle.knowledge.qdrant_client import QdrantManager

logger = structlog.get_logger()

FIVETHIRTYEIGHT_POLLS_URL = (
    "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
)
RCP_BASE_URL = "https://www.realclearpolls.com/"


class PollingScraper:
    """Scraper for political polling data — FiveThirtyEight and RealClearPolitics."""

    def __init__(
        self,
        qdrant: QdrantManager | None = None,
        db_path: str = "oracle_polls.db",
    ) -> None:
        self._qdrant = qdrant
        self._db_path = db_path
        self._setup_db()

    def _setup_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS polls (
                id TEXT PRIMARY KEY,
                race TEXT NOT NULL,
                candidate TEXT NOT NULL,
                pollster TEXT NOT NULL,
                date TEXT NOT NULL,
                sample_size INTEGER DEFAULT 0,
                value REAL NOT NULL,
                margin_of_error REAL DEFAULT 0,
                source TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS polling_averages (
                id TEXT PRIMARY KEY,
                race TEXT NOT NULL,
                candidate TEXT NOT NULL,
                average REAL NOT NULL,
                poll_count INTEGER NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    # ---- FiveThirtyEight Polls ----

    async def fetch_fivethirtyeight(self) -> list[dict[str, Any]]:
        """Fetch presidential polls from FiveThirtyEight CSV."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.get(FIVETHIRTYEIGHT_POLLS_URL)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.warning("polling.538_fetch_error", error=str(e))
                return []

        reader = csv.DictReader(io.StringIO(resp.text))
        polls: list[dict[str, Any]] = []

        for row in reader:
            poll_id = hashlib.md5(
                f"538:{row.get('poll_id', '')}:{row.get('candidate_name', '')}".encode()
            ).hexdigest()

            sample_size = 0
            try:
                sample_size = int(row.get("sample_size", 0) or 0)
            except (ValueError, TypeError):
                pass

            pct = 0.0
            try:
                pct = float(row.get("pct", 0) or 0)
            except (ValueError, TypeError):
                pass

            poll = {
                "id": poll_id,
                "race": row.get("office_type", "President"),
                "candidate": row.get("candidate_name", ""),
                "pollster": row.get("pollster", ""),
                "date": row.get("end_date", ""),
                "sample_size": sample_size,
                "value": pct,
                "margin_of_error": 0,
                "source": "fivethirtyeight",
            }
            polls.append(poll)

        logger.info("polling.538_fetched", count=len(polls))
        return polls

    # ---- RealClearPolitics ----

    async def fetch_realclearpolitics(self) -> list[dict[str, Any]]:
        """Scrape polling data from RealClearPolitics."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    RCP_BASE_URL,
                    headers={"User-Agent": "Oracle/1.0 (research bot)"},
                )
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.warning("polling.rcp_fetch_error", error=str(e))
                return []

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("polling.bs4_not_installed", msg="beautifulsoup4 required for RCP scraping")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        polls: list[dict[str, Any]] = []

        # Find poll tables — RCP uses table elements with polling data
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue

            # Try to extract header to determine race type
            header_cells = rows[0].find_all(["th", "td"])
            headers = [cell.get_text(strip=True) for cell in header_cells]

            for row in rows[1:]:
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue

                cell_texts = [c.get_text(strip=True) for c in cells]

                # Heuristic: look for rows with pollster name, numbers
                pollster = cell_texts[0] if cell_texts else ""
                if not pollster or pollster.isdigit():
                    continue

                # Try to extract numeric values
                for i, text in enumerate(cell_texts[1:], 1):
                    try:
                        value = float(text.replace("%", ""))
                        candidate = headers[i] if i < len(headers) else f"candidate_{i}"
                        poll_id = hashlib.md5(
                            f"rcp:{pollster}:{candidate}:{datetime.now(timezone.utc).date()}".encode()
                        ).hexdigest()

                        polls.append({
                            "id": poll_id,
                            "race": "President",  # Default; refined by table context
                            "candidate": candidate,
                            "pollster": pollster,
                            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                            "sample_size": 0,
                            "value": value,
                            "margin_of_error": 0,
                            "source": "realclearpolitics",
                        })
                    except (ValueError, TypeError):
                        continue

        logger.info("polling.rcp_fetched", count=len(polls))
        return polls

    # ---- Storage & Averages ----

    def store_polls(self, polls: list[dict[str, Any]]) -> int:
        """Store poll records in SQLite. Returns count of new polls stored."""
        if not polls:
            return 0

        conn = sqlite3.connect(self._db_path)
        stored = 0
        for poll in polls:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO polls
                       (id, race, candidate, pollster, date, sample_size, value, margin_of_error, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        poll["id"], poll["race"], poll["candidate"], poll["pollster"],
                        poll["date"], poll["sample_size"], poll["value"],
                        poll["margin_of_error"], poll["source"],
                    ),
                )
                if conn.total_changes > stored:
                    stored = conn.total_changes
            except sqlite3.Error:
                continue

        conn.commit()
        final_count = conn.total_changes
        conn.close()
        return final_count

    def compute_polling_averages(self) -> list[dict[str, Any]]:
        """Compute simple mean of last 5 polls per race/candidate combo."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row

        # Get distinct race/candidate combos
        combos = conn.execute(
            "SELECT DISTINCT race, candidate FROM polls WHERE candidate != ''"
        ).fetchall()

        averages: list[dict[str, Any]] = []

        for combo in combos:
            race, candidate = combo["race"], combo["candidate"]
            rows = conn.execute(
                """SELECT value FROM polls
                   WHERE race = ? AND candidate = ?
                   ORDER BY date DESC LIMIT 5""",
                (race, candidate),
            ).fetchall()

            if not rows:
                continue

            values = [r["value"] for r in rows]
            avg = sum(values) / len(values)

            avg_id = hashlib.md5(f"avg:{race}:{candidate}".encode()).hexdigest()
            averages.append({
                "id": avg_id,
                "race": race,
                "candidate": candidate,
                "average": round(avg, 2),
                "poll_count": len(values),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            })

            # Store in DB
            conn.execute(
                """INSERT OR REPLACE INTO polling_averages
                   (id, race, candidate, average, poll_count, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (avg_id, race, candidate, avg, len(values),
                 datetime.now(timezone.utc).isoformat()),
            )

        conn.commit()
        conn.close()

        logger.info("polling.averages_computed", count=len(averages))
        return averages

    async def ingest_all(self) -> dict[str, Any]:
        """Full polling ingestion cycle: fetch → store → compute averages → ingest summaries."""
        stats: dict[str, Any] = {"polls_fetched": 0, "averages_computed": 0, "summaries_ingested": 0}

        # Fetch from both sources
        fte_polls = await self.fetch_fivethirtyeight()
        rcp_polls = await self.fetch_realclearpolitics()
        all_polls = fte_polls + rcp_polls
        stats["polls_fetched"] = len(all_polls)

        # Store
        self.store_polls(all_polls)

        # Compute averages
        averages = self.compute_polling_averages()
        stats["averages_computed"] = len(averages)

        # Ingest summaries into Qdrant as context
        if self._qdrant and averages:
            from oracle.knowledge.embeddings import EmbeddingService
            embedder = EmbeddingService.get_instance()

            texts = []
            for avg in averages:
                text = (
                    f"Polling Average: {avg['race']} — {avg['candidate']}: "
                    f"{avg['average']}% (based on {avg['poll_count']} recent polls)"
                )
                texts.append(text)

            embeddings = embedder.embed(texts)
            ids = []
            vectors = []
            payloads = []

            for avg, text, embedding in zip(averages, texts, embeddings):
                chunk_id = hashlib.md5(f"poll_avg:{avg['id']}".encode()).hexdigest()
                ids.append(chunk_id)
                vectors.append(embedding)
                payloads.append({
                    "text": text,
                    "source_url": "polling_aggregate",
                    "publication_date": avg["last_updated"],
                    "author": "polling_aggregator",
                    "source_authority_score": 0.85,
                    "entity_ids": [avg["candidate"]],
                    "market_ids": [],
                    "source": "polling_aggregate",
                    "race": avg["race"],
                    "candidate": avg["candidate"],
                })

            await self._qdrant.upsert_chunks(
                collection="news_articles",
                ids=ids,
                vectors=vectors,
                payloads=payloads,
            )
            stats["summaries_ingested"] = len(ids)

        logger.info("polling.ingestion_complete", **stats)
        return stats
