"""Entity extraction and resolution: spaCy NER → LLM deduplication/resolution."""

from __future__ import annotations

import re
from typing import Any

import structlog

logger = structlog.get_logger()

# Mapping from spaCy NER labels to Oracle node types
SPACY_TO_ORACLE = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "EVENT": "Event",
    "LAW": "Policy",
    "NORP": "Organization",  # nationalities/religious/political groups
}

# Known entity aliases for deduplication
KNOWN_ALIASES: dict[str, str] = {
    "the fed": "Federal Reserve",
    "federal reserve": "Federal Reserve",
    "the federal reserve": "Federal Reserve",
    "fed": "Federal Reserve",
    "scotus": "Supreme Court",
    "the supreme court": "Supreme Court",
    "supreme court": "Supreme Court",
    "potus": "President of the United States",
    "sec": "Securities and Exchange Commission",
    "doj": "Department of Justice",
    "gop": "Republican Party",
    "dems": "Democratic Party",
    "democrats": "Democratic Party",
    "republicans": "Republican Party",
    "donald trump": "Donald Trump",
    "trump": "Donald Trump",
    "biden": "Joe Biden",
    "joe biden": "Joe Biden",
    "elon musk": "Elon Musk",
    "musk": "Elon Musk",
    "jerome powell": "Jerome Powell",
    "powell": "Jerome Powell",
    "j. powell": "Jerome Powell",
}


class EntityResolver:
    """Two-stage entity extraction and resolution.

    Stage 1: spaCy NER to extract raw entities
    Stage 2: Alias resolution + fuzzy dedup
    """

    def __init__(self) -> None:
        self._nlp = None
        self._entity_cache: dict[str, dict[str, Any]] = {}

    def _load_spacy(self) -> Any:
        """Lazy-load spaCy model."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("entity_resolver.spacy_loaded")
        return self._nlp

    async def extract_and_resolve(self, text: str) -> list[dict[str, Any]]:
        """Extract entities from text and resolve to canonical forms.

        Returns list of {label: str, properties: dict} ready for Neo4j merge.
        """
        nlp = self._load_spacy()
        doc = nlp(text)

        raw_entities: list[tuple[str, str]] = []  # (text, label)
        for ent in doc.ents:
            if ent.label_ in SPACY_TO_ORACLE:
                raw_entities.append((ent.text.strip(), ent.label_))

        # Deduplicate and resolve
        resolved: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for raw_text, spacy_label in raw_entities:
            oracle_label = SPACY_TO_ORACLE[spacy_label]
            canonical = self._resolve_name(raw_text)

            if canonical in seen_names:
                continue
            seen_names.add(canonical)

            # Skip very short or generic entities
            if len(canonical) < 2 or canonical.lower() in {"the", "a", "an", "it", "they"}:
                continue

            properties = self._build_properties(canonical, oracle_label, spacy_label)
            resolved.append({"label": oracle_label, "properties": properties})

            # Cache for future reference
            self._entity_cache[canonical.lower()] = {
                "canonical": canonical,
                "label": oracle_label,
            }

        logger.info(
            "entity_resolver.extracted",
            raw_count=len(raw_entities),
            resolved_count=len(resolved),
        )
        return resolved

    def _resolve_name(self, raw_text: str) -> str:
        """Resolve entity name to canonical form using aliases and normalization."""
        normalized = raw_text.strip()

        # Check known aliases
        lookup = normalized.lower()
        if lookup in KNOWN_ALIASES:
            return KNOWN_ALIASES[lookup]

        # Check entity cache for fuzzy match
        if lookup in self._entity_cache:
            return self._entity_cache[lookup]["canonical"]

        # Clean up common patterns
        normalized = re.sub(r"^(the|a|an)\s+", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", " ", normalized)

        # Title-case person/org names
        if not normalized.isupper():  # Don't change acronyms
            normalized = normalized.title()

        return normalized.strip()

    @staticmethod
    def _build_properties(
        canonical: str, oracle_label: str, spacy_label: str
    ) -> dict[str, Any]:
        """Build Neo4j node properties based on entity type."""
        base = {"name": canonical}

        if oracle_label == "Person":
            base["role"] = ""
            base["party"] = ""
            base["organization"] = ""
        elif oracle_label == "Organization":
            base["type"] = _infer_org_type(canonical)
            base["sector"] = ""
            base["country"] = ""
        elif oracle_label == "Location":
            base["country"] = ""
            base["region"] = ""
        elif oracle_label == "Event":
            base["status"] = "unknown"
        elif oracle_label == "Policy":
            base["status"] = "unknown"

        return base

    async def resolve_relationships(
        self, entities: list[dict[str, Any]], text: str
    ) -> list[dict[str, Any]]:
        """Infer relationships between co-occurring entities.

        Simple heuristic: entities mentioned in the same sentence are likely related.
        """
        nlp = self._load_spacy()
        doc = nlp(text)

        relationships = []
        for sent in doc.sents:
            sent_entities = [
                e for e in entities
                if e["properties"]["name"].lower() in sent.text.lower()
            ]

            # Create pairwise relationships for co-occurring entities
            for i, e1 in enumerate(sent_entities):
                for e2 in sent_entities[i + 1:]:
                    if e1["label"] != e2["label"] or e1["properties"]["name"] != e2["properties"]["name"]:
                        relationships.append({
                            "from_label": e1["label"],
                            "from_key": e1["properties"]["name"],
                            "rel_type": "RELATED_TO",
                            "to_label": e2["label"],
                            "to_key": e2["properties"]["name"],
                        })

        return relationships


def _infer_org_type(name: str) -> str:
    """Infer organization type from name."""
    name_lower = name.lower()

    gov_keywords = ["department", "agency", "bureau", "commission", "federal", "congress",
                    "senate", "court", "pentagon", "white house"]
    if any(kw in name_lower for kw in gov_keywords):
        return "government"

    corp_keywords = ["inc", "corp", "ltd", "llc", "company", "group", "holdings"]
    if any(kw in name_lower for kw in corp_keywords):
        return "corporation"

    if any(kw in name_lower for kw in ["party", "democratic", "republican"]):
        return "political_party"

    if any(kw in name_lower for kw in ["university", "institute", "school", "college"]):
        return "education"

    return "organization"
