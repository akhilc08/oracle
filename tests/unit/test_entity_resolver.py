"""Tests for entity resolution."""

import pytest

from oracle.ingestion.entity_resolver import EntityResolver, KNOWN_ALIASES, _infer_org_type


class TestEntityResolver:
    def setup_method(self):
        self.resolver = EntityResolver()

    def test_resolve_known_alias(self):
        assert self.resolver._resolve_name("the fed") == "Federal Reserve"
        assert self.resolver._resolve_name("SCOTUS") == "Supreme Court"
        assert self.resolver._resolve_name("Trump") == "Donald Trump"
        assert self.resolver._resolve_name("Musk") == "Elon Musk"

    def test_resolve_strips_articles(self):
        result = self.resolver._resolve_name("the Senate")
        assert not result.lower().startswith("the ")

    def test_resolve_normalizes_whitespace(self):
        result = self.resolver._resolve_name("  John   Smith  ")
        assert "  " not in result

    @pytest.mark.asyncio
    async def test_extract_and_resolve(self):
        text = "Donald Trump met with Elon Musk at the White House to discuss Tesla's new factory in Texas."
        entities = await self.resolver.extract_and_resolve(text)
        assert len(entities) > 0

        names = {e["properties"]["name"] for e in entities}
        # Should extract at least Trump and Musk
        assert "Donald Trump" in names or any("trump" in n.lower() for n in names)

    @pytest.mark.asyncio
    async def test_deduplication(self):
        text = "Trump said that Trump would meet with Trump's team."
        entities = await self.resolver.extract_and_resolve(text)
        names = [e["properties"]["name"] for e in entities]
        # Should not have duplicate names
        assert len(names) == len(set(names))

    @pytest.mark.asyncio
    async def test_short_entities_filtered(self):
        text = "A and the were mentioned."
        entities = await self.resolver.extract_and_resolve(text)
        names = [e["properties"]["name"] for e in entities]
        assert "A" not in names
        assert "The" not in names


def test_infer_org_type():
    assert _infer_org_type("Department of Justice") == "government"
    assert _infer_org_type("Apple Inc") == "corporation"
    assert _infer_org_type("Democratic Party") == "political_party"
    assert _infer_org_type("MIT") == "organization"
    assert _infer_org_type("Harvard University") == "education"
