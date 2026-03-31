"""Tests for Polymarket client."""

from oracle.ingestion.polymarket_client import PolymarketClient


def test_detect_category():
    assert PolymarketClient._detect_category({"question": "Will Trump win the 2026 election?"}) == "politics"
    assert PolymarketClient._detect_category({"question": "Will the Fed cut interest rates?"}) == "economics"
    assert PolymarketClient._detect_category({"question": "Will Bitcoin reach $100k?"}) == "crypto"
    assert PolymarketClient._detect_category({"question": "Will the Lakers win the NBA championship?"}) == "sports"
    assert PolymarketClient._detect_category({"question": "Will OpenAI release GPT-5?"}) == "tech"
    assert PolymarketClient._detect_category({"question": "Will the Supreme Court rule on the case?"}) == "legal"
    assert PolymarketClient._detect_category({"question": "Will Russia invade another country?"}) == "geopolitics"
    assert PolymarketClient._detect_category({"question": "Will it rain tomorrow in Paris?"}) == "other"


def test_market_to_properties():
    client = PolymarketClient.__new__(PolymarketClient)
    market = {
        "id": "abc123",
        "question": "Will the Fed cut rates?",
        "outcomePrices": "0.65",
        "volume": "1000000",
        "endDate": "2026-06-30",
        "active": True,
        "description": "Federal Reserve interest rate decision",
    }
    props = client._market_to_properties(market)
    assert props["polymarket_id"] == "abc123"
    assert props["question"] == "Will the Fed cut rates?"
    assert props["current_price"] == 0.65
    assert props["volume"] == 1000000.0
    assert props["category"] == "economics"
    assert props["active"] is True


def test_extract_id_fallbacks():
    client = PolymarketClient.__new__(PolymarketClient)
    assert client._extract_id({"id": "123"}) == "123"
    assert client._extract_id({"conditionId": "456"}) == "456"
    assert client._extract_id({"questionID": "789"}) == "789"
    assert client._extract_id({}) is None
