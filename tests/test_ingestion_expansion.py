"""Tests for Phase 6 — ingestion expansion (Twitter, Reddit, Whisper, gov scrapers, polling, vision)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---- Task 6.1: Twitter rate limit backoff ----


class TestTwitterRateLimitBackoff:
    """Test that Twitter client handles 429 rate limits with exponential backoff."""

    @pytest.mark.asyncio
    async def test_backoff_on_429(self):
        """Client should retry with backoff when receiving 429."""
        from oracle.ingestion.twitter_client import TwitterClient

        client = TwitterClient()

        # Mock settings to have a bearer token
        with patch("oracle.ingestion.twitter_client.settings") as mock_settings:
            mock_settings.twitter_bearer_token = "test_token"

            # Create mock responses: first 429, then 200
            mock_429 = MagicMock()
            mock_429.status_code = 429
            mock_429.headers = {"x-rate-limit-reset": str(datetime.now(timezone.utc).timestamp() + 1)}

            mock_200 = MagicMock()
            mock_200.status_code = 200
            mock_200.raise_for_status = MagicMock()
            mock_200.json.return_value = {"data": []}

            mock_http_client = AsyncMock()
            mock_http_client.request = AsyncMock(side_effect=[mock_429, mock_200])

            # Use a short sleep to speed up test
            with patch("asyncio.sleep", new_callable=AsyncMock):
                resp = await client._request_with_backoff(
                    mock_http_client, "GET", "https://api.twitter.com/2/tweets/search/recent"
                )

            assert resp.status_code == 200
            assert mock_http_client.request.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Client should raise after max retries on persistent 429."""
        from oracle.ingestion.twitter_client import TwitterClient

        import httpx
        client = TwitterClient()

        with patch("oracle.ingestion.twitter_client.settings") as mock_settings:
            mock_settings.twitter_bearer_token = "test_token"

            mock_429 = MagicMock()
            mock_429.status_code = 429
            mock_429.headers = {}

            mock_http_client = AsyncMock()
            mock_http_client.request = AsyncMock(return_value=mock_429)

            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(httpx.HTTPStatusError):
                    await client._request_with_backoff(
                        mock_http_client, "GET", "https://api.twitter.com/2/tweets/search/recent"
                    )

            # Should have retried 5 times
            assert mock_http_client.request.call_count == 5

    def test_sentiment_detection(self):
        """Test keyword-based sentiment detection."""
        from oracle.ingestion.twitter_client import detect_sentiment

        assert detect_sentiment("Bitcoin is surging, very bullish!") == "positive"
        assert detect_sentiment("Market crash incoming, bearish outlook") == "negative"
        assert detect_sentiment("The weather is nice today") == "neutral"

    def test_clean_tweet_text(self):
        """Test tweet text cleaning."""
        from oracle.ingestion.twitter_client import clean_tweet_text

        raw = "@user Check this out https://t.co/abc  #crypto"
        cleaned = clean_tweet_text(raw)
        assert "https://" not in cleaned
        assert "@user" not in cleaned
        assert "#crypto" in cleaned

    def test_extract_keywords_from_markets(self):
        """Test keyword extraction from market questions."""
        from oracle.ingestion.twitter_client import extract_keywords_from_markets

        markets = [
            {"question": "Will Trump win the 2024 Election?"},
            {"question": "Will Bitcoin reach $100k by December?"},
        ]
        keywords = extract_keywords_from_markets(markets)
        assert "Trump" in keywords
        assert "Election" in keywords
        assert "Bitcoin" in keywords

    @pytest.mark.asyncio
    async def test_graceful_skip_no_api_key(self):
        """Client should return empty list when no API key is set."""
        from oracle.ingestion.twitter_client import TwitterClient

        client = TwitterClient()
        with patch("oracle.ingestion.twitter_client.settings") as mock_settings:
            mock_settings.twitter_bearer_token = ""
            result = await client.search_recent("test query")
            assert result == []


# ---- Task 6.2: Reddit deduplication ----


class TestRedditDeduplication:
    """Test that Reddit client doesn't ingest the same post twice."""

    @pytest.mark.asyncio
    async def test_dedup_same_post(self):
        """Same post ID should not appear twice in results."""
        from oracle.ingestion.reddit_client import RedditClient

        client = RedditClient()

        # Simulate the Reddit API response with duplicate post IDs
        reddit_response = {
            "data": {
                "children": [
                    {"data": {"id": "abc123", "title": "Test Post", "selftext": "", "score": 100, "created_utc": 1700000000}},
                    {"data": {"id": "def456", "title": "Another Post", "selftext": "", "score": 50, "created_utc": 1700000001}},
                ]
            }
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = reddit_response

        with patch("oracle.ingestion.reddit_client.settings") as mock_settings:
            mock_settings.reddit_client_id = "test_id"
            mock_settings.reddit_client_secret = "test_secret"

            with patch.object(client, "_authenticate", return_value="fake_token"):
                with patch("httpx.AsyncClient") as mock_client_cls:
                    mock_instance = AsyncMock()
                    mock_instance.get = AsyncMock(return_value=mock_resp)
                    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                    mock_instance.__aexit__ = AsyncMock(return_value=False)
                    mock_client_cls.return_value = mock_instance

                    # First fetch
                    posts1 = await client.fetch_hot("politics")
                    assert len(posts1) == 2

                    # Second fetch with same data — should be deduplicated
                    posts2 = await client.fetch_hot("politics")
                    assert len(posts2) == 0  # All already seen

    @pytest.mark.asyncio
    async def test_graceful_skip_no_credentials(self):
        """Client should return empty list when no credentials set."""
        from oracle.ingestion.reddit_client import RedditClient

        client = RedditClient()
        with patch("oracle.ingestion.reddit_client.settings") as mock_settings:
            mock_settings.reddit_client_id = ""
            mock_settings.reddit_client_secret = ""
            result = await client.fetch_hot("politics")
            assert result == []


# ---- Task 6.3: Speaker-aware chunking ----


class TestSpeakerAwareChunking:
    """Test speaker detection and chunk splitting in audio transcripts."""

    def test_detect_speaker_changes(self):
        """Should detect speaker labels like CHAIR:, SENATOR X:, etc."""
        from oracle.ingestion.audio_ingestion import detect_speaker_change

        assert detect_speaker_change("CHAIR: Good morning") is not None
        assert detect_speaker_change("SENATOR Smith: Thank you") is not None
        assert detect_speaker_change("MR. Powell: The economy") is not None
        assert detect_speaker_change("Q: What about inflation?") is not None
        assert detect_speaker_change("Just regular text") is None
        assert detect_speaker_change("") is None

    def test_speaker_aware_chunking(self):
        """Should create new chunks on speaker changes."""
        from oracle.ingestion.audio_ingestion import speaker_aware_chunk

        transcript = (
            "CHAIR: Welcome to today's hearing on monetary policy.\n"
            "We have several important topics to discuss.\n"
            "SENATOR Smith: Thank you, Chair. My question is about inflation.\n"
            "What are the current projections for CPI?\n"
            "MR. Powell: Thank you, Senator. The current projections\n"
            "indicate that inflation is trending toward our 2% target.\n"
            "SENATOR Smith: And what about employment numbers?\n"
        )

        chunks = speaker_aware_chunk(transcript, {"source": "test"})

        assert len(chunks) >= 3  # At least 3 speakers
        # Check that speaker metadata is set
        speakers = [c.metadata.get("speaker", "") for c in chunks]
        assert any("CHAIR" in s for s in speakers)
        assert any("SENATOR" in s for s in speakers)
        assert any("MR. Powell" in s for s in speakers)

    def test_empty_transcript(self):
        """Empty transcript should return empty list."""
        from oracle.ingestion.audio_ingestion import speaker_aware_chunk

        assert speaker_aware_chunk("") == []
        assert speaker_aware_chunk("   ") == []

    def test_no_speakers_fallback(self):
        """Transcript without speakers should still produce chunks."""
        from oracle.ingestion.audio_ingestion import speaker_aware_chunk

        # Long text without speaker markers
        transcript = "\n\n".join([f"Paragraph {i} with enough text to matter." * 10 for i in range(20)])
        chunks = speaker_aware_chunk(transcript, max_tokens=100)
        assert len(chunks) >= 1


# ---- Task 6.5: Poll average calculation ----


class TestPollAverageCalculation:
    """Test polling average computation from stored polls."""

    def test_compute_averages(self):
        """Should compute simple mean of last 5 polls per race/candidate."""
        from oracle.ingestion.polling_scrapers import PollingScraper

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        scraper = PollingScraper(db_path=db_path)

        # Insert test polls
        polls = [
            {"id": f"poll_{i}", "race": "President", "candidate": "Candidate A",
             "pollster": f"Pollster {i}", "date": f"2024-01-{i+1:02d}",
             "sample_size": 1000, "value": 45.0 + i, "margin_of_error": 3.0,
             "source": "test"}
            for i in range(7)
        ]
        scraper.store_polls(polls)

        averages = scraper.compute_polling_averages()

        assert len(averages) == 1  # One race/candidate combo
        avg = averages[0]
        assert avg["race"] == "President"
        assert avg["candidate"] == "Candidate A"
        assert avg["poll_count"] == 5  # Last 5 only
        # Last 5 values: 47, 48, 49, 50, 51 → mean = 49.0
        assert avg["average"] == 49.0

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_empty_polls(self):
        """Should handle empty polls gracefully."""
        from oracle.ingestion.polling_scrapers import PollingScraper

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        scraper = PollingScraper(db_path=db_path)
        averages = scraper.compute_polling_averages()
        assert averages == []

        Path(db_path).unlink(missing_ok=True)

    def test_multiple_candidates(self):
        """Should compute separate averages for each candidate."""
        from oracle.ingestion.polling_scrapers import PollingScraper

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        scraper = PollingScraper(db_path=db_path)

        polls = []
        for candidate, base_value in [("Alice", 48.0), ("Bob", 44.0)]:
            for i in range(3):
                polls.append({
                    "id": f"poll_{candidate}_{i}",
                    "race": "President",
                    "candidate": candidate,
                    "pollster": f"Pollster {i}",
                    "date": f"2024-01-{i+1:02d}",
                    "sample_size": 1000,
                    "value": base_value + i,
                    "margin_of_error": 3.0,
                    "source": "test",
                })

        scraper.store_polls(polls)
        averages = scraper.compute_polling_averages()

        assert len(averages) == 2
        candidates = {a["candidate"]: a["average"] for a in averages}
        assert "Alice" in candidates
        assert "Bob" in candidates

        Path(db_path).unlink(missing_ok=True)


# ---- Task 6.6: Vision pipeline (mock Claude response) ----


class TestVisionPipeline:
    """Test vision pipeline with mocked Claude API responses."""

    @pytest.mark.asyncio
    async def test_analyze_chart_mocked(self):
        """Should parse Claude's JSON response into ChartAnalysis."""
        from oracle.ingestion.vision_ingestion import VisionIngestionPipeline, ChartAnalysis

        pipeline = VisionIngestionPipeline()

        mock_response_text = json.dumps({
            "chart_type": "price_chart",
            "key_trend": "upward momentum",
            "key_values": {"current": 52.3, "peak": 67.1, "trend": "declining"},
            "market_relevance": ["Bitcoin price prediction", "Crypto markets"],
            "summary": "The chart shows Bitcoin price with upward momentum over the past week. Key resistance at 67.1k.",
        })

        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = mock_response_text
        mock_message.content = [mock_content]

        with patch("oracle.ingestion.vision_ingestion.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test_key"

            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = mock_message
                mock_anthropic.return_value = mock_client

                # Mock image download
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.content = b"fake_image_data"
                mock_resp.headers = {"content-type": "image/png"}
                mock_resp.raise_for_status = MagicMock()

                with patch("httpx.AsyncClient") as mock_http:
                    mock_instance = AsyncMock()
                    mock_instance.get = AsyncMock(return_value=mock_resp)
                    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                    mock_instance.__aexit__ = AsyncMock(return_value=False)
                    mock_http.return_value = mock_instance

                    result = await pipeline.analyze_chart("https://example.com/chart.png")

        assert isinstance(result, ChartAnalysis)
        assert result.chart_type == "price_chart"
        assert result.key_trend == "upward momentum"
        assert result.key_values["current"] == 52.3
        assert "Bitcoin price prediction" in result.market_relevance
        assert len(result.summary) > 0

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        """Should return graceful result when no API key configured."""
        from oracle.ingestion.vision_ingestion import VisionIngestionPipeline

        pipeline = VisionIngestionPipeline()
        with patch("oracle.ingestion.vision_ingestion.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            result = await pipeline.analyze_chart("https://example.com/chart.png")
            assert "not configured" in result.summary

    def test_parse_response_with_code_fences(self):
        """Should handle response wrapped in markdown code fences."""
        from oracle.ingestion.vision_ingestion import VisionIngestionPipeline

        pipeline = VisionIngestionPipeline()
        text = '```json\n{"chart_type": "poll_chart", "key_trend": "steady", "key_values": {}, "market_relevance": [], "summary": "A poll chart."}\n```'
        result = pipeline._parse_response(text)
        assert result.chart_type == "poll_chart"

    def test_is_relevant_domain(self):
        """Should filter by relevant financial/political domains."""
        from oracle.ingestion.vision_ingestion import is_relevant_domain

        assert is_relevant_domain("https://pbs.twimg.com/media/abc.jpg") is True
        assert is_relevant_domain("https://www.reuters.com/chart.png") is True
        assert is_relevant_domain("https://randomsite.com/image.jpg") is False


# ---- Scheduler source ordering ----


class TestSchedulerSourceOrdering:
    """Test that the scheduler processes sources in the correct order."""

    def test_source_order(self):
        """Sources should be ordered by criticality: news first, then social, then others."""
        from oracle.ingestion.scheduler import IngestionScheduler

        expected_order = ["news", "reddit", "gov_scrapers", "polling", "audio", "twitter"]
        assert IngestionScheduler.SOURCE_ORDER == expected_order

    def test_schedule_intervals(self):
        """Verify correct scheduling intervals."""
        from oracle.ingestion.scheduler import IngestionScheduler

        assert IngestionScheduler.SCHEDULES["news"] == 900       # 15 min
        assert IngestionScheduler.SCHEDULES["reddit"] == 1800    # 30 min
        assert IngestionScheduler.SCHEDULES["gov_scrapers"] == 21600   # 6 hours
        assert IngestionScheduler.SCHEDULES["polling"] == 43200  # 12 hours
        assert IngestionScheduler.SCHEDULES["audio"] == 86400    # daily

    def test_get_status_initial(self):
        """Initial status should show all sources as idle with no runs."""
        from oracle.ingestion.scheduler import IngestionScheduler

        scheduler = IngestionScheduler()
        status = scheduler.get_status()

        assert status["running"] is False
        for source_name in IngestionScheduler.SOURCE_ORDER:
            source = status["sources"][source_name]
            assert source["last_run"] == "never"
            assert source["doc_count"] == 0
            assert source["status"] == "idle"


# ---- Gov scrapers dedup ----


class TestGovScraperDedup:
    """Test government scraper document deduplication via SQLite."""

    def test_doc_dedup(self):
        """Same document ID should not be marked as new twice."""
        from oracle.ingestion.gov_scrapers import GovScraper

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        scraper = GovScraper(db_path=db_path)

        assert scraper._is_doc_seen("doc_1") is False
        scraper._mark_doc_seen("doc_1", "test", "Test Document")
        assert scraper._is_doc_seen("doc_1") is True

        Path(db_path).unlink(missing_ok=True)
