"""Audio transcription pipeline — YouTube/podcast audio → Whisper → speaker-aware chunks → Qdrant."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from oracle.config import settings
from oracle.ingestion.chunker import Chunk
from oracle.knowledge.qdrant_client import QdrantManager

logger = structlog.get_logger()

# Speaker detection patterns
SPEAKER_PATTERNS = [
    re.compile(r"^(CHAIR|CHAIRMAN|CHAIRWOMAN|CHAIRPERSON)\s*[:\-]", re.IGNORECASE),
    re.compile(r"^(SENATOR|REPRESENTATIVE|CONGRESSMAN|CONGRESSWOMAN)\s+\w+\s*[:\-]", re.IGNORECASE),
    re.compile(r"^(SPEAKER|MR\.|MS\.|MRS\.|DR\.)\s+\w+\s*[:\-]", re.IGNORECASE),
    re.compile(r"^(SECRETARY|GOVERNOR|DIRECTOR|COMMISSIONER)\s+\w+\s*[:\-]", re.IGNORECASE),
    re.compile(r"^(Q|A)\s*[:\-]", re.IGNORECASE),
    re.compile(r"^([A-Z][A-Z\s\.]{2,})\s*:", re.MULTILINE),
]

DEFAULT_AUDIO_SOURCES = [
    {"name": "Federal Reserve", "type": "youtube", "url": "https://www.youtube.com/@FederalReserve"},
    {"name": "Senate Committee", "type": "youtube", "url": "https://www.youtube.com/c/SenateCommittee"},
    {"name": "Seeking Alpha Earnings", "type": "rss", "url": "https://seekingalpha.com/feed"},
]


def detect_speaker_change(line: str) -> str | None:
    """Detect if a line starts with a speaker identifier. Returns speaker name or None."""
    for pattern in SPEAKER_PATTERNS:
        match = pattern.match(line.strip())
        if match:
            return match.group(0).rstrip(":- ").strip()
    return None


def speaker_aware_chunk(
    transcript: str,
    source_metadata: dict[str, Any] | None = None,
    max_tokens: int = 500,
) -> list[Chunk]:
    """Split transcript into chunks based on speaker changes.

    Each chunk is tagged with the speaker identity.
    Falls back to paragraph-based splitting if no speakers detected.
    """
    if not transcript.strip():
        return []

    metadata = source_metadata or {}
    lines = transcript.split("\n")
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_speaker: str = "unknown"

    def _flush() -> None:
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            return
        chunk_meta = {**metadata, "speaker": current_speaker}
        chunks.append(Chunk(
            text=text,
            index=len(chunks),
            metadata=chunk_meta,
            token_count=len(text) // 4,
        ))

    for line in lines:
        speaker = detect_speaker_change(line)
        if speaker and speaker != current_speaker:
            # Flush current chunk on speaker change
            _flush()
            current_lines = [line]
            current_speaker = speaker
        else:
            current_lines.append(line)
            # Also enforce max token limit
            current_text = "\n".join(current_lines)
            if len(current_text) // 4 > max_tokens:
                _flush()
                current_lines = []

    _flush()

    # If no speaker changes were detected, fall back to paragraph splitting
    if len(chunks) <= 1 and len(transcript) > max_tokens * 4:
        chunks = []
        paragraphs = transcript.split("\n\n")
        current_text = ""
        for para in paragraphs:
            if len(current_text + para) // 4 > max_tokens and current_text:
                chunks.append(Chunk(
                    text=current_text.strip(),
                    index=len(chunks),
                    metadata=metadata,
                    token_count=len(current_text) // 4,
                ))
                current_text = para
            else:
                current_text = f"{current_text}\n\n{para}" if current_text else para
        if current_text.strip():
            chunks.append(Chunk(
                text=current_text.strip(),
                index=len(chunks),
                metadata=metadata,
                token_count=len(current_text) // 4,
            ))

    return chunks


class AudioIngestionPipeline:
    """Pipeline for downloading audio, transcribing with Whisper, and ingesting."""

    def __init__(
        self,
        qdrant: QdrantManager | None = None,
        db_path: str = "oracle_audio.db",
    ) -> None:
        self._qdrant = qdrant
        self._db_path = db_path
        self._whisper_model = None
        self._sources = DEFAULT_AUDIO_SOURCES
        self._setup_db()

    def _setup_db(self) -> None:
        """Initialize SQLite for tracking processed URLs."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_urls (
                url TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                processed_at TEXT NOT NULL,
                transcript_length INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def _is_processed(self, url: str) -> bool:
        """Check if a URL has already been processed."""
        conn = sqlite3.connect(self._db_path)
        row = conn.execute("SELECT 1 FROM processed_urls WHERE url = ?", (url,)).fetchone()
        conn.close()
        return row is not None

    def _mark_processed(self, url: str, title: str = "", transcript_length: int = 0) -> None:
        """Mark a URL as processed."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT OR REPLACE INTO processed_urls (url, title, processed_at, transcript_length) VALUES (?, ?, ?, ?)",
            (url, title, datetime.now(timezone.utc).isoformat(), transcript_length),
        )
        conn.commit()
        conn.close()

    def _load_whisper(self) -> Any:
        """Lazy-load Whisper model."""
        if self._whisper_model is None:
            import whisper
            model_size = settings.whisper_model_size or "base"
            self._whisper_model = whisper.load_model(model_size)
            logger.info("audio.whisper_loaded", model_size=model_size)
        return self._whisper_model

    def fetch_audio(self, url: str) -> str | None:
        """Download audio from a URL using yt-dlp. Returns path to temp audio file."""
        try:
            tmp_dir = tempfile.mkdtemp(prefix="oracle_audio_")
            output_path = str(Path(tmp_dir) / "audio.%(ext)s")
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--extract-audio",
                    "--audio-format", "mp3",
                    "--audio-quality", "5",
                    "--output", output_path,
                    "--no-playlist",
                    "--quiet",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                logger.error("audio.download_failed", url=url, stderr=result.stderr[:200])
                return None

            # Find the downloaded file
            for f in Path(tmp_dir).iterdir():
                if f.suffix in (".mp3", ".m4a", ".wav", ".opus", ".webm"):
                    logger.info("audio.downloaded", url=url, path=str(f))
                    return str(f)

            logger.error("audio.no_output_file", url=url)
            return None
        except subprocess.TimeoutExpired:
            logger.error("audio.download_timeout", url=url)
            return None
        except FileNotFoundError:
            logger.error("audio.ytdlp_not_found", msg="yt-dlp not installed")
            return None

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file using OpenAI Whisper (local model)."""
        model = self._load_whisper()
        result = model.transcribe(audio_path)
        transcript = result.get("text", "")
        logger.info("audio.transcribed", length=len(transcript), path=audio_path)
        return transcript

    async def process_url(self, url: str, title: str = "") -> dict[str, Any]:
        """Full pipeline: download → transcribe → chunk → ingest."""
        if self._is_processed(url):
            logger.info("audio.already_processed", url=url)
            return {"status": "skipped", "reason": "already_processed"}

        audio_path = self.fetch_audio(url)
        if not audio_path:
            return {"status": "error", "reason": "download_failed"}

        transcript = self.transcribe(audio_path)
        if not transcript:
            return {"status": "error", "reason": "transcription_empty"}

        # Chunk with speaker awareness
        source_metadata = {
            "source_url": url,
            "title": title,
            "source": "audio_transcript",
            "publication_date": datetime.now(timezone.utc).isoformat(),
        }
        chunks = speaker_aware_chunk(transcript, source_metadata)

        # Ingest to Qdrant
        ingested = 0
        if self._qdrant and chunks:
            from oracle.knowledge.embeddings import EmbeddingService
            embedder = EmbeddingService.get_instance()

            texts = [c.text for c in chunks]
            embeddings = embedder.embed(texts)
            ids = []
            vectors = []
            payloads = []

            for chunk, embedding in zip(chunks, embeddings):
                chunk_id = hashlib.md5(f"audio:{url}:{chunk.index}".encode()).hexdigest()
                ids.append(chunk_id)
                vectors.append(embedding)
                payloads.append({
                    "text": chunk.text,
                    "source_url": url,
                    "publication_date": source_metadata["publication_date"],
                    "author": chunk.metadata.get("speaker", "unknown"),
                    "source_authority_score": 0.85,
                    "entity_ids": [],
                    "market_ids": [],
                    "source": "audio_transcript",
                    "speaker": chunk.metadata.get("speaker", "unknown"),
                })

            await self._qdrant.upsert_chunks(
                collection="transcripts",
                ids=ids,
                vectors=vectors,
                payloads=payloads,
            )
            ingested = len(ids)

        self._mark_processed(url, title=title, transcript_length=len(transcript))

        # Clean up temp file
        try:
            Path(audio_path).unlink(missing_ok=True)
            Path(audio_path).parent.rmdir()
        except OSError:
            pass

        return {
            "status": "success",
            "chunks": len(chunks),
            "ingested": ingested,
            "transcript_length": len(transcript),
        }
