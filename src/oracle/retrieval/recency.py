"""Recency weighting with exponential decay on publication_date."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from oracle.models import RetrievalResult


def apply_recency_weight(
    results: list[RetrievalResult],
    decay_days: float = 7.0,
    reference_time: datetime | None = None,
) -> list[RetrievalResult]:
    """Apply exponential decay weighting based on publication_date.

    Score multiplier = exp(-age_days / decay_days)

    A decay_days of 7 means:
    - 1 day old: 0.87x
    - 3 days old: 0.65x
    - 7 days old: 0.37x
    - 14 days old: 0.14x
    - 30 days old: 0.01x
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    for result in results:
        pub_date_str = result.metadata.get("publication_date")
        if not pub_date_str:
            # No date → apply a mild penalty (treat as ~2 weeks old)
            result.score *= 0.15
            continue

        try:
            if isinstance(pub_date_str, str):
                # Handle multiple datetime formats
                for fmt in (
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S.%fZ",
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%d",
                ):
                    try:
                        pub_date = datetime.strptime(pub_date_str, fmt)
                        if pub_date.tzinfo is None:
                            pub_date = pub_date.replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
                else:
                    result.score *= 0.15
                    continue
            elif isinstance(pub_date_str, datetime):
                pub_date = pub_date_str
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=timezone.utc)
            else:
                result.score *= 0.15
                continue

            age_days = (reference_time - pub_date).total_seconds() / 86400.0
            if age_days < 0:
                age_days = 0  # Future dates get full weight

            decay = math.exp(-age_days / decay_days)
            result.score *= decay

        except Exception:
            result.score *= 0.15

    return results
