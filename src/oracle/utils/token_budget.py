"""Token budgeting — enforces context window limits on LLM calls."""

from __future__ import annotations

import functools
from typing import Any, Callable

import structlog
import tiktoken

logger = structlog.get_logger()

MAX_CONTEXT_TOKENS = 4000

_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens in a text string using cl100k_base encoding."""
    return len(_get_encoding().encode(text))


def trim_to_budget(
    texts: list[str],
    budget: int = MAX_CONTEXT_TOKENS,
    strategy: str = "truncate_last",
    weights: list[float] | None = None,
) -> list[str]:
    """Trim a list of text segments to fit within a token budget.

    Strategies:
        truncate_last: Trim the last item first, then second-to-last, etc.
        proportional: Trim all items proportionally to their token counts.
        priority: Trim lowest-weight items first (requires weights param).

    Args:
        texts: List of text segments.
        budget: Maximum total tokens allowed.
        strategy: Trimming strategy name.
        weights: Priority weights for "priority" strategy (higher = keep more).

    Returns:
        Trimmed list of text segments.
    """
    if not texts:
        return texts

    enc = _get_encoding()
    token_counts = [len(enc.encode(t)) for t in texts]
    total = sum(token_counts)

    if total <= budget:
        return list(texts)

    if strategy == "truncate_last":
        return _trim_truncate_last(texts, token_counts, budget, enc)
    elif strategy == "proportional":
        return _trim_proportional(texts, token_counts, budget, total, enc)
    elif strategy == "priority":
        return _trim_priority(texts, token_counts, budget, weights or [], enc)
    else:
        raise ValueError(f"Unknown trim strategy: {strategy}")


def _trim_truncate_last(
    texts: list[str],
    token_counts: list[int],
    budget: int,
    enc: tiktoken.Encoding,
) -> list[str]:
    """Trim from last item backwards."""
    result = list(texts)
    remaining = budget

    # Calculate budget available for each item (front to back)
    for i in range(len(result)):
        if i < len(result) - 1:
            # Reserve tokens for this item, leave rest for following
            remaining -= token_counts[i]
            if remaining < 0:
                # This item needs trimming too
                available = budget - sum(token_counts[:i])
                if available > 0:
                    tokens = enc.encode(result[i])[:available]
                    result[i] = enc.decode(tokens)
                else:
                    result[i] = ""
                # Truncate everything after
                for j in range(i + 1, len(result)):
                    result[j] = ""
                return result
        else:
            # Last item gets whatever budget remains
            available = budget - sum(len(enc.encode(result[j])) for j in range(i))
            if available > 0 and token_counts[i] > available:
                tokens = enc.encode(result[i])[:available]
                result[i] = enc.decode(tokens)
            elif available <= 0:
                result[i] = ""

    return result


def _trim_proportional(
    texts: list[str],
    token_counts: list[int],
    budget: int,
    total: int,
    enc: tiktoken.Encoding,
) -> list[str]:
    """Trim all items proportionally."""
    result = []
    ratio = budget / total

    for text, count in zip(texts, token_counts):
        target = int(count * ratio)
        if target >= count:
            result.append(text)
        elif target > 0:
            tokens = enc.encode(text)[:target]
            result.append(enc.decode(tokens))
        else:
            result.append("")

    return result


def _trim_priority(
    texts: list[str],
    token_counts: list[int],
    budget: int,
    weights: list[float],
    enc: tiktoken.Encoding,
) -> list[str]:
    """Trim lowest-priority items first."""
    # Default equal weights if not enough provided
    while len(weights) < len(texts):
        weights.append(1.0)

    result = list(texts)
    current_total = sum(token_counts)
    excess = current_total - budget

    # Sort indices by weight (ascending — trim lowest first)
    indices_by_priority = sorted(range(len(texts)), key=lambda i: weights[i])

    for idx in indices_by_priority:
        if excess <= 0:
            break

        count = len(enc.encode(result[idx]))
        if count <= excess:
            # Remove entirely
            excess -= count
            result[idx] = ""
        else:
            # Trim to fit
            keep = count - excess
            if keep > 0:
                tokens = enc.encode(result[idx])[:keep]
                result[idx] = enc.decode(tokens)
            else:
                result[idx] = ""
            excess = 0

    return result


def wrap_with_budget(
    prompt_parts: dict[str, str],
    budget: int = MAX_CONTEXT_TOKENS,
) -> str:
    """Assemble a prompt from named parts, respecting token budget.

    Parts are assembled in dict order. If total exceeds budget,
    later parts are truncated first.

    Args:
        prompt_parts: Ordered dict of part_name → text.
        budget: Maximum total tokens.

    Returns:
        Assembled prompt string.
    """
    keys = list(prompt_parts.keys())
    texts = [prompt_parts[k] for k in keys]

    trimmed = trim_to_budget(texts, budget=budget, strategy="truncate_last")

    parts = []
    for key, text in zip(keys, trimmed):
        if text:
            parts.append(text)

    return "\n\n".join(parts)


def budget_aware(max_tokens: int = MAX_CONTEXT_TOKENS) -> Callable:
    """Decorator that enforces token budget on a method's string arguments.

    Trims the longest string argument if total tokens exceed max_tokens.
    Works with both sync and async methods.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            _enforce_budget(args, kwargs, max_tokens)
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            _enforce_budget(args, kwargs, max_tokens)
            return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _enforce_budget(
    args: tuple, kwargs: dict[str, Any], max_tokens: int
) -> None:
    """Trim string kwargs to fit within token budget."""
    enc = _get_encoding()
    str_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, str)}

    if not str_kwargs:
        return

    total = sum(len(enc.encode(v)) for v in str_kwargs.values())
    if total <= max_tokens:
        return

    # Sort by length (descending) — trim longest first
    sorted_keys = sorted(str_kwargs, key=lambda k: len(enc.encode(str_kwargs[k])), reverse=True)

    excess = total - max_tokens
    for key in sorted_keys:
        if excess <= 0:
            break
        tokens = enc.encode(kwargs[key])
        if len(tokens) > excess:
            keep = len(tokens) - excess
            kwargs[key] = enc.decode(tokens[:keep])
            excess = 0
        else:
            excess -= len(tokens)
            kwargs[key] = ""

    logger.info("token_budget.trimmed", original=total, budget=max_tokens)
