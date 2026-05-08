"""
LLM Provider Abstraction
------------------------
Unified interface over Anthropic Claude, OpenAI, and a heuristic fallback.
Selection order:
  1. Explicit `LLM_PROVIDER` env var: "anthropic" | "openai" | "heuristic"
  2. Auto: prefer anthropic if ANTHROPIC_API_KEY is set, else openai if
     OPENAI_API_KEY is set, else heuristic.
  3. If the chosen provider's SDK or key is unavailable at call time,
     the call falls back to heuristic and emits a one-time warning.

Public surface:
  - get_provider() -> Provider
  - Provider.summarize(text: str, max_words: int) -> str
  - Provider.extract_entities(text: str) -> list[dict]
  - Provider.name -> str
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from typing import Protocol

# ---------------------------------------------------------------------------
# Heuristic primitives (always available)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "with", "by", "near", "from", "into", "as", "is", "was", "were", "be",
    "been", "being", "that", "this", "these", "those", "it", "its", "their",
    "they", "his", "her", "he", "she", "we", "our", "have", "has", "had",
    "but", "not", "no", "so", "if", "then", "than", "while", "during",
    "over", "between", "through",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z]+", (text or "").lower())
            if t not in STOPWORDS and len(t) > 2]


def heuristic_summarize(text: str, max_words: int = 18) -> str:
    if not text:
        return ""
    first = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    words = first.split()
    if len(words) <= max_words:
        return first
    return " ".join(words[:max_words]).rstrip(",.;:") + "..."


def heuristic_extract_entities(text: str) -> list[dict]:
    """Naive capitalized-token extraction with type guessing."""
    candidates = re.findall(r"\b([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)\b",
                            text or "")
    seen: list[dict] = []
    for c in candidates:
        if c.lower() in {"the", "a", "an"}:
            continue
        guess = "Location" if any(loc in c for loc in
                                  ["Bamako", "Gao", "Mopti", "Niger",
                                   "Border", "Corridor", "Tower",
                                   "Compound", "Crossing"]) else "Person"
        if not any(s["text"] == c for s in seen):
            seen.append({"text": c, "type": guess})
    return seen


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

class Provider(Protocol):
    name: str
    def summarize(self, text: str, max_words: int = 18) -> str: ...
    def extract_entities(self, text: str) -> list[dict]: ...


_FALLBACK_NOTICE_PRINTED = False


def _warn_fallback(reason: str) -> None:
    global _FALLBACK_NOTICE_PRINTED
    if not _FALLBACK_NOTICE_PRINTED:
        print(f"[llm] falling back to heuristic provider: {reason}",
              file=sys.stderr)
        _FALLBACK_NOTICE_PRINTED = True


# ---------------------------------------------------------------------------
# Heuristic provider (always works, no deps)
# ---------------------------------------------------------------------------

class HeuristicProvider:
    name = "heuristic"

    def summarize(self, text: str, max_words: int = 18) -> str:
        return heuristic_summarize(text, max_words)

    def extract_entities(self, text: str) -> list[dict]:
        return heuristic_extract_entities(text)


# ---------------------------------------------------------------------------
# Anthropic Claude provider
# ---------------------------------------------------------------------------

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. "
                "Install with: pip install anthropic"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic()
        self._model = model or os.environ.get(
            "ANTHROPIC_MODEL", "claude-haiku-4-5"
        )
        self._fallback = HeuristicProvider()

    def summarize(self, text: str, max_words: int = 18) -> str:
        if not text:
            return ""
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=120,
                system=(
                    "You are an intelligence analyst summarizer. Produce a "
                    "single-sentence neutral summary of the provided report "
                    f"excerpt in at most {max_words} words. No preamble, "
                    "no quotes, just the summary."
                ),
                messages=[{"role": "user", "content": text}],
            )
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    return block.text.strip()
            return self._fallback.summarize(text, max_words)
        except Exception as e:
            _warn_fallback(f"anthropic summarize failed: {e}")
            return self._fallback.summarize(text, max_words)

    def extract_entities(self, text: str) -> list[dict]:
        if not text:
            return []
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=400,
                system=(
                    "Extract named entities from the provided intelligence "
                    "report excerpt. Return ONLY valid JSON in the shape: "
                    '{"entities": [{"text": "...", "type": "Person|Group|'
                    'Location|Organization"}]}. No prose, no code fences.'
                ),
                messages=[{"role": "user", "content": text}],
            )
            raw = ""
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    raw = block.text.strip()
                    break
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)
            return data.get("entities", [])
        except Exception as e:
            _warn_fallback(f"anthropic extract_entities failed: {e}")
            return self._fallback.extract_entities(text)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai SDK not installed. Install with: pip install openai"
            ) from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI()
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._fallback = HeuristicProvider()

    def summarize(self, text: str, max_words: int = 18) -> str:
        if not text:
            return ""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": (
                        "You are an intelligence analyst summarizer. Produce "
                        "a single-sentence neutral summary in at most "
                        f"{max_words} words. No preamble, no quotes."
                    )},
                    {"role": "user", "content": text},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            _warn_fallback(f"openai summarize failed: {e}")
            return self._fallback.summarize(text, max_words)

    def extract_entities(self, text: str) -> list[dict]:
        if not text:
            return []
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": (
                        "Extract named entities. Return JSON: "
                        '{"entities": [{"text": "...", "type": '
                        '"Person|Group|Location|Organization"}]}.'
                    )},
                    {"role": "user", "content": text},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            return data.get("entities", [])
        except Exception as e:
            _warn_fallback(f"openai extract_entities failed: {e}")
            return self._fallback.extract_entities(text)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def get_provider() -> Provider:
    """Resolve the active provider per env config."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit == "heuristic":
        return HeuristicProvider()
    if explicit == "anthropic":
        try:
            return AnthropicProvider()
        except Exception as e:
            _warn_fallback(str(e))
            return HeuristicProvider()
    if explicit == "openai":
        try:
            return OpenAIProvider()
        except Exception as e:
            _warn_fallback(str(e))
            return HeuristicProvider()

    # Auto-detect
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicProvider()
        except Exception as e:
            _warn_fallback(str(e))
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIProvider()
        except Exception as e:
            _warn_fallback(str(e))
    return HeuristicProvider()
