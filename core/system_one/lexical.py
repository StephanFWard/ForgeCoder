"""Deterministic lexical weighting for local (free) System One decisions.

Jev's value is not the text it does not generate — it is that *every* answer
comes back as a probability distribution the caller can threshold, sort, and
combine. A paid model is not the only way to produce that shape. This module
produces it from the state itself, with arithmetic that is:

* **deterministic** — no randomness, no time, no process-dependent ordering.
  Python's ``hash()`` is salted per process for strings, so it is never used;
  every mapping here is an insertion-ordered ``dict`` or a ``sorted`` list.
* **weighted** — matches are weighted by inverse document frequency over the
  candidate options, so a token that distinguishes one option counts for more
  than a token every option already contains (the fixed per-token credit in
  :mod:`core.retrieval.ranking` cannot express that).
* **code-aware** — identifiers are also split on camelCase / snake_case /
  kebab-case, so ``getUser`` matches ``get_user`` the way a reader expects.
* **free** — no model, no network, no quota.

The same functions power :class:`core.system_one.backends.DeterministicBackend`
and the weighted reranker in :mod:`core.system_one.rerank`.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Tokens that carry no discriminating weight in code decisions. Kept small on
# purpose: this is a scoring prior, not a linguistic stopword list.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "if", "in", "into", "is", "it", "its", "of", "on", "or",
    "should", "so", "that", "the", "their", "then", "there", "these", "this",
    "to", "was", "were", "what", "when", "which", "with", "would",
})


def _stem(token: str) -> str:
    """Conservative deterministic stem: plural and -ing/-ed suffixes only.

    ``failed``/``fails`` -> ``fail``, ``policies`` -> ``policy``. Deliberately
    not a linguistic stemmer: no dictionary, no rules table, nothing that could
    change between runs or Python versions. ``cached`` -> ``cach`` is accepted
    because the mapping is stable, which is what the weights need.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    return token


def _identifier_parts(token: str) -> list[str]:
    """Split an identifier into sub-words: ``getUserById`` -> get/user/by/id."""
    parts: list[str] = []
    for chunk in _CAMEL_RE.split(token):
        for piece in re.split(r"[_\-]+", chunk):
            if piece and piece.lower() not in STOPWORDS:
                parts.append(_stem(piece.lower()))
    return parts


def tokenize(text: str) -> list[str]:
    """Deterministically tokenize ``text``, de-duplicated, order preserved.

    Emits whole identifiers lowercased plus their sub-words, so both the joined
    and split spellings of an identifier are matchable, and both are lightly
    stemmed so ``failed`` and ``fail`` meet.
    """
    seen: dict[str, None] = {}
    for raw in _WORD_RE.findall(text):
        lowered = raw.lower()
        if lowered in STOPWORDS or len(lowered) < 2:
            continue
        seen.setdefault(_stem(lowered), None)
        for part in _identifier_parts(raw):
            seen.setdefault(part, None)
    return list(seen)


def idf_weights(documents: Sequence[str]) -> dict[str, float]:
    """Inverse document frequency over a small candidate corpus.

    ``idf = log(1 + (N + 1) / (df + 1))`` — always positive, so a token that
    occurs everywhere still contributes, and a token unique to one option
    contributes the most. Deterministic: document order only affects the
    iteration order of an insertion-ordered dict.
    """
    total = max(1, len(documents))
    df: dict[str, int] = {}
    for doc in documents:
        for token in tokenize(doc):
            df[token] = df.get(token, 0) + 1
    return {token: math.log(1.0 + (total + 1.0) / (count + 1.0)) for token, count in df.items()}


def _default_idf(documents: Sequence[str]) -> float:
    """Weight for a state token that appears in no option (df == 0).

    Capped at the most informative *observed* token: a state word the options
    never mention should not outweigh the words the options are competing over,
    which is what an uncapped ``df == 0`` estimate would do.
    """
    weights = idf_weights(list(documents))
    return max(weights.values()) if weights else math.log(2.0)


def similarity_weights(state_tokens: Iterable[str], candidates: Sequence[str]) -> list[float]:
    """IDF-weighted cosine similarity between the state and each candidate.

    Cosine — not coverage — because a decision is about *which* candidate the
    state points at, and coverage divides by the whole state, which penalises a
    short option that mentions exactly the one distinctive token. Similarity
    asks the right question: how much of this candidate's own vocabulary is the
    state talking about, weighted so common tokens cannot carry the answer.

    Deterministic: candidate order in, same floats out, no hashing of strings.
    """
    tokens = list(dict.fromkeys(state_tokens))
    if not tokens or not candidates:
        return [0.0] * len(candidates)

    weights = idf_weights(list(candidates))
    fallback = _default_idf(list(candidates))
    state_set = set(tokens)
    state_norm = math.sqrt(sum(weights.get(token, fallback) ** 2 for token in tokens)) or 1.0

    similarities: list[float] = []
    for candidate in candidates:
        candidate_tokens = list(dict.fromkeys(tokenize(candidate)))
        candidate_norm = math.sqrt(
            sum(weights.get(token, fallback) ** 2 for token in candidate_tokens)
        ) or 1.0
        dot = sum(
            weights.get(token, fallback) ** 2
            for token in candidate_tokens
            if token in state_set
        )
        similarities.append(min(1.0, dot / (state_norm * candidate_norm)))
    return similarities


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax; ``temperature`` sharpens (>0) or flattens."""
    if not values:
        return []
    scale = max(1e-6, float(temperature))
    peak = max(values)
    exps = [math.exp((value - peak) / scale) for value in values]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def logistic(value: float) -> float:
    """Deterministic sigmoid used to turn evidence into a p(yes)."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def normalize(values: Sequence[float]) -> list[float]:
    """Min-max normalise to 0..1; a constant series becomes all zeros.

    Constant — rather than all-ones — because "every candidate scored the same"
    is an absence of signal, and the blend in :mod:`core.system_one.rerank`
    should then defer to the other term.
    """
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def clamp01(value: float) -> float:
    """Clamp to the 0..1 range the API contract requires."""
    return max(0.0, min(1.0, value))


__all__ = [
    "STOPWORDS",
    "clamp01",
    "idf_weights",
    "logistic",
    "normalize",
    "similarity_weights",
    "softmax",
    "tokenize",
]
