"""Clock 2 scaffold: interfaces only. Do NOT implement scoring here yet.

A future session plugs in (1) a news source producing ``NewsItem`` records and (2) one or
more ``DatedScorer`` implementations (e.g. LLMs with known training cutoffs). The planned
tests are described in docs/METHODOLOGY.md ("Clock 2").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class NewsItem:
    id: str
    ticker: str
    published_at: datetime  # when the text became public (the item's own PIT timestamp)
    headline: str
    body: str
    source: str


@runtime_checkable
class DatedScorer(Protocol):
    """A text scorer whose knowledge has a date.

    ``knowledge_cutoff`` is the last date of the scorer's training data. A PIT-honest
    backtest may only use ``score(text, as_of)`` when ``knowledge_cutoff <= as_of``;
    otherwise the model may "remember" the outcome (Clock 2 look-ahead).
    """

    model_id: str
    knowledge_cutoff: date

    def score(self, text: str, as_of: date) -> float:
        """Sentiment-style score in [-1, 1] for ``text`` evaluated as of ``as_of``."""
        ...


class NotImplementedScorer:
    """Placeholder that satisfies the protocol and refuses to score."""

    def __init__(self, model_id: str = "stub", knowledge_cutoff: date = date(1900, 1, 1)):
        self.model_id = model_id
        self.knowledge_cutoff = knowledge_cutoff

    def score(self, text: str, as_of: date) -> float:
        raise NotImplementedError("Clock 2 scoring is intentionally not implemented in this pass.")


def is_pit_usable(scorer: DatedScorer, as_of: date) -> bool:
    """True if the scorer's knowledge ends on/before ``as_of`` (no model look-ahead)."""
    return scorer.knowledge_cutoff <= as_of
