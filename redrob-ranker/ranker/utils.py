"""
Shared utility functions for the Redrob AI Candidate Ranking System.

Consolidates common operations used across multiple modules:
- Date parsing
- Fuzzy string matching
- Value clamping
- Candidate data loading (JSON / JSONL)
- Honeypot flag classification
"""

from __future__ import annotations

import enum
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Honeypot flag categories (replaces fragile string-prefix matching)
# ---------------------------------------------------------------------------
class HoneypotCategory(enum.Enum):
    """Enumeration of honeypot detection categories.

    Used to classify flags without relying on string-prefix parsing.
    """

    TIMELINE_IMPOSSIBLE = "TIMELINE_IMPOSSIBLE"
    SKILL_NOT_ENTAILED = "SKILL_NOT_ENTAILED"
    SEMANTIC_CONTRADICTION = "SEMANTIC_CONTRADICTION"
    MATURITY_IMPOSSIBLE = "MATURITY_IMPOSSIBLE"
    HEAVY_OVERLAP_TIMELINE = "HEAVY_OVERLAP_TIMELINE"
    KEYWORD_STUFFER = "KEYWORD_STUFFER"
    SUSPICIOUS_JUNIOR = "SUSPICIOUS_JUNIOR"
    FICTIONAL_COMPANY = "FICTIONAL_COMPANY"


def parse_date(value: str | None) -> date | None:
    """Parse a YYYY-MM-DD string to a ``date`` object.

    Parameters
    ----------
    value : str or None
        Date string in ISO format, or ``None``.

    Returns
    -------
    date or None
        Parsed date, or ``None`` on failure.
    """
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* to the inclusive range [*low*, *high*].

    Parameters
    ----------
    value : float
        The value to clamp.
    low : float
        Lower bound (default 0.0).
    high : float
        Upper bound (default 1.0).

    Returns
    -------
    float
        Clamped value.
    """
    return max(low, min(high, value))


def fuzzy_match(name: str, known_set: set[str]) -> bool:
    """Check whether *name* matches any entry in *known_set*.

    Matching is case-insensitive and allows substring containment in
    either direction (i.e. ``known in name`` or ``name in known``).

    Parameters
    ----------
    name : str
        The string to test (e.g. a company name).
    known_set : set[str]
        Reference set of known lowercase strings.

    Returns
    -------
    bool
        ``True`` if a match is found.
    """
    normalised = name.lower().strip()
    if normalised in known_set:
        return True
    for known in known_set:
        if known in normalised or normalised in known:
            return True
    return False


def load_candidates(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load candidate records from a JSON array or JSONL file.

    Supports two formats:
    - **JSON array**: file starts with ``[`` and contains a list of objects.
    - **JSONL**: one JSON object per line.

    Records are streamed line-by-line for JSONL to avoid loading the
    entire file into memory (critical for the 487 MB production dataset).

    Parameters
    ----------
    path : str or Path
        Filesystem path to the candidates file.

    Returns
    -------
    dict[str, dict]
        Mapping of ``candidate_id`` to full candidate record.
    """
    path = Path(path)
    candidates: dict[str, dict[str, Any]] = {}

    with open(path, "r", encoding="utf-8") as fh:
        first_char = fh.read(1)
        fh.seek(0)

        if first_char == "[":
            # JSON array -- must read entire file
            for candidate in json.load(fh):
                cid = candidate.get("candidate_id")
                if cid is None:
                    logger.warning("Skipping record without candidate_id")
                    continue
                candidates[cid] = candidate
        else:
            # JSONL -- stream line by line
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                    cid = candidate.get("candidate_id")
                    if cid is None:
                        logger.warning(
                            "Skipping record without candidate_id at line %d",
                            line_num,
                        )
                        continue
                    candidates[cid] = candidate
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSON at line %d: %s", line_num, exc
                    )

    logger.info("Loaded %d candidates from %s", len(candidates), path.name)
    return candidates


def extract_honeypot_flag_features(flags: list[str]) -> dict[str, float]:
    """Derive numeric feature columns from a list of honeypot flag strings.

    Centralises the repeated pattern of checking flag prefixes that was
    previously duplicated in ``features.py``, ``rank.py``, and ``app.py``.

    Parameters
    ----------
    flags : list[str]
        Raw flag strings produced by ``honeypot.detect_honeypot``.

    Returns
    -------
    dict[str, float]
        Feature dict with keys ``honeypot_flag_count``,
        ``has_fictional_company``, and ``has_maturity_impossible``.
    """
    return {
        "honeypot_flag_count": float(len(flags)),
        "has_fictional_company": (
            1.0
            if any(f.startswith(HoneypotCategory.FICTIONAL_COMPANY.value) for f in flags)
            else 0.0
        ),
        "has_maturity_impossible": (
            1.0
            if any(f.startswith(HoneypotCategory.MATURITY_IMPOSSIBLE.value) for f in flags)
            else 0.0
        ),
    }


def compute_notice_score(notice_days: float) -> float:
    """Piecewise-linear notice-period score aligned with JD language.

    Score mapping:
    - 0-30 days  -> 1.0   (JD-preferred)
    - 30-45 days -> 0.925
    - 45-60 days -> 0.85
    - 60-90 days -> 0.70
    - 90-120 days -> 0.55
    - 120+ days  -> decays to 0.30

    Parameters
    ----------
    notice_days : float
        Candidate's notice period in days.

    Returns
    -------
    float
        Score in approximately [0.30, 1.00].
    """
    if notice_days <= 30:
        return 1.0
    if notice_days <= 45:
        return 1.0 - 0.005 * (notice_days - 30)
    if notice_days <= 60:
        return 0.925 - 0.005 * (notice_days - 45)
    if notice_days <= 90:
        return 0.85 - 0.005 * (notice_days - 60)
    if notice_days <= 120:
        return 0.70 - 0.005 * (notice_days - 90)
    return max(0.3, 0.55 - 0.003 * (notice_days - 120))
