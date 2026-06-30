"""
Smoke tests for the Redrob AI Candidate Ranking System.

Run with::

    pytest test_smoke.py -v

These tests validate that core modules produce correct output types,
shapes, and value ranges without requiring any pre-computed artifacts
or network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ranker.features import extract_all_features
from ranker.honeypot import detect_honeypot
from ranker.reasoning import build_template_reasoning
from ranker.utils import clip, compute_notice_score, fuzzy_match, parse_date

SAMPLE_PATH = ROOT.parent / "India_runs_data_and_ai_challenge" / "sample_candidates.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_candidates() -> list[dict]:
    """Load sample candidates from the dataset directory."""
    if not SAMPLE_PATH.exists():
        pytest.skip(f"Sample file not found: {SAMPLE_PATH}")
    with open(SAMPLE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def single_candidate(sample_candidates: list[dict]) -> dict:
    """Return the first sample candidate."""
    return sample_candidates[0]


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------


class TestParseDate:
    """Tests for ranker.utils.parse_date."""

    def test_valid_date(self):
        result = parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_none_input(self):
        assert parse_date(None) is None

    def test_empty_string(self):
        assert parse_date("") is None

    def test_invalid_format(self):
        assert parse_date("not-a-date") is None

    def test_partial_date(self):
        assert parse_date("2024-13-01") is None  # Month 13 is invalid


class TestClip:
    """Tests for ranker.utils.clip."""

    def test_within_range(self):
        assert clip(0.5) == 0.5

    def test_below_range(self):
        assert clip(-0.5) == 0.0

    def test_above_range(self):
        assert clip(1.5) == 1.0

    def test_custom_bounds(self):
        assert clip(5.0, low=0.0, high=10.0) == 5.0
        assert clip(-1.0, low=0.0, high=10.0) == 0.0
        assert clip(15.0, low=0.0, high=10.0) == 10.0


class TestFuzzyMatch:
    """Tests for ranker.utils.fuzzy_match."""

    def test_exact_match(self):
        assert fuzzy_match("google", {"google", "meta"}) is True

    def test_substring_match(self):
        assert fuzzy_match("Google LLC", {"google", "meta"}) is True

    def test_no_match(self):
        assert fuzzy_match("startupxyz", {"google", "meta"}) is False

    def test_case_insensitive(self):
        assert fuzzy_match("GOOGLE", {"google"}) is True


class TestNoticeScore:
    """Tests for ranker.utils.compute_notice_score."""

    def test_immediate(self):
        assert compute_notice_score(0) == 1.0

    def test_30_days(self):
        assert compute_notice_score(30) == 1.0

    def test_long_notice(self):
        score = compute_notice_score(120)
        assert 0.3 <= score <= 0.7

    def test_monotonically_decreasing(self):
        scores = [compute_notice_score(d) for d in range(0, 200, 10)]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Notice score increased from day {i*10} to {(i+1)*10}: "
                f"{scores[i]} -> {scores[i+1]}"
            )


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Tests for ranker.features.extract_all_features."""

    def test_returns_dict(self, single_candidate: dict):
        features = extract_all_features(single_candidate)
        assert isinstance(features, dict)

    def test_contains_required_keys(self, single_candidate: dict):
        features = extract_all_features(single_candidate)
        required_keys = [
            "current_title_relevance",
            "yoe_in_ideal_range",
            "num_core_ai_skills",
            "location_fit",
            "activity_decay_score",
            "is_honeypot",
        ]
        for key in required_keys:
            assert key in features, f"Missing feature key: {key}"

    def test_values_are_numeric(self, single_candidate: dict):
        features = extract_all_features(single_candidate)
        for key, value in features.items():
            assert isinstance(value, (int, float)), (
                f"Feature '{key}' is not numeric: {type(value)}"
            )

    def test_bounded_features(self, single_candidate: dict):
        features = extract_all_features(single_candidate)
        bounded_keys = [
            "current_title_relevance",
            "yoe_in_ideal_range",
            "activity_decay_score",
            "location_fit",
            "is_honeypot",
        ]
        for key in bounded_keys:
            val = features[key]
            assert 0.0 <= val <= 1.0, (
                f"Feature '{key}' out of [0, 1] range: {val}"
            )

    def test_consistent_across_calls(self, single_candidate: dict):
        """Deterministic: same input produces same output."""
        f1 = extract_all_features(single_candidate)
        f2 = extract_all_features(single_candidate)
        assert f1 == f2


# ---------------------------------------------------------------------------
# Honeypot detection tests
# ---------------------------------------------------------------------------


class TestHoneypotDetection:
    """Tests for ranker.honeypot.detect_honeypot."""

    def test_returns_tuple(self, single_candidate: dict):
        result = detect_honeypot(single_candidate)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_bool(self, single_candidate: dict):
        is_hp, _ = detect_honeypot(single_candidate)
        assert isinstance(is_hp, bool)

    def test_second_element_is_list(self, single_candidate: dict):
        _, flags = detect_honeypot(single_candidate)
        assert isinstance(flags, list)

    def test_flags_are_strings(self, single_candidate: dict):
        _, flags = detect_honeypot(single_candidate)
        for flag in flags:
            assert isinstance(flag, str), f"Flag is not string: {type(flag)}"

    def test_deterministic(self, single_candidate: dict):
        """Same input always produces same flags."""
        r1 = detect_honeypot(single_candidate)
        r2 = detect_honeypot(single_candidate)
        assert r1[0] == r2[0]
        assert r1[1] == r2[1]


# ---------------------------------------------------------------------------
# Reasoning generation tests
# ---------------------------------------------------------------------------


class TestReasoningGeneration:
    """Tests for ranker.reasoning.build_template_reasoning."""

    def test_returns_string(self, single_candidate: dict):
        reasoning = build_template_reasoning(single_candidate, rank=1, score=0.95)
        assert isinstance(reasoning, str)

    def test_non_empty(self, single_candidate: dict):
        reasoning = build_template_reasoning(single_candidate, rank=1, score=0.95)
        assert len(reasoning) > 30, f"Reasoning too short: {len(reasoning)} chars"

    def test_contains_factual_content(self, single_candidate: dict):
        reasoning = build_template_reasoning(single_candidate, rank=1, score=0.95)
        # Should contain years of experience reference
        assert "year" in reasoning.lower(), "Reasoning should reference experience"

    def test_rank_affects_tone(self, single_candidate: dict):
        """Top-rank reasoning should differ from low-rank reasoning."""
        r1 = build_template_reasoning(single_candidate, rank=1, score=0.99)
        r50 = build_template_reasoning(single_candidate, rank=50, score=0.50)
        r100 = build_template_reasoning(single_candidate, rank=100, score=0.20)
        # They should not all be identical
        assert not (r1 == r50 == r100), "Reasoning should vary by rank tier"

    def test_deterministic(self, single_candidate: dict):
        """Same inputs always produce same reasoning."""
        r1 = build_template_reasoning(single_candidate, rank=5, score=0.90)
        r2 = build_template_reasoning(single_candidate, rank=5, score=0.90)
        assert r1 == r2

    def test_all_sample_candidates(self, sample_candidates: list[dict]):
        """All sample candidates should produce valid reasonings."""
        for i, cand in enumerate(sample_candidates[:5]):
            reasoning = build_template_reasoning(cand, rank=i + 1, score=0.9)
            assert isinstance(reasoning, str)
            assert len(reasoning) > 20, (
                f"Candidate {cand.get('candidate_id', i)}: "
                f"reasoning too short ({len(reasoning)} chars)"
            )
