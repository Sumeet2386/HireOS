"""
Hybrid reasoning generation for the Redrob AI Candidate Ranking System.

Two-layer approach:
1. Rule-based template (always factually correct) — deterministic
2. SLM polish (optional, if time permits) — uses Qwen2.5-0.5B via llama-cpp

The rule-based template is the safety net. SLM polish improves fluency but
any hallucination detected triggers fallback to the template.
"""

from __future__ import annotations

from typing import Any

from .constants import (
    CONSULTING_FIRMS,
    CORE_AI_SKILLS,
    HIGH_SIGNAL_SKILLS,
    TIER1_INDIA_CITIES,
)


def _get_top_relevant_skills(candidate: dict[str, Any], n: int = 3) -> list[str]:
    """Get top N most relevant skills by proficiency."""
    skills = candidate.get("skills", [])
    proficiency_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}

    relevant = []
    for skill in skills:
        name = (skill.get("name") or "").lower()
        if name in CORE_AI_SKILLS or any(
            term in name for term in CORE_AI_SKILLS if len(term) > 4
        ):
            relevant.append((
                skill.get("name", name),
                proficiency_order.get(skill.get("proficiency", "beginner"), 1),
            ))

    relevant.sort(key=lambda x: -x[1])
    return [name for name, _ in relevant[:n]]


def _is_tier1_india(candidate: dict[str, Any]) -> bool:
    """Check if candidate is in a Tier 1 Indian city."""
    profile = candidate.get("profile", {})
    country = (profile.get("country") or "").lower()
    location = (profile.get("location") or "").lower()
    return country == "india" and any(city in location for city in TIER1_INDIA_CITIES)


def _fuzzy_match_set(name: str, known_set: set[str]) -> bool:
    name = name.lower().strip()
    if name in known_set:
        return True
    for known in known_set:
        if known in name or name in known:
            return True
    return False


def build_template_reasoning(
    candidate: dict[str, Any],
    rank: int,
    score: float,
    features: dict[str, float] | None = None,
) -> str:
    """
    Build a deterministic, fact-based reasoning string.

    Always factually correct. Exactly 2 sentences: positives + concerns.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Professional")
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")
    location = profile.get("location", "")

    # ── Sentence 1: Positive signals ──
    parts = []
    parts.append(f"{title} with {yoe:.1f} years of experience")

    # Skills
    top_skills = _get_top_relevant_skills(candidate, n=3)
    if top_skills:
        if len(top_skills) == 1:
            parts.append(f"demonstrating expertise in {top_skills[0]}")
        else:
            skill_str = ", ".join(top_skills[:-1]) + f" and {top_skills[-1]}"
            parts.append(f"demonstrating expertise in {skill_str}")

    # Company signal
    if company:
        companies_in_career = [
            (job.get("company") or "").lower()
            for job in candidate.get("career_history", [])
        ]
        has_product = any(
            _fuzzy_match_set(c, set(["google", "meta", "amazon", "microsoft",
                                      "flipkart", "swiggy", "uber", "netflix"]))
            for c in companies_in_career
        )
        if has_product:
            parts.append("with product company experience")

    # Engagement
    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate > 0.7:
        parts.append(f"strong recruiter engagement ({response_rate:.0%} response rate)")

    sentence1 = ", ".join(parts) + "."

    # ── Sentence 2: Concerns / risks ──
    concerns = []

    notice_days = signals.get("notice_period_days", 0)
    if notice_days > 60:
        concerns.append(f"{notice_days}-day notice period")
    elif notice_days > 30:
        concerns.append(f"moderate {notice_days}-day notice period")

    if not _is_tier1_india(candidate):
        country = (profile.get("country") or "").lower()
        if country != "india":
            concerns.append(f"based outside India ({profile.get('location', 'unknown')})")
        else:
            concerns.append(f"not in a Tier-1 metro ({profile.get('location', 'unknown')})")

    # Consulting background
    companies = [(job.get("company") or "").lower() for job in candidate.get("career_history", [])]
    consulting_ratio = sum(1 for c in companies if _fuzzy_match_set(c, CONSULTING_FIRMS)) / max(1, len(companies))
    if consulting_ratio > 0.7:
        concerns.append("career heavily in IT consulting")

    # Low engagement
    if response_rate < 0.3 and response_rate > 0:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")

    # Title mismatch
    title_lower = title.lower()
    if any(neg in title_lower for neg in ("hr", "marketing", "sales", "accountant", "content writer")):
        concerns.append(f"current title ({title}) not directly in AI engineering")

    if concerns:
        sentence2 = "Considerations: " + "; ".join(concerns) + "."
    else:
        sentence2 = "No significant concerns identified."

    return f"{sentence1} {sentence2}"


def generate_reasoning(
    candidate: dict[str, Any],
    rank: int,
    score: float,
    features: dict[str, float] | None = None,
    slm_model: Any | None = None,
    max_slm_tokens: int = 80,
) -> str:
    """
    Generate per-candidate reasoning using hybrid approach.

    1. Build rule-based template (always correct)
    2. If SLM model is available, polish the template
    3. Check for hallucinations, fallback to template if detected

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.
    rank : int
        Final rank (1-100).
    score : float
        Final score.
    features : dict, optional
        Pre-computed features dict.
    slm_model : optional
        Loaded SLM model (llama-cpp-python Llama instance).
    max_slm_tokens : int
        Max tokens for SLM generation.

    Returns
    -------
    str
        2-sentence reasoning string.
    """
    template = build_template_reasoning(candidate, rank, score, features)

    # If no SLM available, return template directly
    if slm_model is None:
        return template

    # SLM polish attempt
    try:
        prompt = (
            "You are an AI recruiter. Rewrite this candidate assessment "
            "into exactly 2 fluent sentences. Do NOT add any information "
            "not present in the original. Keep it concise and professional.\n\n"
            f"Original: {template}\n\n"
            "Rewritten:"
        )

        response = slm_model.create_completion(
            prompt,
            max_tokens=max_slm_tokens,
            temperature=0.3,
            stop=["\n\n", "Original:", "---"],
        )

        polished = response["choices"][0]["text"].strip()

        # Hallucination check: verify key facts are preserved
        if _hallucination_detected(polished, candidate):
            return template

        # Quality check: must be non-empty and reasonable length
        if not polished or len(polished) < 30 or len(polished) > 500:
            return template

        return polished

    except Exception:
        return template


def _hallucination_detected(text: str, candidate: dict[str, Any]) -> bool:
    """
    Check if SLM output contains hallucinated information.

    Verifies that key facts in the polished text match the candidate record.
    """
    profile = candidate.get("profile", {})
    text_lower = text.lower()

    # Check for fabricated company names not in the candidate's history
    companies_in_record = {
        (job.get("company") or "").lower()
        for job in candidate.get("career_history", [])
    }
    companies_in_record.add((profile.get("current_company") or "").lower())

    # Known big company names that might be hallucinated
    big_companies = ["google", "meta", "amazon", "microsoft", "openai", "deepmind"]
    for company in big_companies:
        if company in text_lower and not any(company in c for c in companies_in_record):
            return True  # Hallucinated a company

    # Check for fabricated degree claims
    degrees_in_record = {
        (edu.get("degree") or "").lower()
        for edu in candidate.get("education", [])
    }
    if "phd" in text_lower or "ph.d" in text_lower:
        if not any("phd" in d or "ph.d" in d or "doctor" in d for d in degrees_in_record):
            return True

    return False
