"""
Feature engineering for the Redrob AI Candidate Ranking System.

Extracts ~40 structured features from raw candidate JSON for use by the
LightGBM LTR model. Features span 6 categories:

1. Semantic (cosine similarity, BM25 — computed online, stubs here)
2. Structural (YoE, tenure, title relevance, company type)
3. Skill (core AI skill count, proficiency, entailment rate, assessment gap)
4. Behavioral (activity decay, response rate, notice period, etc.)
5. Location (India, Tier-1 city, relocation willingness)
6. Honeypot (boolean flags from honeypot.py)
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from .constants import (
    ADJACENT_SIGNAL_TITLES,
    CONSULTING_FIRMS,
    CORE_AI_SKILLS,
    FICTIONAL_COMPANIES,
    HIGH_SIGNAL_SKILLS,
    HIGH_SIGNAL_TITLES,
    NEGATIVE_TITLE_PATTERNS,
    NON_TECHNICAL_TITLES,
    PRODUCT_BRANDS,
    PRODUCT_INDUSTRY_KEYWORDS,
    REFERENCE_DATE,
    SKILL_SYNONYMS,
    TECH_RELEASE_YEARS,
    TIER1_INDIA_CITIES,
)
from .honeypot import detect_honeypot


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _fuzzy_match(company_name: str, known_set: set[str]) -> bool:
    """Check if a company name matches any known company (case-insensitive, fuzzy)."""
    name = company_name.lower().strip()
    if name in known_set:
        return True
    # Check if any known name is a substring or vice versa
    for known in known_set:
        if known in name or name in known:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction functions
# ──────────────────────────────────────────────────────────────────────────────

def extract_structural_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Extract structural / career features."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    yoe = profile.get("years_of_experience", 0.0)

    # YoE ideal range: peaks at 7, smooth decay outside 5-9
    yoe_in_ideal_range = _clip(max(0.0, 1.0 - 0.15 * abs(yoe - 7.0)))

    # Number of career entries
    num_career_entries = len(career)

    # Average tenure
    durations = [job.get("duration_months", 0) for job in career if job.get("duration_months")]
    avg_tenure_months = sum(durations) / len(durations) if durations else 0.0

    # Title chaser detection
    is_title_chaser = 1.0 if avg_tenure_months < 18 and num_career_entries >= 3 else 0.0

    # Current title relevance
    current_title = (profile.get("current_title") or "").lower()
    current_title_relevance = _compute_title_relevance(current_title)

    # Title fit across career history
    career_titles = [(job.get("title") or "").lower() for job in career]
    best_title_relevance = max(
        [_compute_title_relevance(t) for t in career_titles] + [current_title_relevance]
    )

    # Company type analysis
    companies = [(job.get("company") or "").lower() for job in career]
    industries = [(job.get("industry") or "").lower() for job in career]

    consulting_count = sum(1 for c in companies if _fuzzy_match(c, CONSULTING_FIRMS))
    product_count = sum(1 for c in companies if _fuzzy_match(c, PRODUCT_BRANDS))
    product_industry_count = sum(
        1 for ind in industries
        if any(kw in ind for kw in PRODUCT_INDUSTRY_KEYWORDS)
    )

    all_consulting = 1.0 if (consulting_count == len(companies) and len(companies) > 0) else 0.0
    has_product_company_exp = 1.0 if (product_count > 0 or product_industry_count > 0) else 0.0

    # Company size (smaller = more startup-like)
    size_map = {
        "1-10": 5, "11-50": 30, "51-200": 125, "201-500": 350,
        "501-1000": 750, "1001-5000": 3000, "5001-10000": 7500, "10001+": 15000,
    }
    current_company_size = size_map.get(profile.get("current_company_size", ""), 5000)
    startup_fit = _clip(1.0 - (current_company_size - 50) / 15000)

    # Tenure stability score
    tenure_fit = _clip(avg_tenure_months / 36.0) if avg_tenure_months > 0 else 0.5

    return {
        "years_of_experience": yoe,
        "yoe_in_ideal_range": yoe_in_ideal_range,
        "num_career_entries": float(num_career_entries),
        "avg_tenure_months": avg_tenure_months,
        "is_title_chaser": is_title_chaser,
        "current_title_relevance": current_title_relevance,
        "best_title_relevance": best_title_relevance,
        "consulting_career_ratio": consulting_count / max(1, len(companies)),
        "all_consulting_career": all_consulting,
        "has_product_company_exp": has_product_company_exp,
        "product_company_count": float(product_count),
        "startup_fit": startup_fit,
        "tenure_fit": tenure_fit,
    }


def _compute_title_relevance(title_lower: str) -> float:
    """Score how relevant a title is to the JD."""
    if any(t in title_lower for t in HIGH_SIGNAL_TITLES):
        return 1.0
    if any(t in title_lower for t in ADJACENT_SIGNAL_TITLES):
        return 0.7
    if any(t in title_lower for t in NON_TECHNICAL_TITLES):
        return 0.1
    if "engineer" in title_lower or "developer" in title_lower:
        return 0.55
    if "lead" in title_lower or "architect" in title_lower or "manager" in title_lower:
        return 0.4
    return 0.25


def extract_skill_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Extract skill-related features."""
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    # Build career text for entailment
    career_text = " ".join([
        (profile.get("summary") or ""),
        (profile.get("headline") or ""),
    ] + [
        (job.get("description") or "") + " " + (job.get("title") or "")
        for job in career
    ]).lower()

    # Core AI skill count
    skill_names_lower = [(s.get("name") or "").lower() for s in skills]
    core_ai_count = sum(
        1 for name in skill_names_lower
        if name in CORE_AI_SKILLS or any(
            term in name for term in CORE_AI_SKILLS 
            if len(term) > 3 or term in ("nlp", "rag", "cv", "ocr", "tts")
        )
    )

    # High-signal skill count
    high_signal_count = sum(
        1 for name in skill_names_lower
        if name in HIGH_SIGNAL_SKILLS or any(
            term in name for term in HIGH_SIGNAL_SKILLS if len(term) > 4
        )
    )

    # Skill proficiency score (weighted)
    proficiency_map = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}
    relevant_proficiencies = []
    for skill in skills:
        name = (skill.get("name") or "").lower()
        if name in CORE_AI_SKILLS or any(
            term in name for term in CORE_AI_SKILLS 
            if len(term) > 3 or term in ("nlp", "rag", "cv", "ocr", "tts")
        ):
            prof = proficiency_map.get(skill.get("proficiency", "beginner"), 0.25)
            relevant_proficiencies.append(prof)
    skill_proficiency_score = (
        sum(relevant_proficiencies) / len(relevant_proficiencies)
        if relevant_proficiencies else 0.0
    )

    # Skill-text entailment rate
    advanced_skills = [
        s for s in skills
        if s.get("proficiency") in ("advanced", "expert")
    ]
    entailed = 0
    for s in advanced_skills:
        name = (s.get("name") or "").lower()
        if name in career_text or any(
            syn in career_text
            for key, syns in SKILL_SYNONYMS.items()
            if name == key or name in syns
            for syn in syns
        ):
            entailed += 1
    entailment_rate = entailed / max(1, len(advanced_skills))

    # Assessment vs proficiency gap
    assessments = signals.get("skill_assessment_scores", {})
    trust_penalty = 0.0
    assessment_count = 0
    for skill in skills:
        skill_name = skill.get("name", "")
        if skill_name in assessments:
            assessment_count += 1
            claimed_map = {"beginner": 25, "intermediate": 50, "advanced": 75, "expert": 95}
            claimed = claimed_map.get(skill.get("proficiency", "beginner"), 25)
            actual = assessments[skill_name]
            gap = claimed - actual
            if gap > 30:  # Claims advanced but scores < 45
                trust_penalty += 0.1

    # Specific skill category flags
    has_embedding_skills = 1.0 if any(
        term in " ".join(skill_names_lower)
        for term in ("sentence-transformer", "sentence transformer", "bge", "e5",
                      "embedding", "word2vec", "dense retrieval")
    ) else 0.0

    has_vector_db_skills = 1.0 if any(
        term in " ".join(skill_names_lower)
        for term in ("faiss", "pinecone", "weaviate", "qdrant", "milvus",
                      "chroma", "vector database", "vector db")
    ) else 0.0

    has_evaluation_skills = 1.0 if any(
        term in " ".join(skill_names_lower)
        for term in ("ndcg", "mrr", "map", "a/b testing", "ab testing",
                      "evaluation", "ranking metric")
    ) else 0.0

    has_nlp_ir_skills = 1.0 if any(
        term in " ".join(skill_names_lower)
        for term in ("nlp", "natural language", "information retrieval",
                      "search", "retrieval", "bert", "transformers")
    ) else 0.0

    has_llm_skills = 1.0 if any(
        term in " ".join(skill_names_lower)
        for term in ("llm", "large language", "fine-tun", "finetuning",
                      "langchain", "llamaindex", "prompt engineering",
                      "gpt", "chatgpt", "lora", "qlora", "rag")
    ) else 0.0

    return {
        "num_core_ai_skills": float(core_ai_count),
        "num_high_signal_skills": float(high_signal_count),
        "skill_proficiency_score": skill_proficiency_score,
        "skill_text_entailment_rate": entailment_rate,
        "num_advanced_expert_skills": float(len(advanced_skills)),
        "assessment_trust_penalty": _clip(trust_penalty, 0.0, 1.0),
        "num_assessments_taken": float(assessment_count),
        "has_embedding_skills": has_embedding_skills,
        "has_vector_db_skills": has_vector_db_skills,
        "has_evaluation_skills": has_evaluation_skills,
        "has_nlp_ir_skills": has_nlp_ir_skills,
        "has_llm_skills": has_llm_skills,
        "total_skill_count": float(len(skills)),
    }


def extract_behavioral_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Extract behavioral / engagement features from Redrob signals."""
    signals = candidate.get("redrob_signals", {})

    # Activity decay (exponential)
    last_active = _parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = max(0, (REFERENCE_DATE - last_active).days)
    else:
        days_inactive = 365  # Assume very inactive

    activity_decay = math.exp(-0.005 * days_inactive)
    # 0 days → 1.0, 30 days → 0.86, 90 days → 0.64, 180 days → 0.41

    # Response rate (direct)
    response_rate = signals.get("recruiter_response_rate", 0.0)

    # Response time score (inverse, normalized)
    avg_response_hours = signals.get("avg_response_time_hours", 72)
    response_time_score = _clip(1.0 - (avg_response_hours / 168.0))  # 168h = 1 week

    # Notice period — piecewise linear aligned with JD language
    # JD: "Sub-30 preferred. Can buy out up to 30 days. 30+ still in scope."
    notice_days = signals.get("notice_period_days", 0)
    if notice_days <= 30:
        notice_multiplier = 1.0                                       # JD-preferred
    elif notice_days <= 45:
        notice_multiplier = 1.0 - 0.005 * (notice_days - 30)         # 45d → 0.925
    elif notice_days <= 60:
        notice_multiplier = 0.925 - 0.005 * (notice_days - 45)       # 60d → 0.85
    elif notice_days <= 90:
        notice_multiplier = 0.85 - 0.005 * (notice_days - 60)        # 90d → 0.70
    elif notice_days <= 120:
        notice_multiplier = 0.70 - 0.005 * (notice_days - 90)        # 120d → 0.55
    else:
        notice_multiplier = max(0.3, 0.55 - 0.003 * (notice_days - 120))
    # 0d→1.0, 30d→1.0, 45d→0.925, 60d→0.85, 90d→0.70, 120d→0.55

    # Open to work (binary)
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.0

    # Profile completeness
    profile_completeness = signals.get("profile_completeness_score", 0.0) / 100.0

    # Interview completion rate
    interview_rate = signals.get("interview_completion_rate", 0.0)

    # Offer acceptance rate (handle -1 as unknown → neutral 0.5)
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate < 0:
        offer_rate = 0.5

    # Recruiter demand signals
    saved_30d = signals.get("saved_by_recruiters_30d", 0)
    search_30d = signals.get("search_appearance_30d", 0)
    views_30d = signals.get("profile_views_received_30d", 0)
    apps_30d = signals.get("applications_submitted_30d", 0)

    # GitHub activity (handle -1 as no GitHub)
    github = signals.get("github_activity_score", -1)
    github_score = github / 100.0 if github >= 0 else 0.0

    # Verification score
    verified = (
        (1 if signals.get("verified_email", False) else 0) +
        (1 if signals.get("verified_phone", False) else 0) +
        (1 if signals.get("linkedin_connected", False) else 0)
    )
    verification_score = verified / 3.0

    # Connection count (log-scaled, capped)
    connections = signals.get("connection_count", 0)
    connection_score = _clip(math.log1p(connections) / math.log1p(500))

    # Endorsements (log-scaled)
    endorsements = signals.get("endorsements_received", 0)
    endorsement_score = _clip(math.log1p(endorsements) / math.log1p(100))

    return {
        "days_since_active": float(days_inactive),
        "activity_decay_score": activity_decay,
        "recruiter_response_rate": response_rate,
        "response_time_score": response_time_score,
        "notice_period_days": float(notice_days),
        "notice_period_multiplier": notice_multiplier,
        "open_to_work": open_to_work,
        "profile_completeness": profile_completeness,
        "interview_completion_rate": interview_rate,
        "offer_acceptance_rate": offer_rate,
        "saved_by_recruiters_30d": float(saved_30d),
        "search_appearance_30d": float(search_30d),
        "profile_views_30d": float(views_30d),
        "applications_30d": float(apps_30d),
        "github_activity_score": github_score,
        "verification_score": verification_score,
        "connection_score": connection_score,
        "endorsement_score": endorsement_score,
    }


def extract_location_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Extract location-related features."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    work_mode = (signals.get("preferred_work_mode") or "").lower()

    is_india = 1.0 if country == "india" else 0.0
    is_tier1_india = 1.0 if (
        country == "india" and
        any(city in location for city in TIER1_INDIA_CITIES)
    ) else 0.0
    willing = 1.0 if willing_to_relocate else 0.0
    work_mode_compatible = 1.0 if work_mode in ("hybrid", "onsite", "flexible") else 0.5

    # Composite location score — softened for international candidates
    # JD: "Outside India: case-by-case, no visa sponsorship" (not "strongly penalize")
    if is_india and is_tier1_india:
        location_fit = 1.0
    elif is_india:
        location_fit = 0.85 if willing_to_relocate else 0.75
    else:
        location_fit = 0.70 if willing_to_relocate else 0.50

    return {
        "is_india": is_india,
        "is_tier1_india": is_tier1_india,
        "willing_to_relocate": willing,
        "work_mode_compatible": work_mode_compatible,
        "location_fit": location_fit,
    }


def extract_all_features(candidate: dict[str, Any]) -> dict[str, float]:
    """
    Extract ALL features for a single candidate.

    Returns a flat dictionary of ~45 features suitable for LightGBM.
    """
    features: dict[str, float] = {}

    # Structural
    features.update(extract_structural_features(candidate))

    # Skills
    features.update(extract_skill_features(candidate))

    # Behavioral
    features.update(extract_behavioral_features(candidate))

    # Location
    features.update(extract_location_features(candidate))

    # Honeypot flags
    is_honeypot, honeypot_flags = detect_honeypot(candidate)
    features["is_honeypot"] = 1.0 if is_honeypot else 0.0
    features["honeypot_flag_count"] = float(len(honeypot_flags))
    features["has_maturity_impossible"] = 1.0 if any(f.startswith("MATURITY_IMPOSSIBLE") for f in honeypot_flags) else 0.0
    features["has_fictional_company"] = 1.0 if any(f.startswith("FICTIONAL_COMPANY") for f in honeypot_flags) else 0.0

    return features
