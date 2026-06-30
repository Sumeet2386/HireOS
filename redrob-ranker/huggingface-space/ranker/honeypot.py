"""
Adversarial (honeypot) candidate detection.

Three deterministic heuristic layers plus three extra checks adopted from
competitor analysis:

1. Timeline impossibility — career dates overlap education or exceed lifespan
2. Skill-text entailment failure — advanced/expert skills absent from career text
3. Skill maturity analysis — claimed experience with tech that didn't exist yet
4. Heavy career timeline overlap — concurrent positions beyond what's realistic
5. Keyword stuffer with non-technical title — AI skills but all titles are HR/Accounting
6. Suspiciously perfect junior profile — too-good-to-be-true behavioral signals for low YoE
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

from .constants import (
    CORE_AI_SKILLS,
    FICTIONAL_COMPANIES,
    NEGATIVE_TITLE_PATTERNS,
    NON_TECHNICAL_TITLES,
    REFERENCE_DATE,
    SKILL_SYNONYMS,
    TECH_RELEASE_YEARS,
)


def _parse_date(value: str | None) -> date | None:
    """Parse YYYY-MM-DD string to date, returning None on failure."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _skill_mentioned_in_text(skill_name: str, text: str) -> bool:
    """Check if a skill (or any of its synonyms) appears in the text."""
    skill_lower = skill_name.lower()
    if skill_lower in text:
        return True
    # Check synonyms
    for key, synonyms in SKILL_SYNONYMS.items():
        if skill_lower == key or skill_lower in synonyms:
            return any(syn in text for syn in synonyms) or key in text
    # Check individual words for multi-word skills
    words = skill_lower.split()
    if len(words) > 1:
        return any(w in text for w in words if len(w) > 3)
    return False


def check_timeline_impossibility(candidate: dict[str, Any]) -> list[str]:
    """
    Check 1: Career dates overlap with education impossibly.

    Flag if a candidate had a full-time role (>12 months) that started before
    they graduated.
    """
    flags: list[str] = []
    education = candidate.get("education", [])
    career = candidate.get("career_history", [])

    if not education or not career:
        return flags

    # Get the latest graduation year
    grad_years = [
        edu.get("end_year")
        for edu in education
        if edu.get("end_year") is not None
    ]
    if not grad_years:
        return flags
    latest_grad_year = max(grad_years)

    for job in career:
        start_date = _parse_date(job.get("start_date"))
        duration = job.get("duration_months", 0)
        if start_date and start_date.year < latest_grad_year and duration > 12:
            flags.append(
                f"TIMELINE_IMPOSSIBLE: job at {job.get('company', '?')} "
                f"started {start_date.year} but graduated {latest_grad_year}"
            )

    return flags


def check_skill_text_entailment(candidate: dict[str, Any]) -> list[str]:
    """
    Check 2: Advanced/expert skills should appear somewhere in career text.

    If >50% of advanced/expert skills are absent from ALL career descriptions,
    flag as semantic contradiction.
    """
    flags: list[str] = []
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    # Build full career text
    text_parts = [
        (profile.get("summary") or "").lower(),
        (profile.get("headline") or "").lower(),
    ]
    for job in career:
        text_parts.append((job.get("description") or "").lower())
        text_parts.append((job.get("title") or "").lower())
    career_text = " ".join(text_parts)

    advanced_skills = [
        s for s in skills
        if s.get("proficiency") in ("advanced", "expert")
    ]

    if not advanced_skills:
        return flags

    unentailed = []
    for skill in advanced_skills:
        skill_name = skill.get("name", "")
        if not _skill_mentioned_in_text(skill_name, career_text):
            unentailed.append(skill_name)

    # Flag individual unentailed skills
    for name in unentailed:
        flags.append(f"SKILL_NOT_ENTAILED: {name}")

    # Flag if majority of advanced skills are unentailed
    if len(unentailed) > len(advanced_skills) * 0.5:
        flags.append("SEMANTIC_CONTRADICTION: >50% advanced skills not found in career text")

    return flags


def check_skill_maturity(candidate: dict[str, Any]) -> list[str]:
    """
    Check 3: Skill duration exceeds technology's lifespan.

    E.g. claiming 60 months of LangChain when it was released in 2022.
    """
    flags: list[str] = []
    skills = candidate.get("skills", [])

    for skill in skills:
        skill_name = (skill.get("name") or "").lower()
        duration_months = skill.get("duration_months", 0)

        if skill_name in TECH_RELEASE_YEARS:
            release_year = TECH_RELEASE_YEARS[skill_name]
            max_possible_months = (REFERENCE_DATE.year - release_year) * 12 + REFERENCE_DATE.month
            # Allow 6 months of grace (beta access, pre-release, etc.)
            if duration_months > max_possible_months + 6:
                flags.append(
                    f"MATURITY_IMPOSSIBLE: {skill.get('name', skill_name)} "
                    f"claims {duration_months}mo but tech released in {release_year} "
                    f"(max possible ~{max_possible_months}mo)"
                )

    return flags


def check_heavy_career_overlap(candidate: dict[str, Any]) -> list[str]:
    """
    Check 4: Career entries overlap by >6 months (concurrent positions
    beyond what's realistic).

    Adapted from competitor repo analysis.
    """
    flags: list[str] = []
    career = candidate.get("career_history", [])

    if len(career) < 2:
        return flags

    # Parse all career entries with valid dates
    entries = []
    for job in career:
        start = _parse_date(job.get("start_date"))
        end = _parse_date(job.get("end_date"))
        if start:
            if end is None and job.get("is_current"):
                end = REFERENCE_DATE
            if end:
                entries.append((start, end, job.get("company", "?")))

    # Sort by start date
    entries.sort(key=lambda x: x[0])

    overlap_months_total = 0
    for i in range(len(entries) - 1):
        _, end_i, company_i = entries[i]
        start_j, _, company_j = entries[i + 1]
        if end_i > start_j:
            overlap_days = (end_i - start_j).days
            overlap_mo = overlap_days / 30.0
            if overlap_mo > 6:
                overlap_months_total += overlap_mo

    if overlap_months_total > 12:
        flags.append(
            f"HEAVY_OVERLAP_TIMELINE: {overlap_months_total:.0f} months of career overlap"
        )

    return flags


def check_keyword_stuffer(candidate: dict[str, Any]) -> list[str]:
    """
    Check 5: AI/ML keywords in skills but ALL titles are non-technical.

    E.g. someone with skills like 'PyTorch, BERT, LLM' but every job title
    is 'Marketing Manager' or 'Accountant'.

    Adapted from competitor repo analysis.
    """
    flags: list[str] = []
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    # Count AI-relevant skills
    ai_skill_count = 0
    import re
    for skill in skills:
        name = (skill.get("name") or "").lower()
        if name in CORE_AI_SKILLS or re.search(r'\b(?:ml|ai|deep learning|nlp|neural)\b', name):
            ai_skill_count += 1

    if ai_skill_count < 3:
        return flags  # Not enough AI skills to be suspicious

    # Check if ALL titles are non-technical
    all_titles = [
        (job.get("title") or "").lower() for job in career
    ]
    all_titles.append((profile.get("current_title") or "").lower())

    all_non_technical = all(
        any(pattern in title for pattern in NEGATIVE_TITLE_PATTERNS)
        for title in all_titles
        if title.strip()
    )

    if all_non_technical and ai_skill_count >= 3:
        flags.append(
            f"KEYWORD_STUFFER: {ai_skill_count} AI skills but all titles are non-technical"
        )

    return flags


def check_suspicious_junior(candidate: dict[str, Any]) -> list[str]:
    """
    Check 6: Junior candidate with suspiciously perfect behavioral signals.

    If YoE < 3 but has very high engagement metrics (response rate > 0.8,
    high recruiter saves, near-perfect assessments), flag as potentially
    synthetic.

    Adapted from competitor repo analysis.
    """
    flags: list[str] = []
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    yoe = profile.get("years_of_experience", 0)
    if yoe >= 3:
        return flags

    response_rate = signals.get("recruiter_response_rate", 0)
    saved_30d = signals.get("saved_by_recruiters_30d", 0)
    interview_rate = signals.get("interview_completion_rate", 0)

    # Check assessment scores
    assessments = signals.get("skill_assessment_scores", {})
    avg_assessment = 0.0
    if assessments:
        scores = list(assessments.values())
        avg_assessment = sum(scores) / len(scores) if scores else 0

    suspicion_score = 0
    if response_rate > 0.85:
        suspicion_score += 1
    if saved_30d > 12:
        suspicion_score += 1
    if interview_rate > 0.9:
        suspicion_score += 1
    if avg_assessment > 85:
        suspicion_score += 1

    if suspicion_score >= 3:
        flags.append(
            f"SUSPICIOUS_JUNIOR: {yoe:.1f}y YoE but response_rate={response_rate:.2f}, "
            f"saved={saved_30d}, interview_rate={interview_rate:.2f}, "
            f"avg_assessment={avg_assessment:.1f}"
        )

    return flags


def check_fictional_company(candidate: dict[str, Any]) -> list[str]:
    """
    Check 7: Candidate works at a fictional / trap company.

    The hackathon docs explicitly list fictional company names
    (Dunder Mifflin, Stark Industries, Globex Inc, Initech, Acme Corp)
    as honeypot signals. Hooli and Pied Piper (from Silicon Valley)
    were also found in our data analysis.
    """
    flags: list[str] = []
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    companies_to_check = [
        (profile.get("current_company") or "").lower().strip(),
    ] + [
        (job.get("company") or "").lower().strip()
        for job in career
    ]

    fictional_found = []
    for company in companies_to_check:
        if not company:
            continue
        for fictional in FICTIONAL_COMPANIES:
            if fictional in company or company in fictional:
                fictional_found.append(company)
                break

    if fictional_found:
        unique = list(set(fictional_found))
        flags.append(
            f"FICTIONAL_COMPANY: works at {', '.join(unique)} "
            f"(known fictional/trap company)"
        )

    return flags


def detect_honeypot(candidate: dict[str, Any], threshold: int = 2) -> tuple[bool, list[str]]:
    """
    Run all honeypot checks and return (is_honeypot, flags).

    A candidate is flagged as honeypot if they trigger >= `threshold` distinct
    check categories (not individual flags).

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.
    threshold : int
        Minimum number of distinct check categories to trigger honeypot flag.

    Returns
    -------
    tuple[bool, list[str]]
        (is_honeypot, list_of_all_flags)
    """
    all_flags: list[str] = []
    categories_triggered = 0

    # Check 1: Timeline
    timeline_flags = check_timeline_impossibility(candidate)
    if timeline_flags:
        all_flags.extend(timeline_flags)
        categories_triggered += 1

    # Check 2: Skill-text entailment
    entailment_flags = check_skill_text_entailment(candidate)
    # Only count if the broad SEMANTIC_CONTRADICTION flag fired
    if any("SEMANTIC_CONTRADICTION" in f for f in entailment_flags):
        all_flags.extend(entailment_flags)
        categories_triggered += 1
    elif entailment_flags:
        all_flags.extend(entailment_flags)
        # Individual skill misses are a soft signal, not a full category

    # Check 3: Skill maturity
    maturity_flags = check_skill_maturity(candidate)
    if maturity_flags:
        all_flags.extend(maturity_flags)
        categories_triggered += 1

    # Check 4: Heavy career overlap
    overlap_flags = check_heavy_career_overlap(candidate)
    if overlap_flags:
        all_flags.extend(overlap_flags)
        categories_triggered += 1

    # Check 5: Keyword stuffer
    stuffer_flags = check_keyword_stuffer(candidate)
    if stuffer_flags:
        all_flags.extend(stuffer_flags)
        categories_triggered += 1

    # Check 6: Suspicious junior
    junior_flags = check_suspicious_junior(candidate)
    if junior_flags:
        all_flags.extend(junior_flags)
        categories_triggered += 1

    # Check 7: Fictional company
    fictional_flags = check_fictional_company(candidate)
    if fictional_flags:
        all_flags.extend(fictional_flags)
        categories_triggered += 1

    is_honeypot = categories_triggered >= threshold
    return is_honeypot, all_flags
