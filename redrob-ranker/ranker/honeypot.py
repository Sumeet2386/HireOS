"""
Adversarial (honeypot) candidate detection.

Eight deterministic heuristic layers:

1. Timeline impossibility -- career start ≥3 years before graduation
2. Skill-text entailment failure -- ≥4 advanced/expert skills, >75% absent from career text
3. Skill maturity analysis -- claimed experience with tech that didn't exist yet
4. Heavy career timeline overlap -- concurrent positions beyond realistic limits
5. Keyword stuffer with non-technical title -- AI skills but all titles are HR/Accounting
6. Suspiciously perfect junior profile -- too-good-to-be-true signals for low YoE
7. Fictional company detection -- known trap company names (soft signal only)
8. Expert with zero duration -- "expert" proficiency in skills with 0 months used

NOTE on fictional companies: The dataset uses fictional company names (Dunder
Mifflin, Initech, etc.) as career-history padding for ~82% of candidates.
Having a fictional company is NOT a honeypot signal -- it's dataset design.
The check is kept for soft penalty in scoring but does NOT count toward the
honeypot threshold.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .constants import (
    CORE_AI_SKILLS,
    FICTIONAL_COMPANIES,
    NEGATIVE_TITLE_PATTERNS,
    NON_TECHNICAL_TITLES,
    REFERENCE_DATE,
    SYNONYM_REVERSE_LOOKUP,
    TECH_RELEASE_YEARS,
    COMPANY_FOUNDING_YEARS,
)
from .utils import parse_date

logger = logging.getLogger(__name__)


def _skill_mentioned_in_text(skill_name: str, text: str) -> bool:
    """Check if a skill (or any of its synonyms) appears in the text.

    Uses the pre-built ``SYNONYM_REVERSE_LOOKUP`` for O(1) synonym
    resolution instead of iterating all synonym entries.

    Parameters
    ----------
    skill_name : str
        Skill name to search for.
    text : str
        Lowercased career/profile text corpus.

    Returns
    -------
    bool
        ``True`` if the skill or a synonym is found in *text*.
    """
    skill_lower = skill_name.lower()
    if skill_lower in text:
        return True

    # Use pre-built reverse lookup (O(1) instead of iterating all synonyms)
    related_terms = SYNONYM_REVERSE_LOOKUP.get(skill_lower)
    if related_terms:
        return any(term in text for term in related_terms)

    # Check individual words for multi-word skills
    words = skill_lower.split()
    if len(words) > 1:
        return any(w in text for w in words if len(w) > 3)
    return False


def check_timeline_impossibility(candidate: dict[str, Any]) -> list[str]:
    """Check 1: Career dates overlap with education by ≥3 years.

    In India, campus placements routinely start 1-2 years before
    graduation. Only flag when the gap is ≥3 years, which is genuinely
    impossible for a full-time role.

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.

    Returns
    -------
    list[str]
        List of flag strings (empty if no issues found).
    """
    flags: list[str] = []
    education = candidate.get("education", [])
    career = candidate.get("career_history", [])

    if not education or not career:
        return flags

    grad_years = [
        edu.get("end_year")
        for edu in education
        if edu.get("end_year") is not None
    ]
    if not grad_years:
        return flags
    latest_grad_year = max(grad_years)

    for job in career:
        start_date = parse_date(job.get("start_date"))
        duration = job.get("duration_months", 0)
        if start_date and duration > 12:
            gap = latest_grad_year - start_date.year
            # Only flag if started ≥3 years before graduation
            if gap >= 3:
                flags.append(
                    f"TIMELINE_IMPOSSIBLE: job at {job.get('company', '?')} "
                    f"started {start_date.year} but graduated {latest_grad_year} "
                    f"({gap}y gap)"
                )

    return flags


def check_skill_text_entailment(candidate: dict[str, Any]) -> list[str]:
    """Check 2: Advanced/expert skills should appear somewhere in career text.

    Only triggers the SEMANTIC_CONTRADICTION flag when:
    - The candidate has ≥4 advanced/expert skills (enough for statistical
      significance), AND
    - >75% of those skills are absent from ALL career text.

    This avoids false-flagging candidates who have just 1-2 advanced
    skills that happen not to be mentioned verbatim.

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.

    Returns
    -------
    list[str]
        List of flag strings.
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

    # Flag individual unentailed skills (soft signals)
    for name in unentailed:
        flags.append(f"SKILL_NOT_ENTAILED: {name}")

    # Only flag SEMANTIC_CONTRADICTION when there are enough advanced skills
    # (≥4) and a strong majority (>75%) are unentailed
    if len(advanced_skills) >= 4 and len(unentailed) > len(advanced_skills) * 0.75:
        flags.append(
            f"SEMANTIC_CONTRADICTION: {len(unentailed)}/{len(advanced_skills)} "
            f"advanced skills not found in career text"
        )

    return flags


def check_skill_maturity(candidate: dict[str, Any]) -> list[str]:
    """Check 3: Skill duration exceeds technology's lifespan.

    E.g. claiming 60 months of LangChain when it was released in 2022.

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.

    Returns
    -------
    list[str]
        List of flag strings.
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
    """Check 4: Career entries overlap by >6 months.

    Concurrent positions beyond what is realistic for legitimate
    career histories.

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.

    Returns
    -------
    list[str]
        List of flag strings.
    """
    flags: list[str] = []
    career = candidate.get("career_history", [])

    if len(career) < 2:
        return flags

    # Parse all career entries with valid dates
    entries = []
    for job in career:
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        if start:
            if end is None and job.get("is_current"):
                end = REFERENCE_DATE
            if end:
                entries.append((start, end, job.get("company", "?")))

    # Sort by start date
    entries.sort(key=lambda x: x[0])

    overlap_months_total = 0
    for i in range(len(entries) - 1):
        _, end_i, _ = entries[i]
        start_j, _, _ = entries[i + 1]
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
    """Check 5: AI/ML keywords in skills but ALL titles are non-technical.

    E.g. someone with skills like 'PyTorch, BERT, LLM' but every job
    title is 'Marketing Manager' or 'Accountant'.

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.

    Returns
    -------
    list[str]
        List of flag strings.
    """
    flags: list[str] = []
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    # Count AI-relevant skills
    ai_skill_count = 0
    for skill in skills:
        name = (skill.get("name") or "").lower()
        if name in CORE_AI_SKILLS or re.search(
            r"\b(?:ml|ai|deep learning|nlp|neural)\b", name
        ):
            ai_skill_count += 1

    if ai_skill_count < 3:
        return flags

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
    """Check 6: Junior candidate with suspiciously perfect behavioral signals.

    If YoE < 3 but has very high engagement metrics (response rate > 0.8,
    high recruiter saves, near-perfect assessments), flag as potentially
    synthetic.

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.

    Returns
    -------
    list[str]
        List of flag strings.
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
    """Check 7: Candidate works at a fictional / trap company.

    NOTE: ~82% of candidates in the dataset have fictional company names
    as career-history padding. This is a dataset design choice, NOT a
    reliable honeypot signal. The flags are returned for use as a soft
    scoring penalty but do NOT count toward the honeypot threshold.

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.

    Returns
    -------
    list[str]
        List of flag strings.
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


def check_expert_zero_duration(candidate: dict[str, Any]) -> list[str]:
    """Check 8: Expert proficiency with 0 months of experience.

    Per the official hackathon docs, a key honeypot signal is
    "expert proficiency in 10 skills with 0 years used". Flag when
    a candidate has ≥3 skills at expert/advanced level with 0 duration.

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.

    Returns
    -------
    list[str]
        List of flag strings.
    """
    flags: list[str] = []
    skills = candidate.get("skills", [])

    zero_duration_experts = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 1) == 0
    ]

    if len(zero_duration_experts) >= 3:
        names = [s.get("name", "?") for s in zero_duration_experts[:5]]
        flags.append(
            f"EXPERT_ZERO_DURATION: {len(zero_duration_experts)} expert/advanced skills "
            f"with 0 months used ({', '.join(names)})"
        )

    return flags


def check_company_founding_timeline(candidate: dict[str, Any]) -> list[str]:
    """Check 9: Candidate claims to work at a company before it was founded.

    E.g. Started working at Sarvam AI in 2020, but it was founded in 2023.

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.

    Returns
    -------
    list[str]
        List of flag strings.
    """
    flags: list[str] = []
    career = candidate.get("career_history", [])

    for job in career:
        company = (job.get("company") or "").lower().strip()
        start_date = job.get("start_date")
        if start_date and company in COMPANY_FOUNDING_YEARS:
            try:
                start_year = int(start_date.split('-')[0])
                founding_year = COMPANY_FOUNDING_YEARS[company]
                if start_year < founding_year:
                    flags.append(
                        f"COMPANY_TIMELINE_IMPOSSIBLE: Started at {job.get('company', company)} "
                        f"in {start_year}, but it was founded in {founding_year}"
                    )
            except (ValueError, TypeError, IndexError):
                continue

    return flags


def detect_honeypot(
    candidate: dict[str, Any],
    threshold: int = 3,
) -> tuple[bool, list[str]]:
    """Run all honeypot checks and return (is_honeypot, flags).

    A candidate is flagged as honeypot if they trigger >= ``threshold``
    distinct **hard** check categories (not individual flags).

    IMPORTANT: The fictional company check (Check 7) does NOT count
    toward the threshold because ~82% of candidates in the dataset have
    fictional company names as career-history padding -- it's dataset
    design, not a honeypot signal. Its flags are still returned for use
    as soft penalties in the scoring formula.

    Parameters
    ----------
    candidate : dict
        Full candidate JSON record.
    threshold : int
        Minimum number of distinct hard check categories to trigger
        honeypot flag. Default is 3.

    Returns
    -------
    tuple[bool, list[str]]
        ``(is_honeypot, list_of_all_flags)``
    """
    all_flags: list[str] = []
    categories_triggered = 0

    # Check 9: Impossible company founding dates (immediate fatal flag)
    company_flags = check_company_founding_timeline(candidate)
    if company_flags:
        all_flags.extend(company_flags)
        return True, all_flags

    # Check 1: Timeline (hard signal — only ≥3 year gaps)
    timeline_flags = check_timeline_impossibility(candidate)
    if timeline_flags:
        all_flags.extend(timeline_flags)
        categories_triggered += 1

    # Check 2: Skill-text entailment (hard signal — ≥4 skills, >75% unentailed)
    entailment_flags = check_skill_text_entailment(candidate)
    # Only count if the broad SEMANTIC_CONTRADICTION flag fired
    if any("SEMANTIC_CONTRADICTION" in f for f in entailment_flags):
        all_flags.extend(entailment_flags)
        categories_triggered += 1
    elif entailment_flags:
        all_flags.extend(entailment_flags)
        # Individual skill misses are a soft signal, not a full category

    # Check 3: Skill maturity (hard signal — precise)
    maturity_flags = check_skill_maturity(candidate)
    if maturity_flags:
        all_flags.extend(maturity_flags)
        categories_triggered += 1

    # Check 4: Heavy career overlap (hard signal)
    overlap_flags = check_heavy_career_overlap(candidate)
    if overlap_flags:
        all_flags.extend(overlap_flags)
        categories_triggered += 1

    # Check 5: Keyword stuffer (hard signal)
    stuffer_flags = check_keyword_stuffer(candidate)
    if stuffer_flags:
        all_flags.extend(stuffer_flags)
        categories_triggered += 1

    # Check 6: Suspicious junior (hard signal)
    junior_flags = check_suspicious_junior(candidate)
    if junior_flags:
        all_flags.extend(junior_flags)
        categories_triggered += 1

    # Check 7: Fictional company (SOFT signal — does NOT count toward threshold)
    # ~82% of candidates have fictional companies as dataset padding
    fictional_flags = check_fictional_company(candidate)
    if fictional_flags:
        all_flags.extend(fictional_flags)
        # NOT incrementing categories_triggered

    # Check 8: Expert with zero duration (hard signal)
    zero_dur_flags = check_expert_zero_duration(candidate)
    if zero_dur_flags:
        all_flags.extend(zero_dur_flags)
        categories_triggered += 1

    is_honeypot = categories_triggered >= threshold
    return is_honeypot, all_flags
