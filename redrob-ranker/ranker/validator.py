"""
Output CSV validation for the Redrob submission format.

Ensures:

- Exactly 100 data rows + 1 header
- Columns: candidate_id, rank, score, reasoning
- candidate_id matches ``CAND_XXXXXXX`` pattern
- Ranks 1-100 unique, scores monotonically non-increasing
- Tie-breaking: candidate_id ascending for equal scores
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
EXPECTED_DATA_ROWS = 100


def validate_submission(csv_path: str | Path) -> list[str]:
    """Validate a submission CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the submission CSV file.

    Returns
    -------
    list[str]
        List of error strings. An empty list indicates a valid submission.
    """
    errors: list[str] = []
    path = Path(csv_path)

    if path.suffix.lower() != ".csv":
        errors.append("Filename must use a .csv extension.")

    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)

            try:
                header = next(reader)
            except StopIteration:
                errors.append("File is empty -- must have header + 100 rows.")
                return errors

            if header != REQUIRED_HEADER:
                errors.append(
                    f"Header must be: {','.join(REQUIRED_HEADER)}\n"
                    f"Found: {','.join(header)}"
                )

            data_rows = []
            for row in reader:
                if any(cell.strip() for cell in row):
                    data_rows.append(row)

    except UnicodeDecodeError:
        errors.append("File must be UTF-8 encoded.")
        return errors
    except OSError as exc:
        errors.append(f"Cannot read file: {exc}")
        return errors

    n = len(data_rows)
    if n != EXPECTED_DATA_ROWS:
        errors.append(f"Expected {EXPECTED_DATA_ROWS} data rows, found {n}.")

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    by_rank: list[tuple[int, float, str]] = []

    for i, cells in enumerate(data_rows):
        row_num = i + 2  # 1-indexed, header is row 1

        if len(cells) != len(REQUIRED_HEADER):
            errors.append(f"Row {row_num}: expected {len(REQUIRED_HEADER)} columns, got {len(cells)}.")
            continue

        cid = cells[0].strip()
        rank_s = cells[1].strip()
        score_s = cells[2].strip()

        # Validate candidate_id
        if not CANDIDATE_ID_PATTERN.match(cid):
            errors.append(f"Row {row_num}: invalid candidate_id '{cid}'.")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'.")
        else:
            seen_ids.add(cid)

        # Validate rank
        rank: int | None = None
        try:
            rank = int(rank_s)
            if not 1 <= rank <= 100:
                errors.append(f"Row {row_num}: rank must be 1-100, got {rank}.")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}.")
            else:
                seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {row_num}: rank must be integer, got '{rank_s}'.")

        # Validate score
        score: float | None = None
        try:
            score = float(score_s)
        except ValueError:
            errors.append(f"Row {row_num}: score must be float, got '{score_s}'.")

        if rank is not None and score is not None:
            by_rank.append((rank, score, cid))

    # Check monotonically non-increasing scores
    by_rank.sort(key=lambda x: x[0])
    for i in range(len(by_rank) - 1):
        r1, s1, _ = by_rank[i]
        r2, s2, _ = by_rank[i + 1]
        if s1 < s2:
            errors.append(
                f"Scores must be non-increasing: rank {r1} ({s1}) < rank {r2} ({s2})."
            )

    # Check tie-breaking (ascending candidate_id for equal scores)
    for i in range(len(by_rank) - 1):
        r1, s1, c1 = by_rank[i]
        r2, s2, c2 = by_rank[i + 1]
        if s1 == s2 and c1 > c2:
            errors.append(
                f"Tie-break: equal scores at ranks {r1},{r2} require "
                f"candidate_id ascending ({c1} > {c2})."
            )

    if errors:
        logger.warning("Submission validation found %d error(s)", len(errors))
    else:
        logger.info("Submission validation passed")

    return errors
