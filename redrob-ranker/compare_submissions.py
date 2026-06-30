"""Compare old vs new submission."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load_csv(path):
    cids = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cids.append(row["candidate_id"].strip())
    return cids

old = load_csv(ROOT / "final_submission.csv")
new = load_csv(ROOT / "new_submission.csv")

# Load weak labels
with open(ROOT / "artifacts" / "weak_labels.json") as f:
    wl = json.load(f)
wl_lookup = {w["candidate_id"]: w["weak_label"] for w in wl}

# Load candidates for titles
candidates = {}
cands_path = ROOT.parent / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
for line in open(cands_path, "r", encoding="utf-8"):
    if line.strip():
        c = json.loads(line)
        if c["candidate_id"] in set(old + new):
            candidates[c["candidate_id"]] = c

print("=" * 80)
print("TOP 10 COMPARISON")
print("=" * 80)
print(f"{'Rank':>4} {'OLD':>15} {'NEW':>15} {'Change':>8}")
for i in range(10):
    old_cid = old[i]
    new_cid = new[i]
    change = ""
    if old_cid == new_cid:
        change = "same"
    elif new_cid in old[:100]:
        old_rank = old.index(new_cid) + 1
        change = f"from#{old_rank}"
    else:
        change = "NEW"
    print(f"  #{i+1:>2} {old_cid} {new_cid} {change:>8}")

print(f"\n{'=' * 80}")
print("SET OVERLAP ANALYSIS")
print("=" * 80)

old_set = set(old)
new_set = set(new)

overlap = old_set & new_set
only_old = old_set - new_set
only_new = new_set - old_set

print(f"Overlap: {len(overlap)}/100")
print(f"Dropped from old: {len(only_old)}")
print(f"New additions: {len(only_new)}")

print(f"\nDropped candidates:")
for cid in sorted(only_old):
    old_rank = old.index(cid) + 1
    c = candidates.get(cid, {})
    p = c.get("profile", {})
    title = p.get("current_title", "?")
    company = p.get("current_company", "?")
    wl_s = wl_lookup.get(cid, "N/A")
    print(f"  OLD#{old_rank:>3} {cid} {title} at {company} (wl={wl_s})")

print(f"\nNew additions:")
for cid in sorted(only_new):
    new_rank = new.index(cid) + 1
    c = candidates.get(cid, {})
    p = c.get("profile", {})
    title = p.get("current_title", "?")
    company = p.get("current_company", "?")
    wl_s = wl_lookup.get(cid, "N/A")
    print(f"  NEW#{new_rank:>3} {cid} {title} at {company} (wl={wl_s})")

print(f"\n{'=' * 80}")
print("CUSTOMER SUPPORT CHECK")
print("=" * 80)
for cid in ["CAND_0057340", "CAND_0026316"]:
    if cid in new_set:
        print(f"  ⚠️  {cid} STILL in new submission at rank {new.index(cid) + 1}")
    else:
        print(f"  ✅ {cid} REMOVED from new submission")

print(f"\n{'=' * 80}")
print("TOP 10 TITLE COMPARISON")
print("=" * 80)
print("\nOLD Top 10:")
for i in range(10):
    cid = old[i]
    c = candidates.get(cid, {})
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    title = p.get("current_title", "?")
    yoe = p.get("years_of_experience", 0)
    saved = s.get("saved_by_recruiters_30d", 0)
    search = s.get("search_appearance_30d", 0)
    print(f"  #{i+1:>2} {cid} {title} ({yoe:.1f}y) saved={saved} search={search}")

print("\nNEW Top 10:")
for i in range(10):
    cid = new[i]
    c = candidates.get(cid, {})
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    title = p.get("current_title", "?")
    yoe = p.get("years_of_experience", 0)
    saved = s.get("saved_by_recruiters_30d", 0)
    search = s.get("search_appearance_30d", 0)
    print(f"  #{i+1:>2} {cid} {title} ({yoe:.1f}y) saved={saved} search={search}")

# Check non-technical titles in new submission
print(f"\n{'=' * 80}")
print("NON-TECHNICAL TITLES CHECK IN NEW SUBMISSION")
print("=" * 80)
non_tech = ["customer support", "hr ", "human resource", "project manager", "content", "marketing",
            "sales", "recruiter", "admin", "secretary", "account manager"]
for i, cid in enumerate(new):
    c = candidates.get(cid, {})
    p = c.get("profile", {})
    title = (p.get("current_title", "") or "").lower()
    for nt in non_tech:
        if nt in title:
            company = p.get("current_company", "?")
            print(f"  ⚠️  NEW#{i+1:>3} {cid} title='{p.get('current_title')}' company='{company}'")
            break
