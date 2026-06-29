# Commit Strategy: HireOS Hackathon Repo (Clean History)

## Current Situation

### ❌ Problems with existing hackathon repo (`d:\Hackathons\Data Challenge IndiaRuns`)
- **3 messy commits** that dump everything at once — doesn't look like real development
- **`__pycache__/` committed** — dead giveaway of carelessness
- Files like `things to do`, `Hackathon info` (no extension) look unprofessional
- Analysis outputs and multiple test CSVs committed together with core code
- Everything was dumped in 2 bulk commits

### ✅ Target: HireOS repo (`d:\HireOS\HireOS`)  
- Currently has 3 commits (gitignore + HACKATHON_COMPLETE_REFERENCE.md)
- Clean slate to build a proper history

---

## Strategy: Fresh Start with `--amend` + Force Push

Since the HireOS repo only has 3 small commits, the cleanest approach is to **nuke it and rebuild from scratch** with a proper commit sequence.

> [!IMPORTANT]
> This will rewrite git history on the HireOS repo. Since it's brand new with no collaborators, this is perfectly safe.

---

## The Commit Plan (15 commits — realistic 2-3 day hackathon pace)

Run all commands from **`d:\HireOS\HireOS`**. Copy files from `d:\Hackathons\Data Challenge IndiaRuns` as indicated.

---

### Step 0: Reset the repo
```powershell
# Delete everything except .git
git checkout main
Get-ChildItem -Path . -Exclude .git | Remove-Item -Recurse -Force
git rm -r --cached .
```

---

### Commit 1: `init: project scaffold and hackathon reference`
**Copy these files:**
- `README.md` (from hackathon repo — but we'll create a simple initial version)
- `HACKATHON_COMPLETE_REFERENCE.md`
- `.gitignore`

```powershell
# Create initial README (short, starter version)
# Copy .gitignore and HACKATHON_COMPLETE_REFERENCE.md from source
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\.gitignore" -Destination ".\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\HACKATHON_COMPLETE_REFERENCE.md" -Destination ".\"

# Create a simple initial README
@"
# HireOS — AI Candidate Ranking System

Submission for the Redrob India Runs Data & AI Challenge.
"@ | Out-File -Encoding utf8 README.md

git add .gitignore HACKATHON_COMPLETE_REFERENCE.md README.md
git commit -m "init: project scaffold and hackathon reference"
```

---

### Commit 2: `docs: add research analysis and implementation plan`
```powershell
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\research_deep_analysis.md" -Destination ".\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\implementation_plan.md" -Destination ".\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\hackathon_deep_analysis.md" -Destination ".\"

git add research_deep_analysis.md implementation_plan.md hackathon_deep_analysis.md
git commit -m "docs: add research analysis and implementation plan"
```

---

### Commit 3: `feat: add project structure and dependencies`
```powershell
mkdir redrob-ranker
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\requirements.txt" -Destination "redrob-ranker\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\Dockerfile" -Destination "redrob-ranker\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\submission_metadata.yaml" -Destination "redrob-ranker\"

git add redrob-ranker/requirements.txt redrob-ranker/Dockerfile redrob-ranker/submission_metadata.yaml
git commit -m "feat: add project structure and dependencies"
```

---

### Commit 4: `feat(ranker): add constants and configuration module`
```powershell
mkdir redrob-ranker\ranker
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\ranker\__init__.py" -Destination "redrob-ranker\ranker\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\ranker\constants.py" -Destination "redrob-ranker\ranker\"

git add redrob-ranker/ranker/__init__.py redrob-ranker/ranker/constants.py
git commit -m "feat(ranker): add constants and configuration module"
```

---

### Commit 5: `feat(ranker): implement feature extraction pipeline`
```powershell
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\ranker\features.py" -Destination "redrob-ranker\ranker\"

git add redrob-ranker/ranker/features.py
git commit -m "feat(ranker): implement feature extraction pipeline"
```

---

### Commit 6: `feat(ranker): add honeypot detection logic`
```powershell
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\ranker\honeypot.py" -Destination "redrob-ranker\ranker\"

git add redrob-ranker/ranker/honeypot.py
git commit -m "feat(ranker): add honeypot detection logic"
```

---

### Commit 7: `feat(ranker): implement candidate recall and retrieval`
```powershell
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\ranker\recall.py" -Destination "redrob-ranker\ranker\"

git add redrob-ranker/ranker/recall.py
git commit -m "feat(ranker): implement candidate recall and retrieval"
```

---

### Commit 8: `feat(ranker): add learning-to-rank model integration`
```powershell
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\ranker\ltr.py" -Destination "redrob-ranker\ranker\"

git add redrob-ranker/ranker/ltr.py
git commit -m "feat(ranker): add learning-to-rank model integration"
```

---

### Commit 9: `feat(ranker): implement reasoning engine for candidate explanations`
```powershell
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\ranker\reasoning.py" -Destination "redrob-ranker\ranker\"

git add redrob-ranker/ranker/reasoning.py
git commit -m "feat(ranker): implement reasoning engine for candidate explanations"
```

---

### Commit 10: `feat(ranker): add submission validator`
```powershell
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\ranker\validator.py" -Destination "redrob-ranker\ranker\"

git add redrob-ranker/ranker/validator.py
git commit -m "feat(ranker): add submission validator"
```

---

### Commit 11: `feat(precompute): add EDA and data preprocessing pipeline`
```powershell
mkdir redrob-ranker\precompute
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\precompute\__init__.py" -Destination "redrob-ranker\precompute\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\precompute\01_eda.py" -Destination "redrob-ranker\precompute\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\precompute\02_text_synthesis.py" -Destination "redrob-ranker\precompute\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\precompute\03_generate_embeddings.py" -Destination "redrob-ranker\precompute\"

git add redrob-ranker/precompute/
git commit -m "feat(precompute): add EDA and data preprocessing pipeline"
```

---

### Commit 12: `feat(precompute): add index building and feature extraction steps`
```powershell
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\precompute\04_build_faiss_index.py" -Destination "redrob-ranker\precompute\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\precompute\05_build_bm25_index.py" -Destination "redrob-ranker\precompute\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\precompute\06_extract_features.py" -Destination "redrob-ranker\precompute\"

git add redrob-ranker/precompute/
git commit -m "feat(precompute): add index building and feature extraction steps"
```

---

### Commit 13: `feat(precompute): add weak label generation and LTR training`
```powershell
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\precompute\07_generate_weak_labels.py" -Destination "redrob-ranker\precompute\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\precompute\08_train_ltr_model.py" -Destination "redrob-ranker\precompute\"

git add redrob-ranker/precompute/
git commit -m "feat(precompute): add weak label generation and LTR training"
```

---

### Commit 14: `feat: implement main ranking entry point and evaluation harness`
```powershell
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\rank.py" -Destination "redrob-ranker\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\evaluate.py" -Destination "redrob-ranker\"
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\test_smoke.py" -Destination "redrob-ranker\"

git add redrob-ranker/rank.py redrob-ranker/evaluate.py redrob-ranker/test_smoke.py
git commit -m "feat: implement main ranking entry point and evaluation harness"
```

---

### Commit 15: `docs: add project README and finalize submission`
```powershell
# Copy the full README
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\redrob-ranker\README.md" -Destination "redrob-ranker\"

# Update root README to be more comprehensive (overwrite the initial one)
Copy-Item "d:\Hackathons\Data Challenge IndiaRuns\README.md" -Destination ".\" -Force

git add redrob-ranker/README.md README.md
git commit -m "docs: add project README and finalize submission"
```

---

### Final: Force push
```powershell
git push origin main --force
```

---

## ⚠️ Files to SKIP (do NOT commit these)

| File | Reason |
|------|--------|
| `__pycache__/` | Build artifacts — already in .gitignore |
| `things to do` | Personal notes, unprofessional |
| `Hackathon info` | No extension, looks like scratch notes |
| `analysis_output.txt` | Generated output, not source code |
| `deep_analysis.py` | Appears to be a one-off script |
| `final_submission.csv` | Submission outputs shouldn't be in repo |
| `test_submission.csv` | Test outputs |
| `test_ltr_submission.csv` | Test outputs |
| `test_full_submission.csv` | Test outputs |
| `verify_submission.csv` | Test outputs |
| `run_weak_labels.bat` | Local Windows batch file |
| `[PUB] India_runs_data_and_ai_challenge/` | Raw challenge data |
| `redrob-ranker/artifacts/` | ~400MB+ binary files — way too large |
| `Hackathon_data_info/` | Raw challenge docs |

---

## Commit Naming Convention Used

Following [Conventional Commits](https://www.conventionalcommits.org/):
- `init:` — Project initialization
- `docs:` — Documentation only
- `feat:` — New feature/functionality
- `feat(scope):` — Feature scoped to a module

This gives you **15 clean, logical commits** that tell a story of methodical development: research → setup → core modules → pipeline → integration → docs.
