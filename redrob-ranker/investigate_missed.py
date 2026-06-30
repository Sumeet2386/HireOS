"""Investigate the top 10 missed candidates."""
import json
from pathlib import Path
from ranker.honeypot import detect_honeypot

missed = ['CAND_0080102', 'CAND_0062247', 'CAND_0010149', 'CAND_0008139', 'CAND_0030827',
          'CAND_0072688', 'CAND_0084090', 'CAND_0022852', 'CAND_0010603', 'CAND_0092181']

cands = {}
with open('../India_runs_data_and_ai_challenge/candidates.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        c = json.loads(line)
        if c['candidate_id'] in missed:
            cands[c['candidate_id']] = c

for cid in missed:
    if cid not in cands:
        continue
    c = cands[cid]
    is_hp, flags = detect_honeypot(c)
    p = c.get('profile', {})
    s = c.get('redrob_signals', {})
    title = p.get('current_title', '?')
    company = p.get('current_company', '?')
    yoe = p.get('years_of_experience', 0)
    saved = s.get('saved_by_recruiters_30d', 0)
    search = s.get('search_appearance_30d', 0)
    
    print(f"{cid}: {title} at {company}")
    print(f"  YoE={yoe:.1f}, saved={saved}, search={search}, is_hp={is_hp}, flags={len(flags)}")
    for fl in flags:
        print(f"    - {fl}")
    
    career = c.get('career_history', [])
    for job in career[:3]:
        desc = (job.get('description') or '')[:120]
        jtitle = job.get('title', '?')
        jcompany = job.get('company', '?')
        jdur = job.get('duration_months', 0)
        print(f"  Career: {jtitle} at {jcompany} ({jdur}mo)")
        print(f"    {desc}")
    
    skills = [sk.get('name', '') for sk in c.get('skills', [])[:10]]
    print(f"  Skills: {', '.join(skills)}")
    print()
