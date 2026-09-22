"""
Fill data/quality.csv from the explicit match file and the saved AA download.

Why this way: the match file names the exact Artificial Analysis models for each
OpenRouter model, and the scores come from one saved download. Another person can
repeat both steps.

Needs: data/aa_models.json (saved download), data/aa_manual_matches.csv
Usage:  python apply_matches.py
Output: data/quality.csv  (slug, quality, aa_variant, aa_slugs, note)

Rule: for a model with several effort or reasoning settings, take the highest score.
Attribution: Source: Artificial Analysis (artificialanalysis.ai).
"""
import json
import pandas as pd

DATA = "data"

js = json.load(open(f"{DATA}/aa_models.json"))
aa = {m["slug"]: (m.get("name", ""), (m.get("evaluations") or {}).get("artificial_analysis_intelligence_index"))
      for m in js.get("data", [])}
print(f"AA models in the download: {len(aa)}")

mt = pd.read_csv(f"{DATA}/aa_manual_matches.csv").fillna("")
rows, unknown = [], []
for r in mt.itertuples():
    variants = [v for v in str(r.aa_slugs).split(";") if v]
    unknown += [v for v in variants if v not in aa]
    scored = [(aa[v][1], aa[v][0]) for v in variants if v in aa and aa[v][1] is not None]
    best = max(scored) if scored else (None, "")
    rows.append({"slug": r.slug, "quality": best[0], "aa_variant": best[1],
                 "aa_slugs": r.aa_slugs, "note": r.note})

q = pd.DataFrame(rows)
q.to_csv(f"{DATA}/quality.csv", index=False)
if unknown:
    print(f"WARNING: names not in the download: {sorted(set(unknown))}")
print(f"filled {q['quality'].notna().sum()} of {len(q)} models")
print("no score:", ", ".join(q.loc[q['quality'].isna(), 'slug']) or "none")
