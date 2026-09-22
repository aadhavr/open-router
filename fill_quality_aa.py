"""
Fill data/quality.csv with the Artificial Analysis Intelligence Index from the free API.

Why the API: the website shows the newest index version only for current models.
Older models show scores from older index versions, and those are not comparable.
The API returns one index version for every model in one request.

One-time setup:
    1. Make a free account at artificialanalysis.ai and create an API key.
    2. export AA_API_KEY=your_key
Usage:
    python fill_quality_aa.py
Attribution is required: "Source: Artificial Analysis (artificialanalysis.ai)".

Output:
    data/aa_models.json      raw API response (kept as evidence)
    data/aa_matches.csv      every match, with all candidate variants - REVIEW THIS
    data/quality.csv         'quality' filled from the API; your old values stay in 'quality_leaderboard'
"""
import os, re, json, sys
import requests
import pandas as pd

DATA = "data"
URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
KEY = os.environ.get("AA_API_KEY")
if not KEY:
    sys.exit("Set AA_API_KEY first (see top of file).")

# Words that only name a variant of the same model (effort level, reasoning mode)
VARIANT_WORDS = {"reasoning", "thinking", "non", "max", "xhigh", "high", "medium", "low",
                 "minimal", "adaptive", "with", "fallback", "effort", "default", "instruct", "chat"}

r = requests.get(URL, headers={"x-api-key": KEY}, timeout=60)
r.raise_for_status()
js = r.json()
json.dump(js, open(f"{DATA}/aa_models.json", "w"))
version = js.get("intelligence_index_version", "not given")
rows = []
for m in js.get("data", []):
    ev = m.get("evaluations") or {}
    rows.append({"aa_slug": m.get("slug", ""), "aa_name": m.get("name", ""),
                 "index": ev.get("artificial_analysis_intelligence_index")})
aa = pd.DataFrame(rows).dropna(subset=["index"])
print(f"API: {len(aa)} models with an Intelligence Index, version {version}")


def key_of(or_slug):
    k = or_slug.split("/", 1)[1].split(":")[0].lower()
    k = re.sub(r"-(20\d{6}|20\d{2}-\d{2}-\d{2}|\d{2}-\d{2}|\d{4})$", "", k)   # drop date suffix
    k = re.sub(r"-(it|instruct|preview|exp|experimental)$", "", k)
    return k.replace(".", "-")


def candidates(key):
    out = []
    for row in aa.itertuples():
        s = row.aa_slug.lower()
        if s == key:
            out.append(row)
        elif s.startswith(key + "-"):
            rest = s[len(key) + 1:].split("-")
            if all(w in VARIANT_WORDS for w in rest):     # only variant words, no other version
                out.append(row)
    return out


q = pd.read_csv(f"{DATA}/quality.csv")
if "quality_leaderboard" not in q:
    q["quality_leaderboard"] = q.get("quality")
match_rows = []
for i, s in q["slug"].items():
    c = candidates(key_of(s))
    if c:
        best = max(c, key=lambda x: x.index)                 # rule: highest variant of the model
        q.loc[i, "quality"] = best.index
        q.loc[i, "source"] = f"AA API v{version}: {best.aa_name} (highest of {len(c)} variants)"
        match_rows.append({"slug": s, "key": key_of(s), "chosen": best.aa_name, "index": best.index,
                           "all_variants": "; ".join(f"{x.aa_name}={x.index}" for x in c)})
    else:
        q.loc[i, "quality"] = None
        q.loc[i, "source"] = "no AA match - fill by hand or leave empty"
        match_rows.append({"slug": s, "key": key_of(s), "chosen": "", "index": None, "all_variants": ""})

pd.DataFrame(match_rows).to_csv(f"{DATA}/aa_matches.csv", index=False)
q.to_csv(f"{DATA}/quality.csv", index=False)

n = q["quality"].notna().sum()
print(f"matched {n} of {len(q)} models. Unmatched:")
print("\n".join("  " + s for s in q.loc[q["quality"].isna(), "slug"]))
both = q.dropna(subset=["quality", "quality_leaderboard"])
if len(both):
    diff = (both["quality"] - both["quality_leaderboard"]).abs()
    print(f"\ncheck vs leaderboard values: {len(both)} models, max difference {diff.max():.1f}")
print("\nReview data/aa_matches.csv before you use the scores.")
print("To find an AA name for an unmatched model: search aa_models.json, then put the score in quality.csv by hand.")
