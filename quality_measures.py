"""
Idea 5 - Step 5: quality measures that do not use token weights.

Why: the token-weighted quality gap moves with the shares by arithmetic. These two
measures cannot do that.

    1. Frontier gap  = best closed score - best open score, among all models
       released up to that date (the choice a buyer sees at the top).
    2. Unweighted gap = mean closed score - mean open score, over the models in the
       top 50 that week, each model counted once.

Needs: data/aa_models.json, data/aa_manual_matches.csv, data/quality.csv,
       data/panel_labeled.csv, data/weekly_prices.csv (run price_analysis.py first)
Usage: python quality_measures.py

Output (in ./data):
    weekly_quality.csv      both gaps, coverage, open share
    quality_regressions.txt
    fig_quality.png
Attribution: Source: Artificial Analysis (artificialanalysis.ai); OpenRouter.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

DATA = "data"
WEEK = pd.Grouper(key="date", freq="W-MON", label="left", closed="left")
MIN_COVER = 0.6          # a week needs this share of top-50 models with a score
FONT = "IBM Plex Sans"

# ------------------------------------------------------------ labels for AA models
panel = pd.read_csv(f"{DATA}/panel_labeled.csv", parse_dates=["date"])
panel["is_free"] = panel["is_free"].astype(str).str.lower().eq("true")
panel = panel[panel["label"].isin(["open", "closed"]) & ~panel["is_free"]]
label_of = panel.drop_duplicates("slug").set_index("slug")["label"]

mt = pd.read_csv(f"{DATA}/aa_manual_matches.csv").fillna("")
aa_label = {}
for r in mt.itertuples():
    lab = label_of.get(r.slug)
    if lab in ("open", "closed"):
        for v in str(r.aa_slugs).split(";"):
            if v:
                aa_label[v] = lab

js = json.load(open(f"{DATA}/aa_models.json"))
rows = []
for m in js.get("data", []):
    s = m.get("slug")
    score = (m.get("evaluations") or {}).get("artificial_analysis_intelligence_index")
    if s in aa_label and score is not None and m.get("release_date"):
        rows.append({"aa_slug": s, "label": aa_label[s], "score": float(score),
                     "released": pd.to_datetime(m["release_date"])})
aa = pd.DataFrame(rows)
print(f"AA models with a label, a score and a release date: {len(aa)} "
      f"({(aa['label'] == 'open').sum()} open, {(aa['label'] == 'closed').sum()} closed)")

# ------------------------------------------------------------ 1. frontier gap
wk_index = pd.read_csv(f"{DATA}/weekly_prices.csv", index_col=0, parse_dates=True)
weeks = wk_index.index
front = []
for w in weeks:
    seen = aa[aa["released"] <= w]
    o = seen.loc[seen["label"] == "open", "score"]
    c = seen.loc[seen["label"] == "closed", "score"]
    front.append({"date": w,
                  "front_open": o.max() if len(o) else np.nan,
                  "front_closed": c.max() if len(c) else np.nan})
front = pd.DataFrame(front).set_index("date")
front["frontier_gap"] = front["front_closed"] - front["front_open"]

# ------------------------------------------------------------ 2. unweighted gap
q = pd.read_csv(f"{DATA}/quality.csv")
q["quality"] = pd.to_numeric(q["quality"], errors="coerce")
panel["quality"] = panel["slug"].map(dict(zip(q["slug"], q["quality"])))
per_model = panel.groupby([WEEK, "slug", "label"], as_index=False)["quality"].first()
g = per_model.groupby(["date", "label"])
unw = pd.DataFrame({"mean_q": g["quality"].mean(),
                    "n_models": g["quality"].size(),
                    "n_scored": g["quality"].count()}).unstack()
unw.columns = [f"{a}_{b}" for a, b in unw.columns]
unw["unweighted_gap"] = unw["mean_q_closed"] - unw["mean_q_open"]
unw["cover_open"] = unw["n_scored_open"] / unw["n_models_open"]
unw["cover_closed"] = unw["n_scored_closed"] / unw["n_models_closed"]

wk = wk_index[["open_share_paid", "ratio_g", "ratio_w"]].join([front, unw], how="inner")
low = (wk["cover_open"] < MIN_COVER) | (wk["cover_closed"] < MIN_COVER)
print(f"weeks with model coverage below {MIN_COVER:.0%}: {low.sum()} of {len(wk)}")
# The relative measure is built from these columns, so remove the same weeks there too.
wk.loc[low, ["unweighted_gap", "mean_q_open", "mean_q_closed"]] = np.nan
wk.to_csv(f"{DATA}/weekly_quality.csv")

print("\nmonthly means:")
cols = ["open_share_paid", "front_open", "front_closed", "frontier_gap",
        "mean_q_open", "mean_q_closed", "unweighted_gap"]
print(wk[cols].resample("MS").mean().round(2).to_string())


# ------------------------------------------------------------ regressions
def ols_hac(y, X, lags=4):
    y, X = np.asarray(y, float), np.asarray(X, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        wgt = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += wgt * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, np.sqrt(np.diag(V)), r2


lines = []


def run(name, xcols, df):
    d = df.dropna(subset=["y"] + xcols)
    flat = [c for c in xcols if d[c].nunique() <= 1]
    if len(d) < 10 or flat:
        why = f"only {len(d)} weeks" if len(d) < 10 else f"no change in {flat}"
        lines.append(f"\n===== {name} =====\nnot estimated ({why})")
        return
    X = np.column_stack([np.ones(len(d))] + [d[c] for c in xcols])
    try:
        b, se, r2 = ols_hac(d["y"], X)
    except np.linalg.LinAlgError:
        lines.append(f"\n===== {name} =====\nnot estimated (variables are collinear)")
        return
    rows = [f"{nm:<20}{bi:>10.3f}{si:>10.3f}{bi / si:>8.2f}"
            for nm, bi, si in zip(["const"] + xcols, b, se)]
    warn = "\nWARNING: standard errors not reliable (variables almost collinear)" \
        if np.isnan(se).any() else ""
    lines.append(f"\n===== {name} (n={len(d)} weeks) =====\n"
                 f"{'variable':<20}{'coef':>10}{'HAC se':>10}{'t':>8}\n" + "\n".join(rows)
                 + f"\nR2 = {r2:.3f}" + warn)


wk["y"] = np.log(wk["open_share_paid"] / (1 - wk["open_share_paid"]))
wk["trend"] = np.arange(len(wk)) / 52.0
wk["log_ratio_g"] = np.log(wk["ratio_g"])
# The index scale itself grows over time, so a difference of 6 points means less in 2026
# than in 2025. The relative measure divides instead of subtracting.
wk["frontier_rel"] = np.log(wk["front_closed"] / wk["front_open"])
wk["unweighted_rel"] = np.log(wk["mean_q_closed"] / wk["mean_q_open"])
print("\nrelative gaps (log closed/open), monthly means:")
print(wk[["frontier_rel", "unweighted_rel", "open_share_paid"]].resample("MS").mean().round(3).to_string())
# Users cannot change model on the day a model comes out, so test earlier weeks too.
LAGS = (1, 2, 4)
for lag in LAGS:
    wk[f"frontier_rel_lag{lag}"] = wk["frontier_rel"].shift(lag)
measures = ["frontier_gap", "frontier_rel", "unweighted_gap", "unweighted_rel"]
measures += [f"frontier_rel_lag{l}" for l in LAGS]
for gap in measures:
    run(f"Levels: {gap} + trend", [gap, "trend"], wk)
    run(f"Levels: {gap} + log price ratio + trend", [gap, "log_ratio_g", "trend"], wk)
    d = wk[["y", gap, "log_ratio_g"]].diff().dropna()
    run(f"First differences: d {gap}", [gap], d)
    run(f"First differences: d {gap} + d log price ratio", [gap, "log_ratio_g"], d)

text = "\n".join(lines) + (
    "\n\nA negative coefficient on a gap means: the open share is higher when closed models "
    "lead by less.\nNeither gap uses token weights, so a mechanical link with the share is not "
    "possible.\nStill association, not a causal effect: a strong new open model changes the gap "
    "and the share in the same week.")
open(f"{DATA}/quality_regressions.txt", "w").write(text)
print(text)

# ------------------------------------------------------------ figure
wk.to_csv(f"{DATA}/weekly_quality.csv")   # save again, now with the relative gaps and lags
names = {f.name for f in font_manager.fontManager.ttflist}
if FONT in names:
    plt.rcParams["font.family"] = FONT
plt.rcParams.update({"axes.spines.top": False})
avg = wk.rolling(4, min_periods=4).mean()
fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                       gridspec_kw={"height_ratios": [3, 2]})
BLUE, ORANGE, GRAY = "#1f5fa8", "#e07b24", "#8a8a8a"
ax[0].plot(avg.index, avg["front_closed"], color=BLUE, lw=2, label="Best closed model")
ax[0].plot(avg.index, avg["front_open"], color=ORANGE, lw=2, label="Best open model")
ax[0].plot(avg.index, avg["mean_q_closed"], color=BLUE, lw=1, ls="--", label="Closed, top-50 mean")
ax[0].plot(avg.index, avg["mean_q_open"], color=ORANGE, lw=1, ls="--", label="Open, top-50 mean")
ax[0].set_ylabel("Artificial Analysis Intelligence Index")
ax[0].legend(frameon=False, fontsize=9, loc="upper left")
ax[0].set_title("Model quality and the open-weight share", loc="left", fontsize=14, pad=12)
ax[0].grid(axis="y", color="#e6e6e6", lw=0.8)
ax[1].plot(avg.index, avg["frontier_gap"], color=GRAY, lw=2, label="Frontier gap (closed - open)")
ax[1].plot(avg.index, avg["unweighted_gap"], color=GRAY, lw=1.2, ls="--", label="Top-50 mean gap")
ax[1].axhline(0, color="#cccccc", lw=0.8)
ax[1].set_ylabel("Quality gap (index points)")
ax[1].legend(frameon=False, fontsize=9, loc="upper left")
ax2 = ax[1].twinx()
ax2.plot(avg.index, avg["open_share_paid"], color=ORANGE, lw=2, label="Open share of paid tokens")
ax2.set_ylim(0, 1)
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
ax2.set_ylabel("Open share of paid tokens")
ax2.legend(frameon=False, fontsize=9, loc="lower right")
fig.text(0.01, 0.01, "Sources: Artificial Analysis (artificialanalysis.ai) and OpenRouter. "
         "4-week averages. Frontier: best score among models released up to that week that "
         "appear on OpenRouter.", fontsize=7.5, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{DATA}/fig_quality.png", dpi=220)
print(f"\nsaved {DATA}/fig_quality.png")
