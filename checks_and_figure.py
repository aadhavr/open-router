"""
Idea 5 - Step 3: checks and first figure.

Needs (from collect_wayback.py): data/panel_hist.csv, data/wayback/*.json
Usage:  python checks_and_figure.py

Manual labels:
    The first run makes data/manual_labels.csv with the largest models.
    Fill the 'label' column with one of: open, closed, stealth, exclude
    (leave it empty to keep the automatic label). Then run the script again.

Output (in ./data):
    check_hf_field.csv          Hugging Face field in each snapshot
    review_top_models.csv       top 10 models per quarter, for the hand check
    manual_labels.csv           your corrections (you edit this file)
    weekly_shares.csv           weekly shares used in the figure
    fig_open_share.png          first figure
"""
import os, json, glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data"
LABELS = f"{DATA}/manual_labels.csv"
STEALTH_AUTHORS = {"stealth", "openrouter"}     # anonymous test models
EXCLUDE_WORDS = ["embed", "rerank", "openrouter/auto"]


def title(t):
    print(f"\n=== {t} ===")


def to_bool(s):
    return s.astype(str).str.lower().map({"true": True, "false": False})


panel = pd.read_csv(f"{DATA}/panel_hist.csv", parse_dates=["date"])
panel["open_weights"] = to_bool(panel["open_weights"])
panel["is_free"] = to_bool(panel["is_free"]).fillna(False)
panel["author"] = panel["slug"].str.split("/").str[0]

# ---------------------------------------------------------------- check 2
title("CHECK 2: Hugging Face field in the archived snapshots")
recs, last_seen = [], {}
for fn in sorted(glob.glob(f"{DATA}/wayback/models_*.json")):
    day = pd.to_datetime(os.path.basename(fn)[7:15], format="%Y%m%d")
    try:
        models = json.load(open(fn)).get("data", [])
    except Exception:
        continue
    has_key = sum("hugging_face_id" in m for m in models)
    has_val = sum(bool(m.get("hugging_face_id")) for m in models)
    recs.append({"date": day, "models": len(models),
                 "with_field": has_key, "with_link": has_val})
    for m in models:
        for k in (m.get("canonical_slug"), m.get("id")):
            if k:
                last_seen[k] = max(last_seen.get(k, day), day)
hf = pd.DataFrame(recs).sort_values("date")
hf.to_csv(f"{DATA}/check_hf_field.csv", index=False)
with_field = hf[hf["with_field"] > 0]
field_start = with_field["date"].min() if len(with_field) else None
print(f"snapshots: {len(hf)}, first snapshot with the field: "
      f"{field_start.date() if field_start is not None else 'NONE'}")

# Models marked closed that disappeared before the field existed: flag is uncertain
panel["last_seen"] = panel["slug"].map(last_seen)
panel["flag_uncertain"] = (
    (panel["open_weights"] == False)
    & (field_start is not None)
    & (panel["last_seen"] < field_start if field_start is not None else False))
unc = panel[panel["flag_uncertain"]]
tot_m = panel.groupby(pd.Grouper(key="date", freq="MS"))["total_tokens"].sum()
unc_m = unc.groupby(pd.Grouper(key="date", freq="MS"))["total_tokens"].sum()
print("share of tokens with an uncertain 'closed' flag, by month:")
print((unc_m / tot_m).fillna(0).round(3)[lambda s: s > 0].to_string() or "none")

# ------------------------------------------------------------ labels
auto = pd.Series("closed", index=panel.index)
auto[panel["open_weights"] == True] = "open"
auto[panel["open_weights"].isna()] = "unknown"
auto[panel["author"].isin(STEALTH_AUTHORS)] = "stealth"
for w in EXCLUDE_WORDS:
    auto[panel["slug"].str.contains(w, regex=False)] = "exclude"
panel["label_auto"] = auto

manual = {}
if os.path.exists(LABELS):
    ml = pd.read_csv(LABELS).fillna("")
    manual = {r.slug: r.label.strip().lower() for r in ml.itertuples() if r.label.strip()}
    bad = {k: v for k, v in manual.items() if v not in {"open", "closed", "stealth", "exclude"}}
    if bad:
        print(f"WARNING: unknown labels ignored: {bad}")
        manual = {k: v for k, v in manual.items() if k not in bad}
panel["label"] = panel["slug"].map(manual).fillna(panel["label_auto"])
print(f"\nmanual labels applied: {len(manual)}")

# Rule: a model is open only on dates when its weights are public.
# Column 'open_from' (YYYY-MM-DD): before this date an 'open' model counts as closed.
# Column 'revealed_as' (open/closed): later identity of a stealth model,
# used only in the sensitivity line.
panel["label_sens"] = panel["label"]
if os.path.exists(LABELS):
    ml = pd.read_csv(LABELS).fillna("")
    if "open_from" in ml:
        for r in ml[ml["open_from"].astype(str).str.len() > 0].itertuples():
            m = (panel["slug"] == r.slug) & (panel["date"] < pd.Timestamp(r.open_from)) & (panel["label"] == "open")
            panel.loc[m, "label"] = "closed"
            print(f"{r.slug}: closed before {r.open_from} ({m.sum()} rows)")
    panel["label_sens"] = panel["label"]
    if "revealed_as" in ml:
        rev = {r.slug: r.revealed_as.strip().lower() for r in ml.itertuples()
               if str(r.revealed_as).strip().lower() in {"open", "closed"}}
        m = (panel["label"] == "stealth") & panel["slug"].isin(rev)
        panel.loc[m, "label_sens"] = panel.loc[m, "slug"].map(rev)
        print(f"stealth models with a revealed identity (sensitivity only): {len(rev)}")

panel.to_csv(f"{DATA}/panel_labeled.csv", index=False)   # used by make_figure.py and price_analysis.py

# ---------------------------------------------------------------- check 4
title("CHECK 4: top 10 models per quarter (check the labels by hand)")
panel["quarter"] = panel["date"].dt.to_period("Q").astype(str)
top = (panel.groupby(["quarter", "slug", "label", "label_auto", "flag_uncertain"])
            ["total_tokens"].sum().reset_index())
top["share_of_quarter"] = top["total_tokens"] / top.groupby("quarter")["total_tokens"].transform("sum")
top = top.sort_values(["quarter", "total_tokens"], ascending=[True, False]).groupby("quarter").head(10)
top.to_csv(f"{DATA}/review_top_models.csv", index=False)
for q, g in top.groupby("quarter"):
    print(f"\n{q}")
    print(g[["slug", "label", "share_of_quarter"]].round(3).to_string(index=False))

# Make the manual label file once, with every model in any top-10 list
if not os.path.exists(LABELS):
    tmpl = top.drop_duplicates("slug")[["slug", "label_auto"]].copy()
    unk = panel.loc[panel["label"].isin(["unknown"]) | panel["flag_uncertain"], ["slug", "label_auto"]]
    tmpl = pd.concat([tmpl, unk]).drop_duplicates("slug")
    tmpl["label"] = ""
    tmpl["note"] = ""
    tmpl.to_csv(LABELS, index=False)
    print(f"\nmade {LABELS} with {len(tmpl)} models. Fill 'label' and run again.")

# ------------------------------------------------------ checks 1 and 3
def weekly(df, col="label"):
    g = df.groupby([pd.Grouper(key="date", freq="W-MON", label="left", closed="left"), col])
    w = g["total_tokens"].sum().unstack(fill_value=0)
    for c in ["open", "closed", "stealth", "unknown", "exclude"]:
        if c not in w:
            w[c] = 0
    known = w["open"] + w["closed"]
    return pd.DataFrame({"open_share": w["open"] / known,
                         "stealth_share": w["stealth"] / w.sum(axis=1)})

all_w = weekly(panel)
paid_w = weekly(panel[~panel["is_free"]])
sens_w = weekly(panel, "label_sens")
free_share = (panel[panel["is_free"]].groupby(pd.Grouper(key="date", freq="W-MON", label="left", closed="left"))["total_tokens"].sum()
              / panel.groupby(pd.Grouper(key="date", freq="W-MON", label="left", closed="left"))["total_tokens"].sum()).fillna(0)
out = pd.DataFrame({"open_share_all": all_w["open_share"],
                    "open_share_no_free": paid_w["open_share"],
                    "open_share_stealth_revealed": sens_w["open_share"],
                    "stealth_share": all_w["stealth_share"],
                    "free_share": free_share})
out.to_csv(f"{DATA}/weekly_shares.csv")

title("CHECKS 1 and 3: monthly means (open share excludes stealth, unknown, exclude)")
print(out.resample("MS").mean().round(3).to_string())

title("Largest models in July-August 2025 and December 2025 (the peaks)")
for a, b in [("2025-07-01", "2025-08-31"), ("2025-12-01", "2025-12-31")]:
    p = panel[(panel["date"] >= a) & (panel["date"] <= b)]
    s = p.groupby(["slug", "label", "is_free"])["total_tokens"].sum()
    print(f"\n{a} to {b}")
    print((s / s.sum()).nlargest(8).round(3).to_string())

# ------------------------------------------------------------- figure
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1.3]})
ax[0].plot(out.index, out["open_share_all"], label="All tokens", lw=2)
ax[0].plot(out.index, out["open_share_no_free"], label="Without free variants", lw=2, ls="--")
ax[0].plot(out.index, out["open_share_stealth_revealed"], label="Stealth models as later revealed",
           lw=1, ls=":", color="gray")
ax[0].set_ylabel("Open-weight share of tokens")
ax[0].set_ylim(0, 1)
ax[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax[0].legend(frameon=False)
ax[0].set_title("Open-weight models' share of OpenRouter tokens (weekly)")
ax[1].plot(out.index, out["free_share"], label="Free variants", lw=1.5)
ax[1].plot(out.index, out["stealth_share"], label="Stealth models", lw=1.5)
ax[1].set_ylabel("Share of all tokens")
ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax[1].legend(frameon=False)
as_of = pd.read_csv(f"{DATA}/rankings_daily.csv", nrows=1)["as_of"].iloc[0]
fig.text(0.01, 0.01, f"Source: OpenRouter (openrouter.ai/rankings), as of {as_of}. "
         "Top 50 models per day. Open share excludes stealth and unlabeled models.",
         fontsize=8, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{DATA}/fig_open_share.png", dpi=200)
print(f"\nsaved {DATA}/fig_open_share.png")
