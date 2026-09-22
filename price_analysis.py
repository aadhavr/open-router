"""
Idea 5 - Step 4: price analysis.

Question: when closed models become more expensive relative to open models,
does the open-weight share of paid tokens increase?

Needs: data/panel_labeled.csv (run checks_and_figure.py first)
Optional: data/quality.csv (slug, quality) - the first run makes a template.
Usage:  python price_analysis.py

Output (in ./data):
    weekly_prices.csv     weekly prices, ratios, shares, quality gap
    regression.txt        regression results
    fig_price.png         price ratio and open share over time

Scope: paid tokens only. Free variants, stealth, excluded and unlabeled models
are removed, because they have no real price or no open/closed label.
Prices are list prices, blended 3 input : 1 output, in USD per million tokens.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

DATA = "data"
QUALITY = f"{DATA}/quality.csv"
WEEK = pd.Grouper(key="date", freq="W-MON", label="left", closed="left")
MIN_COVER = 0.6     # a week needs this share of paid tokens with a quality score
FONT = "IBM Plex Sans"

# ------------------------------------------------------------ prices, rebuilt by variant
# Fix 1: free variants share the canonical slug with the paid model, so the old merge
#        sometimes gave a paid model the free price (0). Here each variant has its own key.
# Fix 2: old snapshots have no canonical_slug. Newer snapshots show which id belongs to
#        which canonical slug, so old ids are mapped to it.
# Fix 3: if no snapshot exists on or before a date, the next later snapshot is used.
def build_prices():
    h = pd.read_csv(f"{DATA}/price_history.csv", parse_dates=["snapshot_date"])
    h["id"] = h["id"].astype(str)
    h["base_id"] = h["id"].str.split(":").str[0]
    h["suffix"] = h["id"].str.partition(":")[2]
    h["canon"] = h["slug"].astype(str).str.split(":").str[0]
    known = h[h["canon"] != h["base_id"]].sort_values("snapshot_date")
    idmap = known.drop_duplicates("base_id", keep="last").set_index("base_id")["canon"]
    h["canon"] = h["base_id"].map(idmap).fillna(h["canon"])
    h["key"] = h["canon"] + np.where(h["suffix"] != "", ":" + h["suffix"], "")
    h = h.groupby(["snapshot_date", "key"], as_index=False)["blended_usd_per_mtok"].mean()
    h["key"] = h["key"].astype("string")
    h["snapshot_date"] = h["snapshot_date"].astype("datetime64[ns]")
    return h.sort_values("snapshot_date")


prices = build_prices()
p = pd.read_csv(f"{DATA}/panel_labeled.csv", parse_dates=["date"])
p["is_free"] = p["is_free"].astype(str).str.lower().eq("true")
p = p[p["date"] >= "2025-01-01"]
n0 = p["total_tokens"].sum()
p = p[p["label"].isin(["open", "closed"])].drop(columns=["blended_usd_per_mtok", "snapshot_date"], errors="ignore")
p["key"] = p["model_permaslug"].astype("string")
p["date"] = p["date"].astype("datetime64[ns]")
p = p.sort_values("date").reset_index(drop=True)
kw = dict(left_on="date", right_on="snapshot_date", by="key")
back = pd.merge_asof(p, prices, direction="backward", **kw)
fwd = pd.merge_asof(p, prices, direction="forward", **kw)
p["blended_usd_per_mtok"] = back["blended_usd_per_mtok"].fillna(fwd["blended_usd_per_mtok"])
p["price_source"] = np.where(back["blended_usd_per_mtok"].notna(), "earlier snapshot",
                    np.where(fwd["blended_usd_per_mtok"].notna(), "later snapshot", "none"))
p["price_age_days"] = np.where(p["price_source"] == "earlier snapshot",
                               (p["date"] - back["snapshot_date"]).dt.days,
                               (fwd["snapshot_date"] - p["date"]).dt.days)
# A model with a list price of 0 is free, even without a ':free' suffix
p["free_any"] = p["is_free"] | (p["blended_usd_per_mtok"] == 0)

cand = p[~p["free_any"]]
MONTH = pd.Grouper(key="date", freq="MS")
den = cand.groupby([MONTH, "label"])["total_tokens"].sum()
num = cand[cand["blended_usd_per_mtok"].isna()].groupby([MONTH, "label"])["total_tokens"].sum()
miss = (num.reindex(den.index).fillna(0) / den).unstack()
print("share of paid tokens still WITHOUT a price, by month (should be near 0):")
print(miss.round(3).to_string())
src = cand.groupby("price_source")["total_tokens"].sum() / cand["total_tokens"].sum()
print("\nprice source (share of paid tokens):")
print(src.round(3).to_string())
top_none = cand[cand["price_source"] == "none"].groupby("key")["total_tokens"].sum().nlargest(10)
if len(top_none):
    print("\nlargest models still without a price (trillions):")
    print((top_none / 1e12).round(2).to_string())
z = p.loc[~p["is_free"] & (p["blended_usd_per_mtok"] == 0)].groupby("key")["total_tokens"].sum()
print(f"\nmodels with list price 0 but no ':free' suffix (counted as free): {len(z)} models, "
      f"{z.sum() / n0:.1%} of all tokens")

p = p[~p["free_any"] & (p["blended_usd_per_mtok"] > 0)]
p.to_csv(f"{DATA}/panel_priced.csv", index=False)   # used by spend_split.py
print(f"\npaid, labeled tokens with a price: {p['total_tokens'].sum() / n0:.1%} of all top-50 tokens")
print(f"median age of the price used: {p['price_age_days'].median():.0f} days "
      f"(90th percentile: {p['price_age_days'].quantile(0.9):.0f})")

# ------------------------------------------------------------ quality (optional)
q = None
if os.path.exists(QUALITY):
    q = pd.read_csv(QUALITY).dropna(subset=["quality"])
    q = dict(zip(q["slug"], q["quality"]))
    print(f"quality scores loaded: {len(q)} models")
else:
    top = (p.groupby(["slug", "label"])["total_tokens"].sum()
             .sort_values(ascending=False).reset_index().head(60))
    top["quality"] = ""
    top["source"] = ""
    top[["slug", "label", "quality", "source"]].to_csv(QUALITY, index=False)
    print(f"made {QUALITY} with the top 60 paid models. Fill 'quality' "
          "(for example, the Artificial Analysis Intelligence Index) and run again.")

# ------------------------------------------------------------ weekly measures
def weekly_group(df):
    df = df.assign(_pt=df["blended_usd_per_mtok"] * df["total_tokens"])
    g = df.groupby(WEEK)
    out = pd.DataFrame({
        "tokens": g["total_tokens"].sum(),
        # token-weighted mean price: what the average paid token cost
        "price_w": g["_pt"].sum() / g["total_tokens"].sum(),
        "n_models": g["slug"].nunique(),
    })
    # unweighted geometric mean over models present that week: less affected by the share itself
    per_model = df.groupby([WEEK, "slug"])["blended_usd_per_mtok"].mean().reset_index()
    out["price_g"] = per_model.groupby("date")["blended_usd_per_mtok"].apply(lambda s: np.exp(np.log(s).mean()))
    if q:
        dq = df[df["slug"].isin(q)].assign(quality=lambda x: x["slug"].map(q))
        dq = dq.assign(_qt=dq["quality"] * dq["total_tokens"])
        gq = dq.groupby(WEEK)
        out["quality_w"] = gq["_qt"].sum() / gq["total_tokens"].sum()
        out["quality_coverage"] = gq["total_tokens"].sum() / out["tokens"]
    return out

op = weekly_group(p[p["label"] == "open"]).add_prefix("open_")
cl = weekly_group(p[p["label"] == "closed"]).add_prefix("closed_")
wk = op.join(cl, how="inner")
wk["open_share_paid"] = wk["open_tokens"] / (wk["open_tokens"] + wk["closed_tokens"])
wk["ratio_w"] = wk["closed_price_w"] / wk["open_price_w"]
wk["ratio_g"] = wk["closed_price_g"] / wk["open_price_g"]
if q:
    wk["quality_gap"] = wk["closed_quality_w"] - wk["open_quality_w"]
wk = wk[(wk["open_n_models"] >= 3) & (wk["closed_n_models"] >= 3)]   # drop thin weeks
if q:
    # Quality is known for some models only. The shares always use ALL paid tokens,
    # but the quality gap is trustworthy only when most tokens have a score.
    print("\nquality coverage (share of paid tokens with a score), monthly means:")
    print(wk[["open_quality_coverage", "closed_quality_coverage"]].resample("MS").mean().round(2).to_string())
    low = (wk["open_quality_coverage"] < MIN_COVER) | (wk["closed_quality_coverage"] < MIN_COVER)
    print(f"weeks dropped from the quality regressions (coverage < {MIN_COVER:.0%}): {low.sum()} of {len(wk)}")
    wk.loc[low, ["quality_gap"]] = None
wk.to_csv(f"{DATA}/weekly_prices.csv")

# Test: the open share of paid tokens here should be close to 'open_share_no_free' in weekly_shares.csv.
# Small differences are normal: this script also removes models with a list price of 0.
ws = pd.read_csv(f"{DATA}/weekly_shares.csv", index_col=0, parse_dates=True)
cmp = pd.DataFrame({"this_script": wk["open_share_paid"], "weekly_shares_no_free": ws["open_share_no_free"]})
print("\nTEST - open share of paid tokens, monthly means:")
print(cmp.resample("MS").mean().round(3).to_string())

print("\nmonthly means:")
cols = ["open_share_paid", "open_price_w", "closed_price_w", "ratio_w", "ratio_g"]
cols += ["quality_gap"] if q else []
print(wk[cols].resample("MS").mean().round(3).to_string())

# ------------------------------------------------------------ regressions
# logit(share) = a + b*log(price ratio) [+ c*quality gap] + d*trend
# b is the elasticity of the open/closed token ratio with respect to the price ratio.
# HAC (Newey-West) standard errors, 4 lags, because weekly errors are correlated.
wk["y"] = np.log(wk["open_share_paid"] / (1 - wk["open_share_paid"]))
wk["trend"] = np.arange(len(wk)) / 52.0          # years
lines = []

def ols_hac(y, X, lags=4):
    """OLS with Newey-West (HAC) standard errors. No extra package needed."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, se, r2


def run(name, xcols, df):
    d = df.dropna(subset=["y"] + xcols)
    X = np.column_stack([np.ones(len(d))] + [d[c] for c in xcols])
    b, se, r2 = ols_hac(d["y"], X)
    names = ["const"] + xcols
    rows = [f"{nm:<22}{bi:>10.3f}{si:>10.3f}{bi / si:>8.2f}" for nm, bi, si in zip(names, b, se)]
    lines.append(f"\n===== {name} (n={len(d)} weeks) =====\n"
                 f"{'variable':<22}{'coef':>10}{'HAC se':>10}{'t':>8}\n" + "\n".join(rows)
                 + f"\nR2 = {r2:.3f}")


for r in ["ratio_w", "ratio_g"]:
    wk[f"log_{r}"] = np.log(wk[r])
    run(f"Levels: log {r} + trend", [f"log_{r}", "trend"], wk)
    if q:
        run(f"Levels: log {r} + quality gap + trend", [f"log_{r}", "quality_gap", "trend"], wk)
    # first differences: week-to-week changes, removes slow common trends
    dd = wk[["y", f"log_{r}"] + (["quality_gap"] if q else [])].diff().dropna()
    run(f"First differences: d log {r}", [f"log_{r}"] + (["quality_gap"] if q else []), dd)

text = "\n".join(lines)
text += ("\n\nHow to read b (the coefficient on log price ratio): if closed models become 10% more "
         "expensive relative to open models, the open/closed token ratio changes by about b x 10%.\n"
         "Cautions: prices are list prices; the weighted ratio depends on the shares themselves "
         "(use ratio_g as the main result); weekly data are few; this is association, not a causal effect.")
open(f"{DATA}/regression.txt", "w").write(text)
print(text)

# ------------------------------------------------------------ figure
names = {f.name for f in font_manager.fontManager.ttflist}
if FONT in names:
    plt.rcParams["font.family"] = FONT
plt.rcParams.update({"axes.spines.top": False})
fig, ax1 = plt.subplots(figsize=(11, 5))
avg = wk.rolling(4, min_periods=4).mean()
ax1.plot(avg.index, avg["ratio_g"], color="#1f5fa8", lw=2, label="Price ratio, closed ÷ open (model average)")
ax1.plot(avg.index, avg["ratio_w"], color="#1f5fa8", lw=1, ls="--", label="Price ratio, closed ÷ open (token-weighted)")
ax1.set_yscale("log")
ax1.set_ylabel("Closed price ÷ open price (log scale)")
ax2 = ax1.twinx()
ax2.plot(avg.index, avg["open_share_paid"], color="#e07b24", lw=2, label="Open share of paid tokens")
ax2.set_ylim(0, 1)
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax2.set_ylabel("Open share of paid tokens")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=9)
ax1.set_title("Price gap and open-weight share (4-week averages)", loc="left", fontsize=14)
fig.text(0.01, 0.01, "Source: OpenRouter list prices and rankings. Paid tokens only; "
         "prices blended 3 input : 1 output.", fontsize=7.5, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{DATA}/fig_price.png", dpi=220)
print(f"\nsaved {DATA}/fig_price.png")
