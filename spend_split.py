"""
Idea 5 - Step 7: who collects the money?

Tokens x price = spend. Open models have most of the tokens, but they cost much less,
so their share of spend is smaller. This script measures both shares over time and
the concentration of spend by model creator.

Needs: data/panel_priced.csv (run price_analysis.py first)
Usage: python spend_split.py

Output (in ./data):
    weekly_spend.csv     tokens, spend and shares for each week
    spend_by_creator.csv spend share for each creator, by quarter
    spend_summary.txt    the numbers for the post
    fig_spend.png        token share and spend share, plus concentration

Warning about the words: this is spend on one router, not revenue of the model makers.
Many providers sell the same open model, so most open-model spend goes to the
inference provider, not to the lab that made the model.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

DATA = "data"
WEEK = pd.Grouper(key="date", freq="W-MON", label="left", closed="left")
FONT = "IBM Plex Sans"

p = pd.read_csv(f"{DATA}/panel_priced.csv", parse_dates=["date"])
# Blended price (3 input : 1 output) applied to all tokens of the row.
# OpenRouter reports one token total, so this is an approximation.
p["spend_usd"] = p["total_tokens"] * p["blended_usd_per_mtok"] / 1e6
p["creator"] = p["slug"].str.split("/").str[0]
print(f"weeks: {p['date'].dt.to_period('W').nunique()} | creators: {p['creator'].nunique()}")

# Check 1: a creator with models on both sides. A wrong label moves both shares.
mix = p.groupby("creator")["label"].nunique()
for c in mix[mix > 1].index:
    sub = p[p["creator"] == c].groupby(["label", "slug"])["total_tokens"].sum()
    print(f"\ncreator '{c}' has open and closed models - confirm these labels:")
    print((sub / 1e12).round(2).to_string())

# Check 2: the early months set the eye-catching first number, so list what drives them.
early = p[p["date"] < "2025-03-01"]
if len(early):
    e = (early.groupby(["slug", "label"])
              .agg(tokens=("total_tokens", "sum"),
                   price=("blended_usd_per_mtok", "median"),
                   source=("price_source", lambda x: x.mode()[0]))
              .sort_values("tokens", ascending=False).head(10))
    e["tokens"] = (e["tokens"] / 1e12).round(2)
    print("\nlargest paid models before March 2025 (check these prices by hand):")
    print(e.round(3).to_string())

# ------------------------------------------------------------ weekly shares
g = p.groupby([WEEK, "label"])[["total_tokens", "spend_usd"]].sum().unstack()
g.columns = [f"{a}_{b}" for a, b in g.columns]
wk = pd.DataFrame({
    "tokens_total": g["total_tokens_open"] + g["total_tokens_closed"],
    "spend_total": g["spend_usd_open"] + g["spend_usd_closed"],
    "token_share_open": g["total_tokens_open"] / (g["total_tokens_open"] + g["total_tokens_closed"]),
    "spend_share_open": g["spend_usd_open"] / (g["spend_usd_open"] + g["spend_usd_closed"]),
})
wk["spend_per_mtok"] = 1e6 * wk["spend_total"] / wk["tokens_total"]
wk["share_gap"] = wk["token_share_open"] - wk["spend_share_open"]

# ------------------------------------------------------------ concentration
def conc(df, col):
    s = df.groupby([WEEK, "creator"])[col].sum()
    tot = s.groupby(level=0).sum()
    sh = s / tot
    hhi = (sh ** 2).groupby(level=0).sum()
    top3 = sh.groupby(level=0).nlargest(3).groupby(level=0).sum()
    return hhi, top3

wk["hhi_spend"], wk["top3_spend"] = conc(p, "spend_usd")
wk["hhi_tokens"], wk["top3_tokens"] = conc(p, "total_tokens")
wk.to_csv(f"{DATA}/weekly_spend.csv")

# ------------------------------------------------------------ creators by quarter
p["quarter"] = p["date"].dt.to_period("Q").astype(str)
cre = p.groupby(["quarter", "creator", "label"])[["total_tokens", "spend_usd"]].sum().reset_index()
cre["spend_share"] = cre["spend_usd"] / cre.groupby("quarter")["spend_usd"].transform("sum")
cre["token_share"] = cre["total_tokens"] / cre.groupby("quarter")["total_tokens"].transform("sum")
cre["price_per_mtok"] = 1e6 * cre["spend_usd"] / cre["total_tokens"]
avg_q = (1e6 * cre.groupby("quarter")["spend_usd"].sum() / cre.groupby("quarter")["total_tokens"].sum())
cre["price_vs_average"] = cre["price_per_mtok"] / cre["quarter"].map(avg_q)
cre.sort_values(["quarter", "spend_share"], ascending=[True, False]).to_csv(
    f"{DATA}/spend_by_creator.csv", index=False)

# ------------------------------------------------------------ summary
out = []
m = wk.resample("MS").mean()
out.append("Monthly means: token share and spend share of open models\n")
out.append(m[["token_share_open", "spend_share_open", "share_gap", "spend_per_mtok",
              "hhi_spend", "top3_spend"]].round(3).to_string())

first, last = m.index[0], m.index[-1]
out.append(f"\n\nFirst month ({first:%Y-%m}): tokens {m.loc[first, 'token_share_open']:.1%}, "
           f"spend {m.loc[first, 'spend_share_open']:.1%}")
out.append(f"Last month ({last:%Y-%m}): tokens {m.loc[last, 'token_share_open']:.1%}, "
           f"spend {m.loc[last, 'spend_share_open']:.1%}")
out.append(f"Change: tokens {m.loc[last, 'token_share_open'] - m.loc[first, 'token_share_open']:+.1%} points, "
           f"spend {m.loc[last, 'spend_share_open'] - m.loc[first, 'spend_share_open']:+.1%} points")
out.append("\nTest of the simple arithmetic: if the share of spend followed only from the token "
           "share and\nthe average price gap, it would be:")
ratio = m["spend_per_mtok"]  # only for context
for d in [first, last]:
    t = m.loc[d, "token_share_open"]
    s = m.loc[d, "spend_share_open"]
    implied_gap = (t / (1 - t)) * ((1 - s) / s)
    out.append(f"  {d:%Y-%m}: closed price / open price implied by the two shares = {implied_gap:.1f}x")

out.append("\n\nTop creators by spend share, last 4 quarters")
for qr in sorted(cre["quarter"].unique())[-4:]:
    sub = cre[cre["quarter"] == qr].nlargest(8, "spend_share")
    out.append(f"\n{qr}")
    out.append(sub[["creator", "label", "spend_share", "token_share", "price_per_mtok",
                    "price_vs_average"]].round(3).to_string(index=False))

# Where the money goes: a closed model is sold by the lab that made it. An open model
# is sold by whichever provider hosts it, so that spend goes to the host, not to the lab.
dest = p.groupby([WEEK, "label"])["spend_usd"].sum().unstack().fillna(0)
dest["to_labs"] = dest["closed"] / (dest["open"] + dest["closed"])
dest["to_hosts"] = dest["open"] / (dest["open"] + dest["closed"])
dm = dest.resample("MS").mean()
out.append("\n\nWhere the money goes: labs that sell their own model, or providers that host "
           "open weights")
out.append(dm[["to_labs", "to_hosts"]].round(3).to_string())
out.append(f"\nLast month: {dm['to_labs'].iloc[-1]:.1%} of router spend went to the lab that made "
           f"the model; {dm['to_hosts'].iloc[-1]:.1%} went to a hosting provider.")

out.append("\n\nConcentration of TOKENS and SPEND BY MODEL CREATOR (HHI, 1.0 = one creator "
           "has everything).\nRead this as concentration by who MADE the model. It is not "
           "concentration of the inference market:\nfor an open-weight model the money goes to "
           "whichever provider hosts it, and this dataset does not\nname the host. So the "
           "concentration among hosting providers is unknown here, and it is probably\nhigher "
           "than the creator numbers suggest.")
out.append(m[["hhi_tokens", "hhi_spend", "top3_tokens", "top3_spend"]].round(3).to_string())

text = "\n".join(out)
open(f"{DATA}/spend_summary.txt", "w").write(text)
print(text)

# ------------------------------------------------------------ figure
names = {f.name for f in font_manager.fontManager.ttflist}
if FONT in names:
    plt.rcParams["font.family"] = FONT
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
avg = wk.rolling(4, min_periods=4).mean()
fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                       gridspec_kw={"height_ratios": [3, 2]})
BLUE, ORANGE = "#1f5fa8", "#e07b24"
ax[0].plot(avg.index, avg["token_share_open"], color=ORANGE, lw=2.4, label="Share of tokens")
ax[0].plot(avg.index, avg["spend_share_open"], color=BLUE, lw=2.4, label="Share of spend")
ax[0].fill_between(avg.index, avg["spend_share_open"], avg["token_share_open"],
                   color=ORANGE, alpha=0.12)
ax[0].set_ylim(0, 1)
ax[0].yaxis.set_major_formatter(PercentFormatter(1.0))
ax[0].set_ylabel("Open-weight models")
ax[0].set_title("Open models take the tokens; closed models take the money", loc="left",
                fontsize=14, pad=12)
ax[0].legend(frameon=False, fontsize=9, loc="upper left")
ax[0].grid(axis="y", color="#e6e6e6", lw=0.8)
ax[1].plot(avg.index, avg["top3_tokens"], color=ORANGE, lw=1.8,
           label="Top 3 model creators, tokens")
ax[1].plot(avg.index, avg["top3_spend"], color=BLUE, lw=1.8,
           label="Top 3 model creators, spend")
ax[1].set_ylim(0, 1)
ax[1].yaxis.set_major_formatter(PercentFormatter(1.0))
ax[1].set_ylabel("Concentration by\nmodel creator")
ax[1].legend(frameon=False, fontsize=9, loc="upper left")
ax[1].grid(axis="y", color="#e6e6e6", lw=0.8)
fig.text(0.01, 0.01, "Source: OpenRouter rankings and list prices. Paid tokens only; prices blended "
         "3 input : 1 output. 4-week averages.\nSpend on this router, not revenue of the model makers: "
         "for an open-weight model the money goes to the provider that hosts it.\nThe lower panel "
         "groups spend by who made the model, not by who sold it; concentration among hosting "
         "providers is not measured here.", fontsize=7.5, color="gray")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{DATA}/fig_spend.png", dpi=220)
print(f"\nsaved {DATA}/fig_spend.png")
