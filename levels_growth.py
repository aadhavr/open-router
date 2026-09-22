"""
Idea 5 - Step 8: substitution or differential growth?

The share can rise two ways:
  (a) SUBSTITUTION - closed volume falls while open volume grows. People switch.
  (b) DIFFERENTIAL GROWTH - both grow, open faster. Almost nobody switches; new
      work goes to open models.
Price and quality explanations only make sense under (a). This script decides which.

Needs: data/panel_priced.csv (run price_analysis.py first)
Usage: python levels_growth.py

Output (in ./data):
    weekly_levels.csv     weekly tokens and spend, by label, in levels
    levels_summary.txt    growth rates, peaks, decomposition
    fig_levels.png        absolute volumes (log scale) plus the share
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
p["spend_usd"] = p["total_tokens"] * p["blended_usd_per_mtok"] / 1e6

g = p.groupby([WEEK, "label"])[["total_tokens", "spend_usd"]].sum().unstack().fillna(0)
g.columns = [f"{a}_{b}" for a, b in g.columns]
wk = g.rename(columns={"total_tokens_open": "tokens_open", "total_tokens_closed": "tokens_closed",
                       "spend_usd_open": "spend_open", "spend_usd_closed": "spend_closed"})
wk = wk[(wk["tokens_open"] > 0) & (wk["tokens_closed"] > 0)]
wk["tokens_total"] = wk["tokens_open"] + wk["tokens_closed"]
wk["share_open"] = wk["tokens_open"] / wk["tokens_total"]
wk.to_csv(f"{DATA}/weekly_levels.csv")

out = []
m = wk.resample("MS").mean()
out.append("Monthly means: tokens per week (trillions) and spend per week (USD millions)\n")
tbl = pd.DataFrame({
    "tokens_open_T": m["tokens_open"] / 1e12,
    "tokens_closed_T": m["tokens_closed"] / 1e12,
    "share_open": m["share_open"],
    "spend_open_Musd": m["spend_open"] / 1e6,
    "spend_closed_Musd": m["spend_closed"] / 1e6,
})
out.append(tbl.round(2).to_string())

first, last = m.index[0], m.index[-1]
years = (last - first).days / 365.25
out.append("\n\n=== Did closed volume fall? ===")
for col, name in [("tokens_closed", "closed tokens"), ("tokens_open", "open tokens"),
                  ("spend_closed", "closed spend"), ("spend_open", "open spend")]:
    a, b = m[col].iloc[0], m[col].iloc[-1]
    cagr = (b / a) ** (1 / years) - 1
    peak_at = m[col].idxmax()
    out.append(f"{name:<15} {first:%Y-%m}: {a:.3g}  {last:%Y-%m}: {b:.3g}  "
               f"x{b / a:.1f} over the period ({cagr:+.0%}/yr)  peak month: {peak_at:%Y-%m}"
               f" ({m[col].max() / b:.2f}x the last month)")

# Peak-to-now for closed: the clearest sign of real substitution
pk = m["tokens_closed"].idxmax()
out.append(f"\nClosed tokens since their peak ({pk:%Y-%m}): "
           f"{m['tokens_closed'].iloc[-1] / m['tokens_closed'].max() - 1:+.0%}")
out.append("Read: a large negative number means users left closed models (substitution). "
           "A number near zero or positive means closed use held up and the share moved "
           "because open use grew faster (differential growth).")

# Counterfactual: hold closed volume at its own path, ask what share open growth alone gives
out.append("\n\n=== How much of the share change needs substitution? ===")
t0, t1 = m.index[0], m.index[-1]
o0, c0 = m.loc[t0, "tokens_open"], m.loc[t0, "tokens_closed"]
o1, c1 = m.loc[t1, "tokens_open"], m.loc[t1, "tokens_closed"]
share0, share1 = o0 / (o0 + c0), o1 / (o1 + c1)
share_if_closed_flat = o1 / (o1 + c0)      # closed frozen at its starting level
share_if_open_flat = o0 / (o0 + c1)        # open frozen at its starting level
out.append(f"actual share: {share0:.1%} -> {share1:.1%}")
out.append(f"if open grew as observed and closed had stayed at its {t0:%Y-%m} level: "
           f"{share_if_closed_flat:.1%}")
out.append(f"if closed grew as observed and open had stayed at its {t0:%Y-%m} level: "
           f"{share_if_open_flat:.1%}")
out.append("The first line isolates open growth; the second isolates closed growth.")

# Weekly co-movement: do the two volumes move together or against each other?
d = np.log(wk[["tokens_open", "tokens_closed"]]).diff().dropna()
r = d.corr().iloc[0, 1]
out.append(f"\n\nCorrelation of weekly log changes, open vs closed volume: {r:+.2f}")
out.append("Negative would mean one grows while the other shrinks, week to week "
           "(substitution). Positive means both rise and fall together, which is what "
           "common demand shocks look like.")

text = "\n".join(out)
open(f"{DATA}/levels_summary.txt", "w").write(text)
print(text)

# ------------------------------------------------------------ figure
names = {f.name for f in font_manager.fontManager.ttflist}
if FONT in names:
    plt.rcParams["font.family"] = FONT
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
avg = wk.rolling(4, min_periods=4).mean()
fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1.6]})
BLUE, ORANGE = "#1f5fa8", "#e07b24"
ax[0].plot(avg.index, avg["tokens_open"] / 1e12, color=ORANGE, lw=2.4, label="Open-weight models")
ax[0].plot(avg.index, avg["tokens_closed"] / 1e12, color=BLUE, lw=2.4, label="Closed models")
ax[0].set_yscale("log")
ax[0].set_ylabel("Paid tokens per week (trillions, log scale)")
ax[0].set_title("Both sides grew. Open grew faster.", loc="left", fontsize=14, pad=12)
ax[0].legend(frameon=False, fontsize=9, loc="upper left")
ax[0].grid(axis="y", color="#e6e6e6", lw=0.8)
ax[1].plot(avg.index, avg["share_open"], color=ORANGE, lw=2)
ax[1].set_ylim(0, 1)
ax[1].yaxis.set_major_formatter(PercentFormatter(1.0))
ax[1].set_ylabel("Open share\nof paid tokens")
ax[1].grid(axis="y", color="#e6e6e6", lw=0.8)
fig.text(0.01, 0.01, "Source: OpenRouter rankings and list prices. Paid tokens only, top 50 models "
         "per day. 4-week averages.\nA rising share with both lines rising is differential growth, "
         "not switching.", fontsize=7.5, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{DATA}/fig_levels.png", dpi=220)
print(f"\nsaved {DATA}/fig_levels.png")
