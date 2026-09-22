"""
Idea 5 - Step 6: event study around frontier changes.

The frontier is a step variable: it changes in about 26 of 91 weeks. A weekly
regression is the wrong tool for that. This script asks a simpler question:

    After the best open model improves, does the open share grow faster than usual?
    And what happens after the best closed model improves?

Needs: data/weekly_quality.csv (run quality_measures.py first)
Usage: python event_study.py

Output (in ./data):
    event_windows.csv     one row per week: events and later changes
    event_study.txt       counts, means and regressions
    fig_event_study.png   average path around each event type
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

DATA = "data"
HORIZONS = (4, 8)          # weeks after the event
PRE, POST = 4, 8           # window for the figure
FONT = "IBM Plex Sans"
EPS = 1e-9

wk = pd.read_csv(f"{DATA}/weekly_quality.csv", index_col=0, parse_dates=True).sort_index()
wk["frontier_rel"] = np.log(wk["front_closed"] / wk["front_open"])
wk["y"] = np.log(wk["open_share_paid"] / (1 - wk["open_share_paid"]))   # log odds

wk["open_event"] = (wk["front_open"].diff() > EPS).astype(int)
wk["closed_event"] = (wk["front_closed"].diff() > EPS).astype(int)
wk["no_event"] = ((wk["open_event"] == 0) & (wk["closed_event"] == 0)).astype(int)
both = ((wk["open_event"] == 1) & (wk["closed_event"] == 1)).sum()
print(f"weeks: {len(wk)} | open-frontier events: {wk['open_event'].sum()} | "
      f"closed-frontier events: {wk['closed_event'].sum()} | both in one week: {both} | "
      f"no event: {wk['no_event'].sum()}")

for h in HORIZONS:
    wk[f"d{h}"] = wk["y"].shift(-h) - wk["y"]      # change over the next h weeks
wk.to_csv(f"{DATA}/event_windows.csv")

lines = []


def ols_hac(y, X, lags):
    y, X = np.asarray(y, float), np.asarray(X, float)
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
    return b, np.sqrt(np.diag(V))


for h in HORIZONS:
    d = wk.dropna(subset=[f"d{h}"])
    lines.append(f"\n===== change in log odds over the next {h} weeks (n={len(d)}) =====")
    for name, mask in [("after an open-frontier event", d["open_event"] == 1),
                       ("after a closed-frontier event", d["closed_event"] == 1),
                       ("weeks with no event", d["no_event"] == 1)]:
        v = d.loc[mask, f"d{h}"]
        lines.append(f"{name:<32} n={len(v):>3}  mean={v.mean():+.3f}  median={v.median():+.3f}")

    # Regression: the windows overlap, so use HAC with h lags
    X = np.column_stack([np.ones(len(d)), d["open_event"], d["closed_event"]])
    b, se = ols_hac(d[f"d{h}"], X, lags=h)
    lines.append(f"\n{'variable':<22}{'coef':>10}{'HAC se':>10}{'t':>8}")
    for nm, bi, si in zip(["const (no event)", "open_event", "closed_event"], b, se):
        lines.append(f"{nm:<22}{bi:>10.3f}{si:>10.3f}{bi / si:>8.2f}")

text = "\n".join(lines) + (
    "\n\nHow to read it: 'const' is the usual growth of the open share in a week with no "
    "frontier change.\n'open_event' is the extra growth over the next weeks when the best open "
    "model improves.\nThe windows overlap and the events are few, so treat this as indicative."
    "\nA log-odds change of 0.10 is about a 10% change in the open/closed token ratio.")
open(f"{DATA}/event_study.txt", "w").write(text)
print(text)

# ------------------------------------------------------------ figure
names = {f.name for f in font_manager.fontManager.ttflist}
if FONT in names:
    plt.rcParams["font.family"] = FONT
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def path(mask):
    rows = []
    idx = np.where(mask.to_numpy())[0]
    y = wk["y"].to_numpy()
    for i in idx:
        if i - PRE < 0 or i + POST >= len(y):
            continue
        rows.append(y[i - PRE:i + POST + 1] - y[i])     # set the event week to 0
    return np.array(rows)


fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(-PRE, POST + 1)
for mask, color, label in [(wk["open_event"] == 1, "#e07b24", "Best open model improves"),
                           (wk["closed_event"] == 1, "#1f5fa8", "Best closed model improves"),
                           (wk["no_event"] == 1, "#8a8a8a", "No frontier change")]:
    p = path(mask)
    if len(p) == 0:
        continue
    m = p.mean(axis=0)
    ax.plot(x, m, color=color, lw=2, label=f"{label} (n={len(p)})")
    if len(p) > 1:
        se = p.std(axis=0, ddof=1) / np.sqrt(len(p))
        ax.fill_between(x, m - se, m + se, color=color, alpha=0.15)
ax.axvline(0, color="#cccccc", lw=1)
ax.axhline(0, color="#cccccc", lw=0.8)
ax.set_xlabel("Weeks from the event")
ax.set_ylabel("Change in log odds of the open share")
ax.set_title("Open-weight share around frontier changes", loc="left", fontsize=14, pad=12)
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", color="#e6e6e6", lw=0.8)
fig.text(0.01, 0.01, "Sources: Artificial Analysis and OpenRouter. Bands: standard error of the mean. "
         "Paid tokens only.", fontsize=7.5, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{DATA}/fig_event_study.png", dpi=220)
print(f"\nsaved {DATA}/fig_event_study.png")
