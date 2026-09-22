"""
Main figure: open-weight share of OpenRouter tokens.

Needs: data/weekly_shares.csv and data/panel_labeled.csv (run checks_and_figure.py first)
Usage: python make_figure.py
Output: data/fig_open_share_v2.png

Font: IBM Plex Sans. Install it one time on a Mac:
    brew install --cask font-ibm-plex-sans
If the font is not found, the script tries to rebuild the matplotlib font cache,
then uses the default font and prints a warning.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

DATA = "data"
FONT = "IBM Plex Sans"

# Main model releases to mark. The date is the first day the model is in the top 50.
RELEASES = [
    ("deepseek/deepseek-r1", "DeepSeek R1"),
    ("deepseek/deepseek-chat-v3-0324", "DeepSeek V3-0324"),
    ("qwen/qwen3-coder-480b-a35b-07-25", "Qwen3 Coder"),
    ("openai/gpt-oss-120b", "gpt-oss-120b"),
    ("deepseek/deepseek-v3.2-20251201", "DeepSeek V3.2"),
    ("minimax/minimax-m2.5-20260211", "MiniMax M2.5"),
    ("deepseek/deepseek-v4-flash-20260423", "DeepSeek V4"),
    ("z-ai/glm-5.2-20260616", "GLM-5.2"),
    ("tencent/hy3-20260706", "Hy3"),
    ("z-ai/glm-5.3-flash-20260826", "GLM-5.3 Flash"),
]


def set_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    if FONT not in names:
        font_manager._load_fontmanager(try_read_cache=False)   # rebuild the cache
        names = {f.name for f in font_manager.fontManager.ttflist}
    if FONT in names:
        plt.rcParams["font.family"] = FONT
    else:
        print(f"WARNING: {FONT} not found. Install it (see top of file). Using default font.")


set_font()
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

w = pd.read_csv(f"{DATA}/weekly_shares.csv", index_col=0, parse_dates=True)
w = w[w.index >= "2025-01-01"]
avg = w.rolling(4, min_periods=4).mean()          # trailing 4-week average

panel = pd.read_csv(f"{DATA}/panel_labeled.csv", usecols=["date", "slug"], parse_dates=["date"])
first_day = panel.groupby("slug")["date"].min()
start = panel["date"].min()
# Skip a model that is already in the data on the first day (released before the data starts)
releases = [(first_day[s], name) for s, name in RELEASES
            if s in first_day.index and first_day[s] > start]
missing = [s for s, _ in RELEASES if s not in first_day.index]
if missing:
    print(f"not in data (no label drawn): {missing}")

fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1.2]})
BLUE, ORANGE, GRAY = "#1f5fa8", "#e07b24", "#8a8a8a"

a = ax[0]
a.plot(w.index, w["open_share_all"], color=BLUE, lw=0.8, alpha=0.35)
a.plot(avg.index, avg["open_share_all"], color=BLUE, lw=2.4, label="All tokens (4-week average)")
a.plot(w.index, w["open_share_no_free"], color=ORANGE, lw=0.8, alpha=0.35)
a.plot(avg.index, avg["open_share_no_free"], color=ORANGE, lw=2, ls="--",
       label="Without free variants (4-week average)")
a.plot(avg.index, avg["open_share_stealth_revealed"], color=GRAY, lw=1, ls=":",
       label="Stealth models as later revealed (4-week average)")
a.set_ylim(0, 1)
a.yaxis.set_major_formatter(PercentFormatter(1.0))
a.set_ylabel("Open-weight share of tokens")
a.set_title("Open-weight models' share of OpenRouter tokens", loc="left", fontsize=14, pad=12)
a.legend(frameon=False, loc="upper left", fontsize=9)
a.grid(axis="y", color="#e6e6e6", lw=0.8)

# Release labels: thin line and small rotated text, placed in the empty lower area
for i, (d, name) in enumerate(releases):
    a.axvline(d, color="#bdbdbd", lw=0.7, zorder=0)
    a.text(d, 0.03 + 0.09 * (i % 2), f" {name}", rotation=90, fontsize=7.5,
           color="#555555", va="bottom", ha="right", transform=a.get_xaxis_transform())

b = ax[1]
b.plot(avg.index, avg["free_share"], color=BLUE, lw=1.6, label="Free variants")
b.plot(avg.index, avg["stealth_share"], color=ORANGE, lw=1.6, label="Stealth models")
b.set_ylim(bottom=0)
b.yaxis.set_major_formatter(PercentFormatter(1.0))
b.set_ylabel("Share of all tokens\n(4-week average)")
b.legend(frameon=False, loc="upper left", fontsize=9)
b.grid(axis="y", color="#e6e6e6", lw=0.8)

as_of = pd.read_csv(f"{DATA}/rankings_daily.csv", nrows=1)["as_of"].iloc[0]
fig.text(0.01, 0.005,
         f"Source: OpenRouter (openrouter.ai/rankings), as of {as_of[:10]}. Top 50 models per day. "
         "Open share excludes stealth and unlabeled models. Thin lines: weekly values. "
         "Release labels: first day in the top 50.",
         fontsize=7.5, color="gray")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{DATA}/fig_open_share_v2.png", dpi=220)
print(f"saved {DATA}/fig_open_share_v2.png")
