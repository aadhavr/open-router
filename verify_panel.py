"""
Checks for the output of collect_openrouter.py.
Run from the same folder:  python verify_panel.py
"""
import pandas as pd

rank = pd.read_csv("data/rankings_daily.csv", parse_dates=["date"])
panel = pd.read_csv("data/panel_latest.csv", parse_dates=["date"])


def title(t):
    print(f"\n=== {t} ===")


# 1. Missing dates and row counts
title("1. Dates")
full = pd.date_range(rank["date"].min(), rank["date"].max())
missing = full.difference(rank["date"].unique())
print(f"range: {full.min().date()} to {full.max().date()}, days: {len(full)}")
print(f"missing dates: {[d.date() for d in missing]}")
per_day = rank.groupby("date").size()
print(f"days with a row count other than 51: {(per_day != 51).sum()}")

# 2. Duplicates
title("2. Duplicates")
dups = rank.duplicated(["date", "model_permaslug"]).sum()
print(f"duplicate (date, model) rows: {dups}")

# 3. Coverage: how much traffic the top 50 covers
title("3. Top-50 coverage (top 50 / all tokens)")
tot = rank.groupby("date")["total_tokens"].sum()
oth = rank[rank["model_permaslug"] == "other"].groupby("date")["total_tokens"].sum()
cov = (1 - oth / tot).resample("MS").mean()
print(cov.round(3).to_string())

# 4. Total volume over time (look for sudden jumps)
title("4. Total tokens per day, monthly mean (trillions)")
print((tot.resample("MS").mean() / 1e12).round(2).to_string())
jumps = tot.pct_change().abs()
print("largest day-to-day changes:")
print(jumps.nlargest(5).round(2).to_string())

# 5. Match rate by tokens (not by count of slugs)
title("5. Match rate")
p = panel[panel["model_permaslug"] != "other"]
matched = p["open_weights"].notna()
print(f"slugs matched: {p.loc[matched, 'slug'].nunique()} of {p['slug'].nunique()}")
m = p.assign(m=matched).groupby([pd.Grouper(key="date", freq="MS"), "m"])["total_tokens"].sum().unstack(fill_value=0)
m["matched_share"] = m.get(True, 0) / m.sum(axis=1)
print(m["matched_share"].round(3).to_string())

# 6. Largest unmatched models (classify these first)
title("6. Top 20 unmatched models by tokens")
um = p[~matched].groupby("slug")["total_tokens"].sum().nlargest(20)
print((um / 1e12).round(2).to_string())

# 7. Spot check of the open-weights flag
title("7. Top 25 matched models: check the flag by hand")
top = (p[matched].groupby(["slug", "open_weights"])["total_tokens"].sum()
       .nlargest(25).reset_index())
top["tokens_T"] = (top["total_tokens"] / 1e12).round(2)
print(top[["slug", "open_weights", "tokens_T"]].to_string(index=False))

# 8. Free variants
title("8. Share of tokens from ':free' variants")
free = rank["model_permaslug"].str.endswith(":free")
print(f"{rank.loc[free, 'total_tokens'].sum() / rank['total_tokens'].sum():.1%}")
