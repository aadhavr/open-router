"""
Idea 5 - Open-model demand, adjusted for price.
Step 1: collect OpenRouter usage history and a daily price snapshot.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python collect_openrouter.py            # full backfill + today's price snapshot
    python collect_openrouter.py --prices   # price snapshot only (run daily with cron)

Output (in ./data):
    rankings_daily.csv         date, model_permaslug, total_tokens
    prices/models_YYYY-MM-DD.csv   one price snapshot per day
    panel_latest.csv           usage merged with the newest price snapshot
Cite: "Source: OpenRouter (openrouter.ai/rankings), as of {as_of}." (CC BY 4.0)
"""
import os, sys, time, datetime as dt
import requests
import pandas as pd

BASE = "https://openrouter.ai/api/v1"
KEY = os.environ.get("OPENROUTER_API_KEY")
DATA = "data"
START = dt.date(2025, 1, 1)          # the dataset begins here
CHUNK_DAYS = 30                      # request 30 days at a time

os.makedirs(f"{DATA}/prices", exist_ok=True)


def get(path, params=None, auth=True):
    headers = {"Authorization": f"Bearer {KEY}"} if auth else {}
    for attempt in range(5):
        r = requests.get(f"{BASE}{path}", params=params, headers=headers, timeout=60)
        if r.status_code == 429:     # rate limit: 30/min, 500/day
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed after retries: {path} {params}")


def backfill_rankings():
    end = dt.date.today() - dt.timedelta(days=1)
    rows, as_of = [], None
    start = START
    while start <= end:
        stop = min(start + dt.timedelta(days=CHUNK_DAYS - 1), end)
        js = get("/datasets/rankings-daily",
                 {"start_date": start.isoformat(), "end_date": stop.isoformat(),
                  "period": "day"})
        rows += js["data"]
        as_of = js["meta"]["as_of"]
        print(f"rankings {start} to {stop}: {len(js['data'])} rows")
        start = stop + dt.timedelta(days=1)
        time.sleep(2.5)              # stay under 30 requests per minute
    df = pd.DataFrame(rows)
    df["total_tokens"] = df["total_tokens"].astype("int64")
    df["as_of"] = as_of
    df.to_csv(f"{DATA}/rankings_daily.csv", index=False)
    return df


def snapshot_prices():
    js = get("/models", auth=False)
    recs = []
    for m in js["data"]:
        p = m.get("pricing", {}) or {}
        recs.append({
            "snapshot_date": dt.date.today().isoformat(),
            "id": m.get("id"),
            "canonical_slug": m.get("canonical_slug"),
            "name": m.get("name"),
            "created": m.get("created"),
            "hugging_face_id": m.get("hugging_face_id") or "",
            "context_length": m.get("context_length"),
            "price_prompt_usd_per_token": float(p.get("prompt") or 0),
            "price_completion_usd_per_token": float(p.get("completion") or 0),
        })
    df = pd.DataFrame(recs)
    # First-pass open-weights flag: the model links to Hugging Face weights.
    # Review this flag by hand before you publish.
    df["open_weights"] = df["hugging_face_id"].str.len() > 0
    # Blended price per million tokens, 3 input : 1 output
    df["blended_usd_per_mtok"] = 1e6 * (0.75 * df["price_prompt_usd_per_token"]
                                        + 0.25 * df["price_completion_usd_per_token"])
    df.to_csv(f"{DATA}/prices/models_{dt.date.today().isoformat()}.csv", index=False)
    print(f"price snapshot: {len(df)} models")
    return df


def build_panel(rank, prices):
    rank = rank[rank["model_permaslug"] != "other"].copy()
    rank["slug"] = rank["model_permaslug"].str.split(":").str[0]   # drop ':free' etc.
    keys = prices[["canonical_slug", "open_weights", "blended_usd_per_mtok"]]
    panel = rank.merge(keys, left_on="slug", right_on="canonical_slug", how="left")
    unmatched = panel.loc[panel["canonical_slug"].isna(), "slug"].unique()
    pd.Series(unmatched, name="slug").to_csv(f"{DATA}/unmatched_slugs.csv", index=False)
    print(f"unmatched slugs (classify by hand): {len(unmatched)}")
    panel.to_csv(f"{DATA}/panel_latest.csv", index=False)

    daily = (panel.dropna(subset=["open_weights"])
                  .groupby(["date", "open_weights"])["total_tokens"].sum()
                  .unstack(fill_value=0))
    daily["open_share"] = daily.get(True, 0) / daily.sum(axis=1)
    daily.to_csv(f"{DATA}/open_share_daily.csv")
    print(daily["open_share"].tail())


if __name__ == "__main__":
    prices = snapshot_prices()
    if "--prices" not in sys.argv:
        if not KEY:
            sys.exit("Set OPENROUTER_API_KEY first.")
        build_panel(backfill_rankings(), prices)
