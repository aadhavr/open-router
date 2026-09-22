"""
Idea 5 - Step 2: historical prices from the Wayback Machine.

Finds archived copies of https://openrouter.ai/api/v1/models (one per day),
downloads each one, and builds a price history. It also adds your own daily
snapshots from data/prices/. Then it rebuilds the panel: each usage row gets
the newest price that was known on or before that date.

Usage:
    python collect_wayback.py
Needs: data/rankings_daily.csv from collect_openrouter.py

Safe to stop and run again:
    - the snapshot list is saved in data/snapshot_list.json after the first search
    - files that are already downloaded are not downloaded again

Output (in ./data):
    wayback/models_YYYYMMDD.json   raw archived snapshots (kept as evidence)
    price_history.csv              one row per (snapshot_date, slug)
    panel_hist.csv                 usage + price known at that date
    unmatched_hist.csv             slugs still without a match (classify by hand)
    open_share_daily_hist.csv
"""
import os, json, time, glob
import requests
import pandas as pd

DATA = "data"
RAW = f"{DATA}/wayback"
CACHE = f"{DATA}/snapshot_list.json"
os.makedirs(RAW, exist_ok=True)
TARGET = "openrouter.ai/api/v1/models"
WAIT_BETWEEN = 6        # seconds between downloads (the archive blocks fast clients)


def list_snapshots():
    snaps = []
    for year in ["2024", "2025", "2026"]:
        params = {"url": TARGET, "output": "json",
                  "filter": "statuscode:200", "collapse": "timestamp:8",
                  "from": year, "to": year}
        rows = None
        for attempt in range(6):
            try:
                r = requests.get("https://web.archive.org/cdx/search/cdx",
                                 params=params, timeout=180)
                if r.status_code in (429, 503):
                    raise requests.HTTPError(r.status_code)
                r.raise_for_status()
                rows = r.json()
                break
            except Exception as e:
                wait = 30 * (attempt + 1)
                print(f"{year}: {e}, retry in {wait}s")
                time.sleep(wait)
        if rows is None:
            raise RuntimeError(f"CDX search failed for {year}. Wait and run again.")
        if rows:
            header = rows[0]
            snaps += [dict(zip(header, x)) for x in rows[1:]]
        print(f"{year}: {len(rows[1:]) if rows else 0} snapshots")
        time.sleep(5)
    print(f"archived snapshots found: {len(snaps)}")
    return snaps


def get_snapshot_list():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            snaps = json.load(f)
        print(f"snapshot list read from {CACHE}: {len(snaps)}")
        return snaps
    snaps = list_snapshots()
    with open(CACHE, "w") as f:
        json.dump(snaps, f)
    print(f"snapshot list saved to {CACHE}")
    return snaps


BAD_DIR = f"{DATA}/wayback_bad"          # captures that are not valid JSON
os.makedirs(BAD_DIR, exist_ok=True)


def download(snaps):
    def done(ts):
        d = ts[:8]
        return (os.path.exists(f"{RAW}/models_{d}.json")
                or os.path.exists(f"{BAD_DIR}/models_{d}.txt"))

    todo = [s for s in snaps if not done(s["timestamp"])]
    print(f"to download: {len(todo)} (already done: {len(snaps) - len(todo)})")
    failed, bad = [], []
    for i, s in enumerate(todo, 1):
        ts = s["timestamp"]
        d = ts[:8]
        url = f"https://web.archive.org/web/{ts}id_/{s['original']}"
        ok = False
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
            except Exception as e:
                # Network or server problem: wait and try again
                wait = 120 * (attempt + 1)
                print(f"[{i}/{len(todo)}] {d}: {type(e).__name__}, retry in {wait}s")
                time.sleep(wait)
                continue
            try:
                json.loads(r.text)
            except json.JSONDecodeError:
                # The archived copy itself is not JSON. A retry does not help.
                with open(f"{BAD_DIR}/models_{d}.txt", "w") as f:
                    f.write(r.text[:5000])
                print(f"[{i}/{len(todo)}] {d}: archived copy is not JSON, skip "
                      f"(first 5,000 characters saved in {BAD_DIR})")
                bad.append(d)
                ok = True
                break
            with open(f"{RAW}/models_{d}.json", "w") as f:
                f.write(r.text)
            print(f"[{i}/{len(todo)}] saved {d}")
            ok = True
            break
        if not ok:
            failed.append(d)
        time.sleep(WAIT_BETWEEN)
    if bad:
        print(f"not JSON (skipped for good): {bad}")
    if failed:
        print(f"failed after retries: {failed}. Run the script again later.")


def parse_all():
    recs = []

    def add(day, models):
        for m in models:
            p = m.get("pricing", {}) or {}
            try:
                pin, pout = float(p.get("prompt") or 0), float(p.get("completion") or 0)
            except (ValueError, TypeError):
                continue
            recs.append({"snapshot_date": day,
                         "slug": m.get("canonical_slug") or m.get("id"),
                         "id": m.get("id"),
                         "hugging_face_id": m.get("hugging_face_id") or "",
                         "blended_usd_per_mtok": 1e6 * (0.75 * pin + 0.25 * pout)})

    # 1. Wayback snapshots
    for fn in sorted(os.listdir(RAW)):
        day = pd.to_datetime(fn[7:15], format="%Y%m%d")
        try:
            with open(f"{RAW}/{fn}") as f:
                add(day, json.load(f).get("data", []))
        except Exception as e:
            print(f"cannot read {fn}: {e}")

    # 2. Your own daily snapshots from collect_openrouter.py
    for fn in sorted(glob.glob(f"{DATA}/prices/models_*.csv")):
        snap = pd.read_csv(fn).fillna({"hugging_face_id": ""})
        for _, row in snap.iterrows():
            recs.append({"snapshot_date": pd.to_datetime(row["snapshot_date"]),
                         "slug": row["canonical_slug"] if isinstance(row["canonical_slug"], str) else row["id"],
                         "id": row["id"],
                         "hugging_face_id": row["hugging_face_id"],
                         "blended_usd_per_mtok": row["blended_usd_per_mtok"]})

    hist = pd.DataFrame(recs)
    hist.to_csv(f"{DATA}/price_history.csv", index=False)
    print(f"price history: {hist['slug'].nunique()} models, "
          f"{hist['snapshot_date'].nunique()} dates")
    return hist


def build_panel(hist):
    rank = pd.read_csv(f"{DATA}/rankings_daily.csv")
    rank = rank[rank["model_permaslug"] != "other"].copy()
    rank["slug"] = rank["model_permaslug"].str.split(":").str[0]
    rank["is_free"] = rank["model_permaslug"].str.endswith(":free")
    rank["date"] = pd.to_datetime(rank["date"])

    # Match on slug; if that fails, try the old 'id' field
    cols = ["snapshot_date", "slug", "hugging_face_id", "blended_usd_per_mtok"]
    by_id = hist.drop(columns="slug").rename(columns={"id": "slug"})[cols]
    prices = pd.concat([hist[cols], by_id]).dropna(subset=["slug"])
    prices = prices.drop_duplicates(["snapshot_date", "slug"]).sort_values("snapshot_date")

    # Open flag: true if any snapshot links the model to Hugging Face weights
    flag = prices.groupby("slug")["hugging_face_id"].apply(
        lambda s: (s.astype(str).str.len() > 0).any())

    rank = rank.sort_values("date")
    # merge_asof needs the same data types on both sides
    rank["slug"] = rank["slug"].astype("string")
    rank["date"] = pd.to_datetime(rank["date"]).astype("datetime64[ns]")
    prices = prices.copy()
    prices["slug"] = prices["slug"].astype("string")
    prices["snapshot_date"] = pd.to_datetime(prices["snapshot_date"]).astype("datetime64[ns]")
    prices["blended_usd_per_mtok"] = pd.to_numeric(prices["blended_usd_per_mtok"], errors="coerce")
    prices = prices.sort_values("snapshot_date")
    flag.index = flag.index.astype("string")

    panel = pd.merge_asof(rank, prices[["snapshot_date", "slug", "blended_usd_per_mtok"]],
                          left_on="date", right_on="snapshot_date", by="slug",
                          direction="backward")
    panel["open_weights"] = panel["slug"].map(flag)
    panel["days_since_price"] = (panel["date"] - panel["snapshot_date"]).dt.days

    miss = panel[panel["open_weights"].isna()].groupby("slug")["total_tokens"].sum()
    miss.sort_values(ascending=False).to_csv(f"{DATA}/unmatched_hist.csv")
    share = miss.sum() / panel["total_tokens"].sum()
    print(f"unmatched slugs: {len(miss)}  ({share:.1%} of top-50 tokens)")

    m = (panel.assign(m=panel["open_weights"].notna())
              .groupby([pd.Grouper(key="date", freq="MS"), "m"])["total_tokens"].sum()
              .unstack(fill_value=0))
    m["matched_share"] = m.get(True, 0) / m.sum(axis=1)
    print("match rate by month:")
    print(m["matched_share"].round(3).to_string())

    panel.to_csv(f"{DATA}/panel_hist.csv", index=False)
    daily = (panel.dropna(subset=["open_weights"])
                  .groupby(["date", "open_weights"])["total_tokens"].sum()
                  .unstack(fill_value=0))
    daily["open_share"] = daily.get(True, 0) / daily.sum(axis=1)
    daily.to_csv(f"{DATA}/open_share_daily_hist.csv")
    print("open share by month (not final - classify unmatched models first):")
    print(daily["open_share"].resample("MS").mean().round(3).to_string())


if __name__ == "__main__":
    download(get_snapshot_list())
    build_panel(parse_all())
