"""
Build data/snapshot_list.json without the full CDX search.

1. Reads the snapshot timestamps from terminal_log.txt (the "skip" lines).
2. Sends a CDX search only for the period that is not in the log.
3. Saves the combined list for collect_wayback.py.

Usage:
    1. Save the terminal output as terminal_log.txt in this folder.
    2. python build_snapshot_list.py
    3. python collect_wayback.py
"""
import os, re, json, time
import requests

LOG = "terminal_log.txt"
OUT = "data/snapshot_list.json"
RAW = "data/wayback"
TARGET = "openrouter.ai/api/v1/models"
EXPECTED_TOTAL = 173          # from the first CDX search (15 + 79 + 79)

os.makedirs("data", exist_ok=True)

# 1. Snapshots from the log
with open(LOG) as f:
    text = f.read()
found = re.findall(r"/web/(\d{14})id_/(https://\S+?)\s", text)
snaps = {ts: url for ts, url in found}
if not snaps:
    raise SystemExit(f"No snapshots found in {LOG}. Check the file.")
last = max(snaps)
print(f"from log: {len(snaps)} snapshots, last one {last[:8]}")

# 2. CDX search only after the last date in the log
start = str(int(last[:8]) + 1)        # the next day (collapse keeps one per day)
params = {"url": TARGET, "output": "json", "filter": "statuscode:200",
          "collapse": "timestamp:8", "from": start}
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
        print(f"CDX: {e}, retry in {wait}s")
        time.sleep(wait)
if rows is None:
    raise SystemExit("CDX search failed. Wait 30 to 60 minutes and run again.")

new = 0
if rows:
    header = rows[0]
    for x in rows[1:]:
        d = dict(zip(header, x))
        if d["timestamp"] not in snaps:
            snaps[d["timestamp"]] = d["original"]
            new += 1
print(f"from CDX search (from {start}): {new} snapshots")

# 3. Check the count
saved = len([f for f in os.listdir(RAW) if f.endswith(".json")]) if os.path.isdir(RAW) else 0
total = len(snaps) + saved
print(f"in list: {len(snaps)} | already saved: {saved} | total: {total}")
if total < EXPECTED_TOTAL:
    print(f"WARNING: fewer than {EXPECTED_TOTAL}. Some snapshots are possibly missing.")
elif total > EXPECTED_TOTAL:
    print("Note: more than before. The archive possibly has new snapshots. This is OK.")

# 4. Save
with open(OUT, "w") as f:
    json.dump([{"timestamp": t, "original": u} for t, u in sorted(snaps.items())], f)
print(f"saved {OUT}")
