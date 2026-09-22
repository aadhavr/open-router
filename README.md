# open-router

Code and data for **Open and Closed Models on OpenRouter Are Not Substitutes**
(<https://www.aadhavrajesh.com/posts/open-router/>).

A weekly panel of OpenRouter usage, list prices and benchmark scores, January 1,
2025 to September 21, 2026: 973 models priced, 121 labelled open or closed by
hand, 169 archived price snapshots.

## Finding

Open-weight models went from about a quarter of OpenRouter's paid tokens to about
three quarters. Closed-model volume never fell: it grew 115 times over the period
and peaked in the last month of the sample, while open-model volume grew 792
times. The price gap held at roughly five to one throughout, and no price or
quality measure predicts the share once you test the timing. What did change is
where the money goes: the share of router spend on models that any provider can
host went from 2% to 36%.

## Data

| Source | What it gives | Where |
|---|---|---|
| OpenRouter `rankings-daily` | Top 50 models by tokens for each day, plus an "other" row, 2025-01-01 to 2026-09-21 | `data/rankings_daily.csv` |
| Wayback Machine captures of `openrouter.ai/api/v1/models` | List prices and model metadata at 169 dates, 2024-03 to 2026-09 | `data/wayback/` |
| OpenRouter models endpoint, own capture | Prices from 2026-09-22 onward | `data/prices/` |
| Artificial Analysis API | Intelligence Index for 647 models, one pull, one index version | `data/aa_models.json` |
| Hand labelling | Open or closed, with a source for each model | `data/manual_labels.csv` |
| Hand matching | Each OpenRouter model to its Artificial Analysis variants | `data/aa_manual_matches.csv` |

Two days are missing at the source: 2025-06-15 and 2025-07-15. Four archived
captures returned content that was not valid JSON and were skipped; each has a
valid capture within a few days.

OpenRouter data is licensed CC BY 4.0. Artificial Analysis data requires
attribution to artificialanalysis.ai.

## Pipeline

Run in this order. Steps 1 and 2 download; the rest are local.

| Script | What it does |
|---|---|
| `collect_openrouter.py` | Downloads the daily rankings and one price snapshot |
| `build_snapshot_list.py` | Builds the list of Wayback captures (only needed once) |
| `collect_wayback.py` | Downloads the archived captures and builds the price history |
| `checks_and_figure.py` | Applies labels, runs the data checks, writes `panel_labeled.csv` |
| `apply_matches.py` | Fills `quality.csv` from the saved Artificial Analysis pull |
| `price_analysis.py` | Weekly prices, the price ratio, the share regressions |
| `quality_measures.py` | Frontier and top-50 quality gaps, absolute and relative |
| `event_study.py` | The open share around frontier changes |
| `levels_growth.py` | Volumes in levels: substitution or differential growth |
| `spend_split.py` | Tokens against spend, prices by creator, concentration |
| `make_figs.py` | The six interactive figures for the post |

Two keys are needed, both free:

```bash
export OPENROUTER_API_KEY=...   # rankings dataset
export AA_API_KEY=...           # Artificial Analysis
```

```bash
pip install -r requirements.txt
python collect_openrouter.py
python collect_wayback.py
python checks_and_figure.py
python apply_matches.py
python price_analysis.py
python quality_measures.py
python event_study.py
python levels_growth.py
python spend_split.py
python make_figs.py
```

Replication should use the saved Artificial Analysis pull in `data/aa_models.json`
rather than a fresh one, because scores change when the index version changes.

## Coding decisions

- **Open** means the weights are publicly downloadable on that date. License
  restrictions do not change the label, and weights released later do not count
  retroactively: MiniMax M3 is closed from 2026-05-31 to 2026-06-11 and open after.
- **Stealth** models (anonymous pre-release models) are excluded from the open and
  closed shares. A sensitivity line reassigns them to their revealed identities.
- **Excluded**: embedding and rerank models, and structured-output models that do
  not generate free text.
- **Free** covers any `:free` variant, plus any model with a list price of exactly
  zero.
- Prices are blended at three input tokens for every output token, because
  OpenRouter reports one token total for each model and day.
- Each usage row joins to prices on the full variant name, so a `:free` variant
  keeps its own zero price instead of inheriting the paid model's price.
- Captures before May 2025 carry no canonical model name, so older identifiers are
  mapped forward using later captures that carry both.
- Where a model has several effort or reasoning settings in the Artificial Analysis
  data, the highest-scoring variant is used.
- Two labels could not be confirmed: GLM-5-Turbo and GLM-5V-Turbo are labelled
  closed. Together they are 3.8 trillion tokens, under 1% of the period.

## Checks

The scripts print these, and the post reports them:

- Match rate between usage and prices: 97% to 100% in every month.
- Paid tokens with a usable price: 100%, of which 98.3% use a price from an
  earlier capture, median age two days.
- Top 50 coverage of all traffic: 90% to 96%.
- Quality-score coverage of paid tokens: below 60% in 18 of 91 weeks, and those
  weeks are dropped from the quality regressions.
- The open share computed in `price_analysis.py` matches the one in
  `checks_and_figure.py` in every month, which catches price-join errors.

## Sources

- OpenRouter rankings dataset, <https://openrouter.ai/rankings>, CC BY 4.0.
- Artificial Analysis Intelligence Index, <https://artificialanalysis.ai>.
- Internet Archive Wayback Machine, <https://web.archive.org>.
- Fradkin, A. (2025). Demand for large language models: evidence from an AI router.

## License

Code: MIT (see `LICENSE`). Data belongs to its original providers, and the
archived captures are redistributed here only as evidence for the price history.
