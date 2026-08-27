# Intelligent Customer Signal Detector

Correlates support text, product usage, billing and satisfaction data into a single
prioritised watchlist of at-risk customers — each with a risk score, the signals that
drove it, and a recommended retention action. Built so a customer-operations team can
intervene *before* the cancellation email arrives.

## Approach

Two layers, deliberately separated:

**1. Deterministic signal extraction** (pandas, no model). Five structured signals, each
normalised to `0.0–1.0` and each carrying a human-readable evidence string:

| Signal | Built from | Weight |
|---|---|---|
| `usage_decline` | 30d logins vs prior 30d | 0.16 |
| `support_strain` | ticket volume spike, reopened tickets, open P1s, first-response time | 0.15 |
| `billing_friction` | failed payments, invoice disputes, days overdue, recent downgrade | 0.13 |
| `satisfaction_drop` | CSAT trend + CSAT level + NPS level | 0.12 |
| `engagement_breadth` | seat utilisation + feature adoption | 0.08 |

**2. An LLM layer** (OpenAI `gpt-4o-mini` by default, one call per account, strict JSON-schema
structured output). **All sentiment and churn-intent scoring is done by the model — there is
no keyword, lexicon or heuristic path.** It reads the account's actual chat/email/survey text alongside the structured signals and
returns two more scores plus the narrative:

| Signal | Weight |
|---|---|
| `churn_language` — explicit exit intent (cancellation, notice period, data export, named competitor) | 0.20 |
| `text_sentiment` — how the customer sounds, judged in context | 0.16 |

…and the `themes`, `rationale`, `recommended_action` and `action_owner` shown in the UI.

Without a usable key the pipeline raises `NoLLMError` and stops. If an individual account's
call fails, that account is reported as `TEXT NOT ANALYSED` with an incomplete score — a
sentiment number is never guessed at or back-filled from rules.

**Banding.** Absolute cut points (Critical ≥70, High ≥50, Watch ≥30) suit the demo data,
where accounts decay dramatically. On the realistic 700-account book every raw severity
averaged 0.02–0.18, the weighted score topped out at 43, and the top two bands were empty
even though the ranking was sound. `apply_percentile_bands()` bands the top 3% / 8% / 15% of
the book instead — that turned a 13-account flag list (77% precision, 12% recall) into 103
accounts at 52% precision and 64% recall. Use it on real data; the sidebar has a toggle.

**Score.** `risk = 100 × Σ(weight × severity)`, then multiplied by a renewal-urgency
amplifier (×1.25 inside 45 days, ×1.12 inside 90). Bands: Critical ≥ 70, High ≥ 50,
Watch ≥ 30. `priority = risk × value_multiplier`, where the multiplier is bounded to
[1.0, 2.0] by log ARR — so a large account never outranks a genuinely burning small one by
more than 2×.

**Why split it this way.** The score is computed from evidence the code extracted, not from
a number the model invented, so the ranking is auditable and reproducible. The model is used
where it is genuinely better than rules — reading intent out of prose and writing the
"why" a CSM reads before a call.

## Results

### Point-in-time backtest against real outcomes (700 accounts, 24 months)

`python backtest.py` — three quarterly prediction dates, features built only from records
dated ≤ T, evaluated against what happened in the next 90 days.

```
695 account-months, 84 churn events (12.1% base rate), 3 prediction dates

Ranking quality        ROC AUC 0.851

Worklist precision - if a CSM works the top N accounts each cycle
  top 10 per cycle     precision 76.7%   lift 6.3x base rate
  top 20 per cycle     precision 61.7%   lift 5.1x base rate
  top 50 per cycle     precision 39.3%   lift 3.3x base rate

Band performance (rank-based)
  Critical  n  19   churn rate 73.7%   captures 16.7% of all churn
  High      n  36   churn rate 52.8%   captures 22.6%
  Watch     n  48   churn rate 43.8%   captures 25.0%
  flagged (top 15%)   precision 52.4%   recall 64.3%

Lift over simpler baselines (AUC)
  CSAT alone (inverted)     0.655
  usage decline alone       0.684
  LLM text signals only     0.685
  deterministic rules only  0.775
  full detector             0.851

Lead time on caught churn  median 42 days   p25 24   p75 60   (n 54)
```

This is the first result measured against **outcomes rather than labels**. Signal fusion is
worth +0.076 AUC over the best single component and +0.196 over CSAT alone, and the median
caught account is flagged 42 days before the customer decides.

Two traps in that dataset are handled explicitly in `backtest.py`: `outcome_within_90d`
counts renewals as positives (1176 of its 1571), so the churn label is rebuilt from
`outcome_type`; and `eligible_for_prediction` is true for censored rows, which are dropped
rather than scored as negatives.

### LLM layer, scored independently (300 human-labelled messages)

`python eval_llm.py`

```
CHURN INTENT   precision 100.0%  recall 80.9%  F1 89.4%
  AUC(churn_language vs human label)   1.000
SENTIMENT      accuracy 32.0%
  AUC(sentiment_risk vs human=negative) 0.965
```

Churn-intent detection is clean. The sentiment *accuracy* figure is a calibration artefact,
not a signal failure: the model's scores rank almost perfectly (AUC 0.965) but compress into
0.1–0.7, so the fixed 0.6/0.35 cut points put most negatives in the neutral bucket. The fix
is re-mapping the thresholds, not changing the model.


### On the provided labelled test set (30 accounts x 3 monthly snapshots)

`python adapt.py customer_signal_detector_test_data.csv && python signals.py`

```
Evaluation vs held-out `_ground_truth_risk` (at risk = critical or high)
  rank correlation  Spearman 0.956   Kendall 0.866
  band agreement    exact 57%   within one band 100%
  risk >=  30 (Watch   ) precision  87%  recall 100%  (tp 13, fp 2, fn 0, n 30)
  risk >=  50 (High    ) precision 100%  recall  62%  (tp  8, fp 0, fn 5, n 30)
  risk score by ground_truth_risk:
    critical    min 55  mean 66.4  max 75  (n 5)
    high        min 32  mean 45.0  max 54  (n 8)
    medium      min 19  mean 25.8  max 33  (n 8)
    low         min  0  mean  7.8  max 20  (n 9)
```

**No `low` account outranks any `high` or `critical` account** — the label bands separate
cleanly and monotonically. Recall is 100% at the Watch threshold; the two "false positives"
there are `medium` accounts, which is what the Watch band is for. The absolute scores run
lower than on the full-schema set because this feed carries 5 of the 7 signals (no seat,
adoption or NPS data) and no renewal dates, so the urgency amplifier never fires — the
weights renormalise over what is present rather than scoring absent data as clean.

### On the bundled synthetic set

The generator tags each synthetic account with an archetype (`healthy`, `quiet_decline`,
`billing_friction`, `support_strain`, `churn_intent`, `onboarding_stall`) that is held out
of everything the detector sees. `python signals.py` scores against it:

```
Model: gpt-4o-mini            (38 accounts, 0 failed calls)
  risk >=  30 (Watch   ) precision 100%  recall 100%  (tp 24, fp 0, fn 0, n 38)
  risk >=  50 (High    ) precision 100%  recall  21%  (tp  5, fp 0, fn 19, n 38)
  risk >=  70 (Critical) precision 100%  recall  17%  (tp  4, fp 0, fn 20, n 38)
  mean risk by archetype:
    churn_intent 86.8 | support_strain 45.8 | onboarding_stall 45.0
    quiet_decline 40.7 | billing_friction 36.8 | healthy 1.3
```

Zero false positives at every threshold. Every non-healthy account clears the Watch band;
the Critical band is reserved for the four accounts with explicit exit language. The quiet
decliners — falling usage, no complaints filed — average 40.7 against healthy at 1.3. That
gap is the point of the prototype.

## Running it

```bash
pip install -r requirements.txt

# Required. Edit .env in the project root:
#   OPENAI_API_KEY=sk-...
#   OPENAI_MODEL=gpt-4o-mini     (optional, this is the default)

python make_data.py                    # regenerate the synthetic dataset (optional)
python signals.py                      # CLI: ranked list + evaluation + CSV export
python test_signals.py                 # self-checks (no key or network needed)
streamlit run app.py                   # the demo UI
```

`.env` is read automatically by `signals.py` via `python-dotenv`, and is gitignored.

The UI has three tabs: **Prioritised watchlist** (ranked table + a "today's briefing" card
per top account, CSV export), **Signal heatmap** (per-account severity across all seven
signals, so you can see which lever to pull), and **Account detail** (score breakdown in
points-of-risk, plus the source interactions the assessment was made from).

Bring your own data by uploading `customers.csv` / `interactions.csv` in the sidebar — see
`data/` for the expected columns.

### Bringing your own data

Two shapes are supported.

**Wide** (one row per customer) — drop it in as `data/customers.csv` plus a
`data/interactions.csv` of `customer_id, date, channel, text`. Every column is optional
except `customer_id`: signals whose source columns are absent are **skipped and their weight
redistributed**, never scored as zero. CSAT is expected on a 1–5 scale.

**Long** (one row per customer per period) — convert it first:

```bash
python adapt.py customer_signal_detector_test_data.csv
```

`adapt.py` folds the monthly snapshots into current-vs-baseline deltas, annualises revenue,
halves a 1–10 CSAT to the 1–5 scale, derives billing signals from `billing_status`, and
turns each `customer_message` into an interaction. It uses the **first** observed period as
the usage/ticket baseline rather than the immediately preceding one, because a
month-over-month delta reads a sustained 95 → 81 → 69 slide as three mild steps and misses
the trajectory. Ground-truth label columns are carried through prefixed with `_` so
`evaluate()` can score against them; `detect()` never reads them.

## Worked example

**Input** — one row of `customers.csv`:

```
customer_id=C1019, account_name=Vertex Group, segment=Enterprise, arr_usd=280000,
renewal_date=2026-10-23, logins_prev_30d=1318, logins_last_30d=369,
seats_licensed=466, seats_active=195, feature_adoption_pct=0.41,
tickets_prev_30d=9, tickets_last_30d=26, tickets_reopened_30d=4, open_p1_tickets=1,
csat_prev_quarter=3.6, csat_current=1.5, nps_last=6,
failed_payments_90d=2, days_payment_overdue=30, downgraded_last_90d=0
```

…plus its rows in `interactions.csv`:

```
2026-07-15 email  We are evaluating alternatives ahead of the December renewal. Please send our data export options.
2026-08-09 email  Being straight with you - a competitor demoed something that solves this out of the box.
2026-08-18 chat   Leadership has asked what it would take to cancel. What is the notice period in our contract?
```

**Output** — one row of the signal summary:

```
Vertex Group · Enterprise · $280,000 ARR · renewal in 57 days
RISK 93  (Critical)   priority 174.3

Top drivers
  churn language   1.00  → 20.0 pts   Explicit exit language in recent messages
  usage decline    1.00  → 16.0 pts   Logins 1318 -> 369 (72% drop in 30d)
  text sentiment   1.00  → 16.0 pts   Latest: "Leadership has asked what it would take
                                       to cancel. What is the notice period in our contract?"

Rationale     Three separate exit signals in five weeks — a competitor evaluation, a data
              export request, and a direct question about contractual notice — against a
              72% collapse in logins and CSAT falling 3.6 → 1.5. This account has already
              started leaving; the renewal is 57 days out.

Next step     Account Exec to open a save conversation this week before the renewal
              window closes.
```

## Assumptions

- **Synthetic data.** No real customer data was used. `make_data.py` generates 38 accounts
  across six behavioural archetypes with realistic support text. The `_archetype` column is
  ground truth for evaluation only and is never shown to the detector.
- **Batch, not streaming.** The prototype scores a snapshot on demand. Production would run
  this nightly and alert on band transitions rather than absolute scores.
- **Weights are hand-set, not learned.** With real churn outcomes they should be fitted
  (logistic regression on the same normalised signals) — the extraction layer would not change.
- **Renewal proximity is an urgency amplifier, not a risk signal.** Same decay matters more
  with 30 days left than 300.
- **No non-LLM path.** Sentiment and churn intent are always model-scored. A missing key is a
  hard stop, and a failed call is reported as an unanalysed account rather than filled in with
  a heuristic guess — a keyword matcher is not sentiment analysis and should not be presented
  as one. The self-checks inject a stub client so the maths and wiring stay testable offline.
- **Single-tenant, single-language, English text.** No PII redaction layer; a production
  version would strip PII before the model call.
- **Missing data is missing, not zero.** A feed lacking a signal has that signal omitted and
  the remaining weights renormalised, so scores stay comparable across datasets with
  different coverage. Absent inputs are also left out of the LLM prompt rather than padded
  with defaults the model would reason over as facts.
- **Band thresholds (70/50/30) are a policy dial, not a fitted parameter.** They are held
  fixed across both datasets rather than tuned per dataset; that is why band agreement on the
  labelled set is 57% exact while ranking is 0.956. Aligning bands to a specific book is a
  one-line change once real churn outcomes exist.

## Files

| File | What it is |
|---|---|
| `signals.py` | Detection core — signal extraction, LLM layer, scoring, evaluation |
| `app.py` | Streamlit UI |
| `make_data.py` | Synthetic dataset generator (`python make_data.py [outdir]`) |
| `adapt.py` | Converts a long-format monthly CSV into the two input frames |
| `backtest.py` | Point-in-time backtest against real outcomes in `data_extensive/` |
| `eval_llm.py` | Scores the LLM layer against human-labelled messages |
| `test_signals.py` | Self-checks (`python test_signals.py`) |
| `data/` | `customers.csv`, `interactions.csv`, generated `signal_summary.csv` |
| `DATA_REQUEST.md` | What real data would be needed to validate predictive accuracy |
| `.env` | `OPENAI_API_KEY` and optional `OPENAI_MODEL` (gitignored) |

## Tools

Python 3.10 · pandas · Streamlit · OpenAI Python SDK (`gpt-4o-mini`, strict JSON-schema
structured output) · python-dotenv · matplotlib (heatmap gradient only).

The provider is isolated to `_client()` and `_analyse_text()` in `signals.py` — everything
else is provider-agnostic.
