# Intelligent Customer Signal Detector

Drop a folder of CSVs in one end — CRM export, usage snapshots, support chat, ticket export,
billing extract — and get out a ranked worklist of customers who need attention, each with a risk
score, the evidence behind it, and a recommended retention action. Built so a customer-operations
team can intervene *before* the cancellation email arrives.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` in the project root (gitignored, read via `python-dotenv`):

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini      # optional, this is the default
```

Then either:

```bash
streamlit run app.py                    # the demo UI - upload CSVs, watch the layers run
python pipeline.py sample_inbox/        # same engine, batch - writes data/nodes.json (~13s)
python pipeline.py data_extensive/ --as-of 2025-06-30 -o data/q2.json
python pipeline.py sample_inbox/ --watch 60      # keep polling the folder
```

The UI has three tabs: **Worklist** (ranked table, per-account briefing card, CSV/JSON export),
**Buckets & signals** (bucket definitions with SLAs, plus a signal-severity heatmap), and
**Account detail** (score breakdown with evidence). Sidebar takes uploads, a bundled folder, or a
previously saved run; the *Max customers* slider caps how many reach the model, one call each.

## Approach

Four layers. The UI shows each one completing, so the working is visible rather than implied.

**L1 · Document processing.** Identifies each CSV on its own. Known column sets are recognised
outright, with no model call. Unfamiliar headers go to the LLM, which returns a column → canonical
field mapping plus a note explaining itself — that is how `HealthSurveyOutOfTen` becomes
`csat_current` *and* gets halved onto the 1–5 scale. Any file carrying outcome or label columns
(`outcome`, `churn`, `ground_truth`, `arr_after`, `is_censored`, …) is **quarantined before
scoring**, structurally, without asking the model's opinion. A drop folder eventually receives the
answers by accident; training a worklist on those produces a beautiful, meaningless result.

**L2 · Consolidation.** Merges accepted files into one profile per customer plus a pooled list of
*customer-authored* messages (agent replies excluded). Precedence is per-field, not per-file, so a
finance extract can win on `days_payment_overdue` while the CRM wins on `arr_usd`. Fields absent
everywhere stay absent — never zero-filled.

**L3 · Sentiment & scoring.** Two halves, deliberately separated:

| Deterministic (pandas, no model) | w | LLM-scored (one call per customer) | w |
|---|---|---|---|
| `usage_decline` — 30d logins vs prior 30d | .16 | `churn_language` — explicit exit intent | .20 |
| `support_strain` — volume spike, reopens, open P1s | .15 | `text_sentiment` — how they sound, in context | .16 |
| `billing_friction` — failed payments, overdue, downgrade | .13 | | |
| `satisfaction_drop` — CSAT trend + level, NPS | .12 | | |
| `engagement_breadth` — seat use + feature adoption | .08 | | |

Each signal normalises to 0–1 and carries a human-readable evidence string.
`risk = 100 × Σ(weight × severity)`, then a renewal amplifier (×1.25 inside 45 days, ×1.12 inside
90). Signals with no data are dropped and their **weight redistributed**, so scores stay comparable
across feeds with different coverage. The same call returns the issue title, rationale, recommended
action and owner.

**L4 · Bucketing.** Ranks the book and cuts it by *share*, not fixed thresholds — top 5% *Needs
attention now*, next 10% *At risk*, next 15% *Worth watching*, rest *Likely to stay* — each with an
SLA. Fixed cut-points tuned on one population leave the worst buckets empty on a calmer one; shares
keep the worklist workable whatever arrives. L4 also diffs against the previous run and flags who
escalated.

**Why split it this way.** The ranking is arithmetic over evidence the code extracted, so it is
reproducible and auditable. The model is used where it genuinely beats rules — reading intent out of
prose, and writing the "why" a CSM reads before a call. There is **no keyword sentiment path**: a
missing key is a hard stop, and a failed call is reported as `TEXT NOT ANALYSED` rather than
back-filled with a guess.

## Example — input to output

**Input**, three files in `sample_inbox/`. `C001` appears in all three:

```
crm_accounts.csv      customer_id=C001, account_name=Acme Retail, segment=Enterprise, arr_usd=68472,
                      logins_prev_30d=95, logins_last_30d=69, tickets_prev_30d=1, tickets_last_30d=9,
                      csat_prev_quarter=3.5, csat_current=1.5, days_payment_overdue=30

finance_extract.csv   AcctRef=C001, ClientName=Acme Retail, HealthSurveyOutOfTen=3.0,
                      BillingArrearsDays=30            <- headers the detector has never seen

support_chat.csv      2026-07-01 support  "The workflow is acceptable, but there are a few issues."
                      2026-07-31 support  "If these issues continue, we may need to evaluate alternatives."
                      2026-08-30 support  "We are seriously considering cancelling and moving to another provider."
```

**Output**, one entry of the ranked worklist (`data/nodes.json`, rendered in the UI as a briefing card):

```
Acme Retail  ·  C001  ·  Enterprise  ·  $68,472 ARR
URGENCY 75        bucket: Needs attention now

What drove the score
  churn_language   0.90  -> 18.0 pts   Explicit exit language in recent messages
  text_sentiment   0.90  -> 14.4 pts   Latest: "We are seriously considering cancelling and
                                        moving to another provider."
  support_strain   0.89  -> 13.3 pts   Tickets 1 -> 9

Rationale   Acme Retail has shown a significant decline in usage from 95 to 69 and an alarming
            drop in CSAT from 3.5 to 1.5. Their recent support interactions indicate they are
            seriously considering cancellation and moving to another provider.

Next step   CSM - Schedule a call to discuss their concerns and potential solutions.
Built from  crm_accounts.csv, finance_extract.csv, support_chat.csv
```

That run scores 30 customers from 3 files and 90 messages in ~13 seconds:
`urgent 1 · high 3 · watch 5 · stable 21`.

## Results

Point-in-time backtest on the bundled 700-account, 24-month panel. Three quarterly prediction dates;
features built only from records dated ≤ T; graded against what actually happened in the next 90
days (`python validation/check_pooled.py`).

```
695 account-months, 84 churn events (12.1% base rate), 3 prediction dates

Ranking quality      pooled ROC AUC 0.852   95% CI [0.807, 0.893]
Worklist precision   top 10 per cycle 66.7% (5.5x base)   top 20 65.0%   top 50 40.0%
Bucket performance   urgent 69.6% churn  ·  high 41.8%  ·  watch 12.9%  ·  stable 3.2%
Lead time            median 43 days before the customer decides (p25 24, p75 60)

Lift over simpler baselines (AUC)
  CSAT alone 0.655 · usage decline alone 0.684 · LLM text only 0.685
  deterministic rules only 0.775 · full detector 0.851
```

Signal fusion is worth **+0.076 AUC** over the best single component and +0.196 over CSAT alone. The
LLM layer graded on its own against 300 human-labelled messages (`validation/eval_llm.py`): churn
intent precision 100% / recall 80.9%; sentiment AUC 0.965.

## Assumptions

- **Synthetic data only.** `data_extensive/` is a generated 700-account, 24-month panel with real
  outcomes; `sample_inbox/` is a small three-file demo. No client or proprietary data.
- **Batch, not streaming.** Scores a snapshot on demand. Production would run nightly and alert on
  *bucket transitions* rather than absolute scores.
- **Weights are hand-set, not learned.** With real churn outcomes they should be fitted (logistic
  regression over the same normalised signals); the extraction layer would not change.
- **Bucket shares are a policy dial** matched to team capacity, not a fitted parameter.
- **Renewal proximity amplifies urgency, it is not risk itself** — the same decay matters more with
  30 days left than 300. It only looks forward; past-dated contracts never amplify.
- **Missing data is missing, not zero.** Absent signals are dropped with their weight redistributed,
  and are left out of the model prompt rather than padded with defaults it would reason over as fact.
- **English, single-tenant, no PII redaction.** Production would strip PII before the model call.

## Tools

Python 3.10 · pandas · Streamlit · OpenAI Python SDK (`gpt-4o-mini`, strict JSON-schema structured
output) · python-dotenv · matplotlib (heatmap gradient only). The provider is isolated to `_client()`
and `_analyse_text()` in `signals.py`; everything else is provider-agnostic.

## Files

| Path | What it is |
|---|---|
| `pipeline.py` | The four layers + CLI |
| `signals.py` | Detection core — signal extraction, LLM layer, scoring |
| `app.py` | Streamlit UI |
| `adapt.py` | Folds a long-format monthly panel into current-vs-baseline deltas (used by L2) |
| `sample_inbox/` | 3-file demo input — CRM, finance extract, support chat |
| `data_extensive/` | 700 accounts × 24 months, with outcomes and labels for grading |
| `validation/` | Backtest and graders that produce the Results numbers |
| `data/` | Generated output only (gitignored) — every file reproducible by re-running |
