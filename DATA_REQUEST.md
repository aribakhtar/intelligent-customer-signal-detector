# Data Request — Customer Signal Detector

**For:** Data / RevOps
**Purpose:** everything needed to turn the prototype's ranking result into a defensible accuracy claim
**Effort:** one extract, then one backtest

---

## Why the current numbers aren't enough

On the labelled test set the detector reaches **0.96 rank correlation** with the assigned risk
levels, with no low-risk account outranking a high or critical one. That is a real result, but it
measures agreement with *a person's labels*, not with outcomes. Nobody in the test data actually
churned.

Two findings make the caveat concrete:

- **Ablation.** Deterministic rules alone reach 0.916; the language model alone 0.932; combined,
  0.958. Signal fusion helps — but most of the ranking is already recoverable without AI.
- **Templated text.** The test transcripts contain 15 unique message strings across 90
  interactions, so detecting "considering cancelling" is close to a string match.

Neither number survives contact with real support text.

**The one question this data answers:** if we score every account as it looked on a past date, do
the accounts we flagged actually leave — and how many days of warning did we give?

---

## 1. Outcomes, with dates — REQUIRED

Without this there is no target and nothing to measure. One row per account per contract period.

| Field | Notes |
|---|---|
| `account_id` | Join key, consistent across every source below |
| `outcome` | `renewed` / `churned_voluntary` / `churned_nonpayment` / `downgraded` / `active` |
| `decision_date` | When the customer *decided* — notice given, save call lost, competitor signed |
| `contract_end_date` | Separate field, not a substitute for the above |
| `arr_before`, `arr_after` | Catches contraction, which is churn in slow motion |
| `is_censored` | True where the renewal hasn't come up yet |

**Trap:** `decision_date` is the field people substitute away. The intervention window closes when
the customer makes up their mind — often 60–90 days before the contract lapses. Score against
`contract_end_date` and the model gets credit for "predicting" churn that had already happened.

**Trap:** censoring. An account whose renewal hasn't arrived is *unknown*, not a non-churner.
Counting it as a negative silently inflates precision.

---

## 2. History as snapshots, not current state — REQUIRED

This is the requirement that most often can't be met, so check it early. We need the data **as it
looked at each point in time** — monthly rows per account — not the current-state export a CRM
produces by default.

If a field like `csat_current` is overwritten in place, then today's value for a churned account
reflects the moment they quit. A model trained on that scores beautifully and is useless in
production, because at scoring time that number doesn't exist yet.

| Requirement | Target |
|---|---|
| Grain | One row per account per month |
| Span | 18–24 months |
| Why that span | Two annual renewal cycles: fit on the first, evaluate out-of-time on the second |
| Split | Temporal, not random — a random split puts the same account on both sides and leaks |

If only current-state data exists, say so now. There are still useful things to measure, but
predictive accuracy is not one of them, and we would reframe rather than produce a number we
can't defend.

---

## 3. Enough churn events to say anything — REQUIRED

Statistical power comes from **events, not rows**. A hundred thousand rows covering twelve churns
tells us nothing.

| Churn events | Precision estimate | What can be claimed |
|---|---|---|
| ~30 | ±18 pts | Nothing defensible |
| ~100 | ±10 pts | Practical floor — a headline figure |
| 200+ | ±7 pts | Confident; can segment by tier or region |

At 10% annual churn, 100 events means roughly **1,000 account-years** — about 700 accounts
observed over 18 months. Below that, we validate on ranking quality and say plainly that
predictive accuracy is out of reach on this book.

---

## 4. The intervention log — STRONGLY WANTED

Which accounts the CS team contacted, when, and what they did.

**The paradox it resolves:** the model flags an account, a CSM intervenes, the account is saved —
and it scores as a **false positive**. The model is punished for working. Any book with an active
retention team has this contaminating every number. With the log we can evaluate on the
un-intervened subset or treat outreach as a treatment effect. Without it, expect a systematic
underestimate we cannot correct for.

| Field | Notes |
|---|---|
| `account_id` | — |
| `contact_date` | — |
| `action_type` | Save call, discount, exec sponsor, roadmap commitment |
| `triggered_by` | Whether an existing health score or alert prompted it |

---

## 5. Real support text — STRONGLY WANTED

Realism matters far more than volume. A few thousand genuinely messy messages beat a hundred
thousand templated ones.

| Requirement | Why |
|---|---|
| Raw transcripts, not summaries | Summaries are written after the fact, often after the outcome is known |
| Per-message timestamps | To cut cleanly at the scoring date |
| Author role — customer vs agent | Agent text in the buffer pollutes the sentiment score |
| Cancellation conversation excluded | It is the outcome, not a predictor |

**Plus a small annotation set:** 200–300 messages with human sentiment and intent labels. Cheap to
produce, and it lets us score the language model on its own task independently of the churn
outcome — so when something is wrong we know which layer is wrong.

---

## 6. Leakage traps to check before sending — REQUIRED

Each of these lets information from after the decision leak backwards into the scoring window.
All produce suspiciously excellent results and a model that fails in production.

- Status fields that flip to `cancelled` and backfill historical rows
- Downgrades stamped with the effective date rather than the request date
- Billing marked `payment_failed` because the account was already being wound down
- Any field the CS team edits *in response to* risk — notes, health scores, priority flags
- Support tickets auto-created by the cancellation workflow
- Usage dropping to zero because access was revoked, not because engagement fell

---

## What each tier buys

| Tier | Gets us | Contents |
|---|---|---|
| **Minimum** | A real backtest | 18 months of monthly snapshots; ≥100 churn events with decision dates; raw text, timestamped and attributed |
| **Good** | A number we'd defend to a sceptic | Everything above + intervention log + a second year of history |
| **Ideal** | A publishable result | Everything above + a period before any retention tooling existed, as a clean control + annotated text set |

---

## What we'd report back

| Measure | The question it answers |
|---|---|
| Precision @ top-20 worklist | If a CSM works 20 accounts a week, how many are real? |
| Recall at each band | How much churn do we catch, and how much walks past us? |
| Median lead time | How many days of warning before the decision — the whole point |
| Lift over baseline | Does it beat sorting the book by CSAT descending? |
| Rules vs rules + LLM | What is the AI layer actually contributing? |

---

## Closing note

**The backtest should be allowed to falsify the premise, not just measure it.**

If the results come back showing the deterministic rules reach 0.90 and the language model adds
0.03, that is a genuine finding and a cheaper product. If lead time turns out to be six days, the
tool is a reporting layer rather than an early-warning system, and should be sold as one. Both
outcomes are worth more than a confident number nobody can stand behind — and either is reachable
from the extract described above.
