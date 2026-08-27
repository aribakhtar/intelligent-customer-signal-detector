"""Adapt a long-format monthly snapshot CSV into the detector's two frames.

Usage:
    python adapt.py customer_signal_detector_test_data.csv

Expected input: one row per customer per month, with at least
`customer_id`, `month`, and whatever of {company, segment, monthly_usage_pct,
support_tickets, csat_score, monthly_revenue_usd, billing_status,
customer_message} the feed carries. Anything absent is simply not emitted -
signals.py skips missing signals and renormalises rather than assuming zero.

Conversions that matter:
  - csat_score is assumed 1-10 here and is halved to the detector's 1-5 scale.
  - "last 30d vs prior 30d" maps to the latest month vs the month before it.
  - monthly_revenue_usd is annualised (x12) into arr_usd.
  - billing_status counts across the window become failed_payments_90d and
    days_payment_overdue (30 days per month observed in that state).
  - Columns the detector wants but this feed lacks (seats, NPS, renewal date,
    reopened tickets, P1s) are left out entirely, not defaulted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Label columns that must never reach the model - they are ground truth.
GROUND_TRUTH = ["ground_truth_risk", "scenario_type", "scenario_description",
                "sentiment", "churn_intent"]


BASELINE_MONTHS = 3


def adapt(src: pd.DataFrame, baseline_months: int = BASELINE_MONTHS
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = src.copy()
    src["month"] = pd.to_datetime(src["month"])
    src = src.sort_values(["customer_id", "month"])

    customers, interactions = [], []
    for cid, g in src.groupby("customer_id", sort=False):
        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None
        rec: dict = {"customer_id": cid}

        if "company" in g:
            rec["account_name"] = last.company
        if "segment" in g:
            rec["segment"] = last.segment
        if "monthly_revenue_usd" in g:
            rec["arr_usd"] = float(last.monthly_revenue_usd) * 12

        # Baseline is a FIXED LOOKBACK, not the immediately preceding month and
        # not the first row. Month-over-month reads a sustained slide as three
        # mild steps; the first observed row is worse still on a long panel -
        # on 18 months of history it compares June against the previous January,
        # where ordinary drift and seasonality swamp the recent signal. Falls
        # back to the earliest row when history is shorter than the lookback.
        cutoff = last.month - pd.DateOffset(months=baseline_months)
        earlier = g[g.month <= cutoff]
        base = earlier.iloc[-1] if len(earlier) else (g.iloc[0] if len(g) > 2 else prev)
        if "monthly_usage_pct" in g and base is not None:
            rec["logins_last_30d"] = float(last.monthly_usage_pct)
            rec["logins_prev_30d"] = float(base.monthly_usage_pct)
        if "support_tickets" in g and base is not None:
            rec["tickets_last_30d"] = float(last.support_tickets)
            rec["tickets_prev_30d"] = float(base.support_tickets)
        if "csat_score" in g:
            rec["csat_current"] = round(float(last.csat_score) / 2, 1)   # 1-10 -> 1-5
            prior = base if base is not None else g.iloc[0]
            rec["csat_prev_quarter"] = round(float(prior.csat_score) / 2, 1)

        if "billing_status" in g:
            status = g.billing_status.astype(str)
            rec["failed_payments_90d"] = int((status == "payment_failed").sum())
            # Each month observed in a bad state counts as ~30 days of exposure.
            rec["days_payment_overdue"] = int(30 * (status != "paid_on_time").sum())

        customers.append(rec)

        if "customer_message" in g:
            for row in g.itertuples():
                msg = getattr(row, "customer_message", None)
                if isinstance(msg, str) and msg.strip():
                    interactions.append({
                        "customer_id": cid,
                        "date": row.month.date().isoformat(),
                        "channel": "support",
                        "text": msg.strip(),
                    })

    cust = pd.DataFrame(customers)
    # Carry ground-truth labels through for evaluation only. detect() never sees
    # them: it reads named signal columns, and evaluate() is given them separately.
    for col in GROUND_TRUTH:
        if col in src.columns:
            cust[f"_{col}"] = cust.customer_id.map(
                src.groupby("customer_id")[col].last())
    return cust, pd.DataFrame(interactions)


def main(path: str) -> None:
    src = pd.read_csv(path)
    cust, inter = adapt(src)
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    cust.to_csv(out / "customers.csv", index=False)
    inter.to_csv(out / "interactions.csv", index=False)
    have = [c for c in cust.columns if not c.startswith("_")]
    print(f"{len(cust)} customers, {len(inter)} interactions -> {out}")
    print(f"columns carried: {', '.join(have)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {Path(__file__).name} <long-format.csv>")
    main(sys.argv[1])
