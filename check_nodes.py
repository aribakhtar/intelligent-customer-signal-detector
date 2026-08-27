"""Grade pipeline output against the outcome labels the pipeline never saw.

    python check_nodes.py data/nodes_extensive.json

The label guard in L1 quarantines outcomes.csv, so nothing in it reached the
scorer. That makes it a clean grading set here.

This is not the point-in-time backtest (see backtest.py) - it scores the book as
it stands using all available history, which is what a production run does. Read
it as "do the buckets separate the customers who left from the ones who stayed",
not as a forward-looking accuracy claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CHURN = {"churned_voluntary", "churned_nonpayment", "downgraded"}
SRC = Path(__file__).parent / "data_extensive" / "outcomes.csv"


LABELS = Path(__file__).parent / "data_extensive" / "prediction_labels_90d.csv"


def main(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes = pd.DataFrame(payload["nodes"])
    as_of = payload.get("as_of")

    if as_of:
        # A point-in-time run is graded on the 90-day forward label for that same
        # date - the identical basis backtest.py uses, so the two are comparable.
        lab = pd.read_csv(LABELS, parse_dates=["prediction_date"])
        lab = lab[(lab.prediction_date == pd.Timestamp(as_of))
                  & lab.eligible_for_prediction
                  & (lab.outcome_type != "censored")].copy()
        lab["churned"] = lab.outcome_type.isin(CHURN).astype(int)
        m = nodes.merge(lab[["account_id", "churned", "outcome_type"]],
                        left_on="id", right_on="account_id", how="inner")
        m = m.rename(columns={"outcome_type": "outcome"})
        print(f"POINT-IN-TIME run as of {as_of}, graded on the next 90 days")
    else:
        out = pd.read_csv(SRC)
        m = nodes.merge(out, left_on="id", right_on="account_id", how="inner")
        m = m[~m.is_censored.astype(bool)]      # unknown outcome is not a negative
        m["churned"] = m.outcome.isin(CHURN).astype(int)
        print("FULL-HISTORY run, graded on final outcomes. Not a forward-looking "
              "result:\nrecent months may post-date a customer's decision.")

    print(f"{len(m)} customers with a known outcome "
          f"({m.churned.sum()} churned, {m.churned.mean():.1%} base rate)\n")

    order = {b["key"]: b["rank"] for b in payload["buckets"]}
    labels = {b["key"]: b["label"] for b in payload["buckets"]}
    print(f"{'bucket':<10}{'n':>5}{'churned':>9}{'rate':>8}{'lift':>7}   captures")
    for key in sorted(order, key=order.get):
        g = m[m.bucket == key]
        if not len(g):
            continue
        rate = g.churned.mean()
        print(f"{labels[key]:<10}"[:10] + f"{len(g):>5}{int(g.churned.sum()):>9}"
              f"{rate:>7.1%}{rate / max(m.churned.mean(), 1e-9):>6.1f}x"
              f"   {g.churned.sum() / max(m.churned.sum(), 1):>5.1%} of all churn")

    flagged = m[m.bucket.isin(["urgent", "high"])]
    print(f"\nflagged (urgent + high): n {len(flagged)}  "
          f"precision {flagged.churned.mean():.1%}  "
          f"recall {flagged.churned.sum() / max(m.churned.sum(), 1):.1%}")

    d = pd.DataFrame({"s": m.urgency, "y": m.churned}).dropna()
    pos, neg = (d.y == 1).sum(), (d.y == 0).sum()
    r = d.s.rank()
    auc = (r[d.y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    print(f"ROC AUC (urgency vs churned): {auc:.3f}")

    print("\ntop 5 by urgency:")
    for _, n in m.nlargest(5, "urgency").iterrows():
        print(f"  {n.id}  {n.urgency:>3}  {n.bucket:<7} {n.outcome:<20} {n['issue'][:44]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else
         Path(__file__).parent / "data" / "nodes_extensive.json")
