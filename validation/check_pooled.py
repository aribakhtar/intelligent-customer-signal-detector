"""Pool several point-in-time pipeline runs and compare against backtest.py.

    python validation/check_pooled.py

Reports per-date AUC as well as pooled, because a single good quarter can carry
a pooled figure, and a bootstrap interval, because a point estimate on this many
events is what made the earlier single-date readings unreliable.

The accounts recur across dates, so the readings are not fully independent - the
quarterly spacing keeps the 90-day horizons from overlapping, so no churn event
is counted twice, but account-level correlation remains. Read the interval as
indicative rather than exact.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # import signals/adapt from the project root

import json

import numpy as np
import pandas as pd

CHURN = {"churned_voluntary", "churned_nonpayment", "downgraded"}
LABELS = ROOT / "data_extensive" / "prediction_labels_90d.csv"


def auc(score, y) -> float:
    d = pd.DataFrame({"s": np.asarray(score, dtype=float),
                      "y": np.asarray(y, dtype=int)}).dropna()
    pos, neg = int((d.y == 1).sum()), int((d.y == 0).sum())
    if not pos or not neg:
        return float("nan")
    r = d.s.rank()
    return float((r[d.y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def bootstrap(df, n=2000, seed=7) -> tuple[float, float]:
    """Resample ACCOUNTS, not rows - the same account recurs across dates."""
    rng = np.random.default_rng(seed)
    ids = df.id.unique()
    by_id = {k: g for k, g in df.groupby("id")}
    out = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        s = pd.concat([by_id[i] for i in pick], ignore_index=True)
        a = auc(s.urgency, s.churned)
        if not np.isnan(a):
            out.append(a)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main(paths: list[Path]) -> None:
    lab = pd.read_csv(LABELS, parse_dates=["prediction_date"])
    frames = []
    for p in paths:
        payload = json.loads(p.read_text(encoding="utf-8"))
        as_of = payload.get("as_of")
        if not as_of:
            print(f"skipping {p.name}: not a point-in-time run")
            continue
        nodes = pd.DataFrame(payload["nodes"])
        g = lab[(lab.prediction_date == pd.Timestamp(as_of))
                & lab.eligible_for_prediction
                & (lab.outcome_type != "censored")].copy()
        g["churned"] = g.outcome_type.isin(CHURN).astype(int)
        m = nodes.merge(g[["account_id", "churned"]], left_on="id", right_on="account_id")
        m["as_of"] = as_of
        frames.append(m)

    if not frames:
        sys.exit("no point-in-time runs found - generate some with "
                 "`python pipeline.py data_extensive/ --as-of YYYY-MM-DD -o data/nodes_asof_X.json`")

    df = pd.concat(frames, ignore_index=True)

    print(f"{'date':<14}{'n':>6}{'churn':>7}{'rate':>8}{'AUC':>8}")
    for d, g in df.groupby("as_of"):
        print(f"{d:<14}{len(g):>6}{int(g.churned.sum()):>7}"
              f"{g.churned.mean():>7.1%}{auc(g.urgency, g.churned):>8.3f}")
    lo, hi = bootstrap(df)
    print(f"{'POOLED':<14}{len(df):>6}{int(df.churned.sum()):>7}"
          f"{df.churned.mean():>7.1%}{auc(df.urgency, df.churned):>8.3f}"
          f"   95% CI [{lo:.3f}, {hi:.3f}]")

    ref = ROOT / "data" / "backtest_results.csv"
    if ref.exists():
        print("\nreference - backtest.py, hand-built features, same dates:")
        b = pd.read_csv(ref)
        b["prediction_date"] = pd.to_datetime(b.prediction_date)
        for d, g in b.groupby("prediction_date"):
            print(f"  {d:%Y-%m-%d}  n {len(g):>3}  AUC {auc(g.risk_score, g.churn):.3f}")
        print(f"  POOLED      n {len(b)}  AUC {auc(b.risk_score, b.churn):.3f}")

    print("\nworklist precision, pooled across cycles:")
    for k in (10, 20, 50):
        hits = tot = 0
        for _, g in df.groupby("as_of"):
            top = g.nlargest(k, "urgency")
            hits += int(top.churned.sum())
            tot += len(top)
        print(f"  top {k:>2} per cycle   precision {hits / max(tot, 1):>5.1%}   "
              f"lift {(hits / max(tot, 1)) / df.churned.mean():>4.1f}x")

    # Lead time is the whole point of an early-warning system: a tool that fires
    # the week someone cancels is a reporting layer, not a warning.
    out = pd.read_csv(ROOT / "data_extensive" / "outcomes.csv",
                      parse_dates=["decision_date"])
    dd = out.set_index("account_id").decision_date
    caught = df[(df.churned == 1) & (df.bucket != "stable")].copy()
    caught["lead_days"] = [
        (dd.get(i) - pd.Timestamp(d)).days if pd.notna(dd.get(i)) else None
        for i, d in zip(caught.id, caught.as_of)]
    ld = caught.lead_days.dropna()
    if len(ld):
        print(f"\nlead time on caught churn   median {ld.median():.0f} days   "
              f"p25 {ld.quantile(.25):.0f}   p75 {ld.quantile(.75):.0f}   (n {len(ld)})")

    print("\nbucket performance, pooled:")
    for key in ["urgent", "high", "watch", "stable"]:
        g = df[df.bucket == key]
        if len(g):
            print(f"  {key:<8} n {len(g):>4}  churn {g.churned.mean():>6.1%}  "
                  f"captures {g.churned.sum() / max(df.churned.sum(), 1):>5.1%}")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        args = sorted((ROOT / "data").glob("nodes_asof_*.json"))
    main(args)
