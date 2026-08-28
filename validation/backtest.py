"""Point-in-time backtest of the signal detector against real outcomes.

    python backtest.py                 # default: three quarterly prediction dates
    python backtest.py 2025-06-30      # a single date

For each prediction date T the detector is scored using ONLY records timestamped
<= T, then evaluated against what actually happened in the following 90 days.

Two things in this dataset will silently ruin a backtest if taken at face value:

1. `outcome_within_90d == 1` includes `renewed` - 1176 of its 1571 positives are
   renewals. It means "an outcome event occurred", not "the customer left". The
   churn label is rebuilt here from `outcome_type`.
2. `eligible_for_prediction` is True for censored rows too. Censored means the
   outcome is unknown; those rows are dropped rather than counted as negatives.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # import signals/adapt from the project root
import sys
from datetime import date
from pathlib import Path

import pandas as pd

import signals as sig

SRC = ROOT / "data_extensive"
CHURN = {"churned_voluntary", "churned_nonpayment", "downgraded"}
# Quarterly, so the 90-day horizons do not overlap and no account is counted twice.
DEFAULT_DATES = ["2025-03-31", "2025-06-30", "2025-09-30"]
BASELINE_MONTHS = 3   # "current vs prior quarter" - see adapt.py for why not 1
MAX_MSGS = 6          # most recent customer messages fed to the model


def _load():
    snaps = pd.read_csv(SRC / "account_monthly_snapshots.csv", parse_dates=["snapshot_date"])
    msgs = pd.read_csv(SRC / "support_messages.csv", parse_dates=["timestamp"])
    tix = pd.read_csv(SRC / "support_tickets.csv", parse_dates=["created_at", "resolved_at"])
    acct = pd.read_csv(SRC / "accounts.csv", parse_dates=["contract_end_date"])
    labels = pd.read_csv(SRC / "prediction_labels_90d.csv", parse_dates=["prediction_date"])
    outcomes = pd.read_csv(SRC / "outcomes.csv", parse_dates=["decision_date"])
    return snaps, msgs, tix, acct, labels, outcomes


def features_at(T: pd.Timestamp, snaps, msgs, tix, acct, accounts: set[str]):
    """Build the detector's two input frames using only records dated <= T."""
    hist = snaps[(snaps.snapshot_date <= T) & (snaps.account_id.isin(accounts))]
    base_date = T - pd.DateOffset(months=BASELINE_MONTHS)

    cur = hist[hist.snapshot_date == T].set_index("account_id")
    base = (hist[hist.snapshot_date <= base_date]
            .sort_values("snapshot_date").groupby("account_id").last())

    # Ticket detail the monthly snapshot does not carry, from tickets opened
    # in the 30 days before T and not resolved by T.
    win = tix[(tix.created_at <= T) & (tix.created_at > T - pd.Timedelta(days=30))
              & (tix.account_id.isin(accounts))]
    open_at_T = win[win.resolved_at.isna() | (win.resolved_at > T)]
    reopened = win[win.reopened].groupby("account_id").size()
    critical_open = open_at_T[open_at_T.severity == "critical"].groupby("account_id").size()

    end_date = acct.set_index("account_id").contract_end_date

    rows = []
    for aid in sorted(accounts & set(cur.index)):
        c = cur.loc[aid]
        b = base.loc[aid] if aid in base.index else None
        rec = {
            "customer_id": aid,
            "account_name": aid,
            "segment": acct.loc[acct.account_id == aid, "tier"].iloc[0],
            "arr_usd": float(c.arr),
            "csat_current": round(float(c.csat_score) / 2, 2),     # 1-10 -> 1-5
            "nps_last": float(c.nps_score),
            "failed_payments_90d": float(c.payment_failure_count_30d),
            "days_payment_overdue": float(c.invoice_overdue_days),
            "tickets_reopened_30d": float(reopened.get(aid, 0)),
            "open_p1_tickets": float(critical_open.get(aid, 0)),
        }
        if b is not None:
            rec["logins_last_30d"] = float(c.monthly_usage_pct)
            rec["logins_prev_30d"] = float(b.monthly_usage_pct)
            rec["tickets_last_30d"] = float(c.support_tickets_opened)
            rec["tickets_prev_30d"] = float(b.support_tickets_opened)
            rec["csat_prev_quarter"] = round(float(b.csat_score) / 2, 2)
        # Contract end is set at signing, so it is known at T - not leakage.
        if aid in end_date.index and pd.notna(end_date[aid]):
            rec["renewal_date"] = end_date[aid].date().isoformat()
        rows.append(rec)

    # Customer-authored messages only: agent text would pollute the sentiment score.
    m = msgs[(msgs.timestamp <= T) & (msgs.author_role == "customer")
             & (msgs.account_id.isin(accounts))].sort_values("timestamp")
    m = m.groupby("account_id").tail(MAX_MSGS)
    inter = pd.DataFrame({
        "customer_id": m.account_id,
        "date": m.timestamp.dt.date.astype(str),
        "channel": m.channel,
        "text": m.text,
    })
    return pd.DataFrame(rows), inter


def run_date(T_str: str, data) -> pd.DataFrame:
    snaps, msgs, tix, acct, labels, outcomes = data
    T = pd.Timestamp(T_str)

    lab = labels[(labels.prediction_date == T) & labels.eligible_for_prediction].copy()
    lab = lab[lab.outcome_type != "censored"]          # unknown, not negative
    lab["churn"] = lab.outcome_type.isin(CHURN).astype(int)
    accounts = set(lab.account_id)

    cust, inter = features_at(T, snaps, msgs, tix, acct, accounts)
    if cust.empty:
        return pd.DataFrame()

    # Renewal proximity must be measured from the prediction date, not from now.
    res = sig.detect(cust, inter, as_of=T.date())
    # Band within each cycle's customers - see signals.apply_percentile_bands.
    sig.apply_percentile_bands(res)

    out = sig.to_frame(res).merge(
        lab[["account_id", "churn", "outcome_type"]],
        left_on="customer_id", right_on="account_id")
    out["prediction_date"] = T
    dd = outcomes.set_index("account_id").decision_date
    out["lead_days"] = out.customer_id.map(lambda a: (dd.get(a) - T).days
                                           if pd.notna(dd.get(a)) else None)
    return out


def report(df: pd.DataFrame, data) -> None:
    snaps = data[0]
    df = df.copy()
    df["prediction_date"] = pd.to_datetime(df.prediction_date)   # survives a CSV round-trip
    n, pos = len(df), int(df.churn.sum())
    print(f"\n{'=' * 74}\nBACKTEST  {n} account-months, {pos} churn events "
          f"({pos / n:.1%} base rate), {df.prediction_date.nunique()} prediction dates\n{'=' * 74}")

    print("\nRanking quality")
    print(f"  ROC AUC (rank-based)      {_auc(df.risk_score, df.churn):.3f}")

    print("\nWorklist precision - if a CSM works the top N accounts each cycle")
    per_date = df.prediction_date.nunique()
    for k in (10, 20, 50):
        hits = tot = 0
        for _, g in df.groupby("prediction_date"):
            top = g.nlargest(k, "risk_score")
            hits += int(top.churn.sum())
            tot += len(top)
        print(f"  top {k:>2} per cycle          precision {hits / max(tot, 1):>5.1%}   "
              f"({hits} of {tot})   lift {(hits / max(tot, 1)) / df.churn.mean():>4.1f}x base rate")

    print("\nBand performance")
    for band in ("Critical", "High", "Watch"):
        sub = df[df.band == band]
        if len(sub):
            print(f"  {band:<9} n {len(sub):>4}   churn rate {sub.churn.mean():>5.1%}   "
                  f"captures {sub.churn.sum() / max(pos, 1):>5.1%} of all churn")
    flagged = df[df.band != "Healthy"]
    print(f"  flagged (top 15%)  n {len(flagged)}   precision {flagged.churn.mean():.1%}   "
          f"recall {flagged.churn.sum() / max(pos, 1):.1%}")

    print("\nLift over simpler baselines (AUC)")
    base = df.merge(snaps, left_on=["customer_id", "prediction_date"],
                    right_on=["account_id", "snapshot_date"], how="left")
    rules = [k for k in sig.WEIGHTS if k not in ("text_sentiment", "churn_language")
             and f"sig_{k}" in df.columns]
    tw = sum(sig.WEIGHTS[k] for k in rules)
    rules_only = sum(df[f"sig_{k}"] * sig.WEIGHTS[k] for k in rules) / tw
    text_only = ((df.sig_text_sentiment * sig.WEIGHTS["text_sentiment"]
                  + df.sig_churn_language * sig.WEIGHTS["churn_language"])
                 / (sig.WEIGHTS["text_sentiment"] + sig.WEIGHTS["churn_language"]))
    print(f"  CSAT alone (inverted)     {_auc(-base.csat_score.fillna(5), base.churn):.3f}")
    print(f"  usage decline alone       {_auc(df.sig_usage_decline, df.churn):.3f}")
    print(f"  deterministic rules only  {_auc(rules_only, df.churn):.3f}")
    print(f"  LLM text signals only     {_auc(text_only, df.churn):.3f}")
    print(f"  full detector             {_auc(df.risk_score, df.churn):.3f}")

    tp = df[(df.churn == 1) & (df.band != "Healthy")]
    if len(tp) and tp.lead_days.notna().any():
        ld = tp.lead_days.dropna()
        print(f"\nLead time on caught churn  median {ld.median():.0f} days   "
              f"p25 {ld.quantile(.25):.0f}   p75 {ld.quantile(.75):.0f}   (n {len(ld)})")


def _auc(score, y) -> float:
    """Rank-based AUC - equivalent to the Mann-Whitney U statistic."""
    d = pd.DataFrame({"s": pd.Series(score).values, "y": pd.Series(y).values}).dropna()
    pos, neg = (d.y == 1).sum(), (d.y == 0).sum()
    if not pos or not neg:
        return float("nan")
    r = d.s.rank()
    return float((r[d.y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


if __name__ == "__main__":
    dates = sys.argv[1:] or DEFAULT_DATES
    data = _load()
    frames = []
    for d in dates:
        print(f"scoring {d} ...", flush=True)
        f = run_date(d, data)
        print(f"  {len(f)} accounts, {int(f.churn.sum())} churn in next 90d")
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(ROOT / "data" / "backtest_results.csv", index=False)
    report(df, data)
    print(f"\nper-account results -> data/backtest_results.csv")
