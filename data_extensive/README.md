# Customer Signal Detector — realistic synthetic backtest data

Synthetic longitudinal SaaS/customer-success dataset.

- 700 accounts
- 16,800 monthly account snapshots over 24 months
- 6,771 timestamped raw support messages
- 6,219 support tickets
- 700 outcome records
- 12,264 90-day prediction-label rows
- 300 independently labelled LLM evaluation messages

## Temporal leakage rule
For prediction date T, features may only use records with timestamps <= T.
Do not use outcomes, decision_date, future snapshots or post-decision messages as predictors.

`is_censored=true` means the future outcome is unknown and must not be treated as a negative.

Use an out-of-time split, not a random account-month split.

This dataset is synthetic and is intended for engineering/backtesting validation, not real-world churn-rate claims.
