"""Score the LLM layer on its own task, independent of the churn outcome.

    python eval_llm.py [n]

Uses the 300 human-labelled messages in data_extensive/llm_evaluation_set.csv.
This isolates where error comes from: if the backtest is weak but this is strong,
the problem is the scoring model or the data, not the language layer.

The detector emits continuous scores, the labels are categorical, so the mapping
is fixed up front rather than fitted: sentiment_risk >0.6 negative, <0.35
positive, else neutral; churn_language >0.5 means churn intent.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # import signals/adapt from the project root
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import signals as sig
from backtest import _auc

SRC = ROOT / "data_extensive" / "llm_evaluation_set.csv"

SYSTEM = """You classify a single customer support message from a B2B SaaS account.

- sentiment_risk 0.0-1.0, where 1.0 is the most negative.
- churn_language 0.0-1.0. Score above 0.5 ONLY for explicit exit behaviour:
  asking about cancellation or notice periods, requesting a data export to
  leave, naming a competitor they are moving to, or saying they will not renew.
  Frustration about a bug or a billing error is not exit intent.
Judge only the message text. Do not infer history that is not there."""

SCHEMA = {
    "type": "object",
    "properties": {"sentiment_risk": {"type": "number"},
                   "churn_language": {"type": "number"}},
    "required": ["sentiment_risk", "churn_language"],
    "additionalProperties": False,
}


def classify(client, text: str) -> dict | None:
    try:
        r = client.chat.completions.create(
            model=sig.MODEL, temperature=0.0,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": text}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "message_label", "strict": True, "schema": SCHEMA}})
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        print(f"[warn] {e}")
        return None


def main(limit: int | None = None) -> None:
    client = sig._client()
    if client is None:
        raise sig.NoLLMError("eval_llm needs OPENAI_API_KEY in .env")

    df = pd.read_csv(SRC)
    if limit:
        df = df.sample(limit, random_state=7)

    cache = ROOT / "data" / "llm_eval_scores.csv"
    if cache.exists() and not limit:
        raw = pd.read_csv(cache)
        out = [{"sentiment_risk": a, "churn_language": b} if pd.notna(a) else None
               for a, b in zip(raw.sentiment_risk, raw.churn_language)]
        print(f"(reusing cached scores from {cache.name})")
    else:
        with ThreadPoolExecutor(max_workers=16) as pool:
            out = list(pool.map(lambda t: classify(client, t), df.text))
        pd.DataFrame({"text": df.text.values,
                      "sentiment_risk": [o["sentiment_risk"] if o else None for o in out],
                      "churn_language": [o["churn_language"] if o else None for o in out],
                      "human_sentiment": df.human_sentiment.values,
                      "human_intent": df.human_intent.values}).to_csv(cache, index=False)

    ok = [(o, s, i) for o, s, i in zip(out, df.human_sentiment, df.human_intent) if o]
    print(f"scored {len(ok)} of {len(df)} messages with {sig.MODEL}\n")

    pred_sent = ["negative" if o["sentiment_risk"] > 0.6 else
                 "positive" if o["sentiment_risk"] < 0.35 else "neutral" for o, _, _ in ok]
    truth_sent = [s for _, s, _ in ok]
    acc = sum(p == t for p, t in zip(pred_sent, truth_sent)) / len(ok)
    print(f"SENTIMENT  accuracy {acc:.1%}")
    print(pd.crosstab(pd.Series(truth_sent, name="human"),
                      pd.Series(pred_sent, name="model")).to_string(), "\n")

    truth_churn = [i == "churn_intent" for _, _, i in ok]
    pred_churn = [o["churn_language"] > 0.5 for o, _, _ in ok]
    tp = sum(p and t for p, t in zip(pred_churn, truth_churn))
    fp = sum(p and not t for p, t in zip(pred_churn, truth_churn))
    fn = sum(not p and t for p, t in zip(pred_churn, truth_churn))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    print(f"CHURN INTENT  precision {prec:.1%}  recall {rec:.1%}  "
          f"F1 {2 * prec * rec / max(prec + rec, 1e-9):.1%}  (tp {tp}, fp {fp}, fn {fn})")

    # Accuracy conflates ranking with threshold placement. Separate them: if the
    # ordering is good, the mapping is miscalibrated, not the model.
    risk = pd.Series([o["sentiment_risk"] for o, _, _ in ok])
    print()
    print("  ranking check - AUC(sentiment_risk vs human=negative)  "
          f"{_auc(risk, pd.Series(truth_sent) == 'negative'):.3f}")
    print("  AUC(churn_language vs human=churn_intent)             "
          f"{_auc(pd.Series([o['churn_language'] for o, _, _ in ok]), pd.Series(truth_churn)):.3f}")
    print("\n  score distribution by human label:")
    print(risk.groupby(pd.Series(truth_sent)).agg(["min", "mean", "max"]).round(2).to_string())

    miss = pd.DataFrame({"text": [t for t, (_, _, i) in zip(df.text[:len(ok)], ok)],
                         "human_intent": [i for _, _, i in ok],
                         "pred_churn": pred_churn})
    bad = miss[(miss.human_intent == "churn_intent") & ~miss.pred_churn]
    if len(bad):
        print("\nmissed churn-intent messages:")
        for t in bad.text.head(5):
            print(f"  - {t[:100]}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
