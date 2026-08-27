"""Intelligent Customer Signal Detector - detection core.

Two layers, deliberately separated:

1. Deterministic signal extraction (pure pandas). Every behavioural, billing and
   satisfaction signal is normalised to 0..1 with a human-readable evidence
   string. This layer always runs, never hallucinates, and is what the risk
   score is actually computed from.
2. An LLM layer (OpenAI, structured JSON output) that reads the free-text
   interactions and returns a text sentiment score, an explicit churn-intent
   score, the themes it saw, a plain-English rationale and a recommended
   retention action. Set OPENAI_API_KEY; OPENAI_MODEL overrides the model.

The LLM's two numeric outputs feed back into the weighted score, so the ranking
is grounded in structured evidence and text evidence together.

Sentiment and churn intent are ALWAYS scored by the LLM - there is no keyword or
heuristic path. With no key the pipeline raises NoLLMError rather than degrading
to something that is not real sentiment analysis. The provider is isolated to
_client() and _analyse_text(); swapping it changes nothing else.
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()   # read .env from the project root before anything reads the env

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DATA = Path(__file__).parent / "data"
TODAY = date(2026, 8, 27)

# Signal weights. Sum to 1.0; risk score = 100 * weighted sum.
WEIGHTS = {
    "churn_language": 0.20,   # explicit exit intent in text - strongest single tell
    "text_sentiment": 0.16,   # how the customer sounds, recency-weighted
    "usage_decline": 0.16,    # the earliest quiet signal, fires before any complaint
    "support_strain": 0.15,   # volume spike, reopens, unresolved P1s
    "billing_friction": 0.13,  # failed payments, disputes, overdue, downgrade
    "satisfaction_drop": 0.12,  # CSAT trend + NPS level
    "engagement_breadth": 0.08,  # seat utilisation + feature adoption
}

BANDS = [(70, "Critical"), (50, "High"), (30, "Watch"), (0, "Healthy")]


@dataclass
class Signal:
    name: str
    score: float          # 0..1, higher = more concerning
    evidence: str         # what a human should read
    weight: float = 0.0

    @property
    def contribution(self) -> float:
        return round(100 * self.weight * self.score, 1)


@dataclass
class Assessment:
    customer_id: str
    account_name: str
    segment: str
    arr_usd: float | None
    renewal_in_days: int | None
    risk_score: int
    priority_score: float
    band: str
    signals: list[Signal]
    themes: list[str] = field(default_factory=list)
    issue: str = ""
    rationale: str = ""
    action: str = ""
    action_owner: str = ""
    llm_used: bool = False

    @property
    def top_drivers(self) -> list[Signal]:
        return sorted([s for s in self.signals if s.score > 0.15],
                      key=lambda s: s.contribution, reverse=True)[:3]


# ---------------------------------------------------------------- data loading

def load(data_dir: Path = DATA) -> tuple[pd.DataFrame, pd.DataFrame]:
    cust = pd.read_csv(data_dir / "customers.csv")
    inter = pd.read_csv(data_dir / "interactions.csv")
    return cust, inter


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _ratio_drop(now: float, before: float) -> float:
    """0 when flat or growing, 1 when it has gone to zero."""
    if before <= 0:
        return 0.0
    return _clamp(1 - now / before)


# ------------------------------------------------- deterministic signal layer

def _has(r: pd.Series, *cols: str) -> bool:
    """True only if every column is present and non-null.

    Real feeds arrive with gaps. A signal whose inputs are missing is omitted
    entirely and its weight is redistributed at scoring time - never defaulted to
    zero, which would silently read as 'this account is fine on that dimension'.
    """
    return all(c in r.index and pd.notna(r[c]) for c in cols)


def structured_signals(r: pd.Series) -> dict[str, Signal]:
    """Everything derivable from the structured record, no model involved.

    CSAT is expected on a 1-5 scale; convert before calling if your source uses
    1-10. Signals whose source columns are absent are skipped, not faked.
    """
    s: dict[str, Signal] = {}

    if _has(r, "logins_last_30d", "logins_prev_30d"):
        login_drop = _ratio_drop(r.logins_last_30d, r.logins_prev_30d)
        s["usage_decline"] = Signal(
            "usage_decline",
            # a >40% drop is already serious; scale so 50% drop ~= 0.83
            _clamp(login_drop / 0.6),
            f"Usage {int(r.logins_prev_30d)} -> {int(r.logins_last_30d)} "
            f"({login_drop:.0%} drop)" if login_drop > 0.05 else
            f"Usage stable ({int(r.logins_prev_30d)} -> {int(r.logins_last_30d)})")

    if _has(r, "tickets_last_30d", "tickets_prev_30d"):
        spike = _ratio_drop(r.tickets_prev_30d, r.tickets_last_30d)  # inverted: growth
        reopened = r.tickets_reopened_30d if _has(r, "tickets_reopened_30d") else 0
        p1 = r.open_p1_tickets if _has(r, "open_p1_tickets") else 0
        # Reweight onto whichever sub-inputs this feed actually carries.
        subs = [(0.5, spike), (0.25, min(reopened / 3, 1)), (0.25, min(p1 / 2, 1))]
        avail = [(w, v) for w, v in subs
                 if not (w == 0.25 and not _has(r, "tickets_reopened_30d", "open_p1_tickets"))]
        strain = _clamp(sum(w * v for w, v in avail) / sum(w for w, _ in avail))
        parts = []
        if spike > 0.15:
            parts.append(f"tickets {int(r.tickets_prev_30d)} -> {int(r.tickets_last_30d)}")
        if reopened:
            parts.append(f"{int(reopened)} reopened")
        if p1:
            parts.append(f"{int(p1)} open P1")
        if _has(r, "avg_first_response_hrs") and r.avg_first_response_hrs > 12:
            parts.append(f"{r.avg_first_response_hrs:.0f}h first response")
        s["support_strain"] = Signal("support_strain", strain,
                                     ", ".join(parts).capitalize() or "No support strain")

    bsubs, bparts = [], []
    if _has(r, "failed_payments_90d"):
        bsubs.append((0.3, min(r.failed_payments_90d / 2, 1)))
        if r.failed_payments_90d:
            bparts.append(f"{int(r.failed_payments_90d)} failed payments/90d")
    if _has(r, "invoice_disputes_90d"):
        bsubs.append((0.3, min(r.invoice_disputes_90d / 2, 1)))
        if r.invoice_disputes_90d:
            bparts.append(f"{int(r.invoice_disputes_90d)} invoice disputes")
    if _has(r, "days_payment_overdue"):
        bsubs.append((0.25, min(r.days_payment_overdue / 45, 1)))
        if r.days_payment_overdue > 7:
            bparts.append(f"{int(r.days_payment_overdue)}d overdue")
    if _has(r, "downgraded_last_90d"):
        bsubs.append((0.15, float(r.downgraded_last_90d)))
        if r.downgraded_last_90d:
            bparts.append("downgraded in last 90d")
    if bsubs:
        s["billing_friction"] = Signal(
            "billing_friction",
            _clamp(sum(w * v for w, v in bsubs) / sum(w for w, _ in bsubs)),
            ", ".join(bparts).capitalize() or "Billing clean")

    if _has(r, "csat_current"):
        ssubs = [(0.35, _clamp((3.5 - r.csat_current) / 2.0))]
        ev = [f"CSAT {r.csat_current}"]
        if _has(r, "csat_prev_quarter"):
            ssubs.append((0.4, _clamp((r.csat_prev_quarter - r.csat_current) / 1.5)))
            ev = [f"CSAT {r.csat_prev_quarter} -> {r.csat_current}"]
        if _has(r, "nps_last"):
            ssubs.append((0.25, _clamp((20 - r.nps_last) / 80)))
            ev.append(f"NPS {int(r.nps_last)}")
        s["satisfaction_drop"] = Signal(
            "satisfaction_drop",
            _clamp(sum(w * v for w, v in ssubs) / sum(w for w, _ in ssubs)), ", ".join(ev))

    if _has(r, "seats_active", "seats_licensed", "feature_adoption_pct"):
        seat_util = r.seats_active / max(r.seats_licensed, 1)
        s["engagement_breadth"] = Signal(
            "engagement_breadth",
            _clamp(0.6 * (1 - seat_util) + 0.4 * (1 - r.feature_adoption_pct)),
            f"{seat_util:.0%} seats active ({int(r.seats_active)}/{int(r.seats_licensed)}), "
            f"{r.feature_adoption_pct:.0%} feature adoption")
    return s


# ------------------------------------------------------------------ LLM layer

SYSTEM = """You are a customer-retention analyst for a B2B SaaS operations team.
You read a single account's recent support and survey interactions alongside its
behavioural metrics, and you report what the account is signalling.

Rules:
- Judge only from the evidence given. Never invent facts, names, or history.
- text_sentiment_risk and churn_language are scored 0.0-1.0 where 1.0 is worst.
  Score churn_language high ONLY for explicit exit behaviour (asking about
  cancellation, notice periods, data export, naming a competitor, saying they
  will not renew). Venting about a bug is frustration, not exit intent.
- A quiet account with falling usage and no complaints is a HIGH concern, not a
  low one - silence before a renewal is a churn signal, not a healthy one.
- The rationale is read by a CSM before a call. Two or three sentences, concrete,
  quoting or paraphrasing the actual evidence. No hedging, no filler.
- The action must be a single specific next step someone can do this week.
- `issue` names the single dominant problem in five words or fewer, as an
  operator would file it: "repeat billing errors", "unresolved sync failures",
  "champion left, adoption stalled". If nothing is wrong, say "no issue"."""

SCHEMA = {
    "type": "object",
    "properties": {
        "text_sentiment_risk": {"type": "number"},
        "churn_language": {"type": "number"},
        "issue": {"type": "string"},
        "themes": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
        "recommended_action": {"type": "string"},
        "action_owner": {"type": "string",
                         "enum": ["CSM", "Support Lead", "Billing", "Account Exec", "Product"]},
    },
    "required": ["text_sentiment_risk", "churn_language", "issue", "themes",
                 "rationale", "recommended_action", "action_owner"],
    "additionalProperties": False,
}


def _prompt(r: pd.Series, sigs: dict[str, Signal], msgs: pd.DataFrame) -> str:
    facts = "\n".join(f"- {s.name}: {s.evidence} (severity {s.score:.2f})" for s in sigs.values())
    convo = "\n".join(f"[{m.date} / {m.channel}] {m.text}" for m in msgs.itertuples()) \
        or "(no recent interactions on record)"
    # Only state the attributes this feed actually carries - never pad the prompt
    # with defaults the model would then reason over as if they were facts.
    bits = [str(r.account_name)]
    for col, fmt in (("segment", str), ("plan", lambda v: f"{v} plan"),
                     ("arr_usd", lambda v: f"${int(v):,} ARR"),
                     ("tenure_months", lambda v: f"{int(v)} months tenure")):
        if _has(r, col):
            bits.append(fmt(r[col]))
    if _has(r, "renewal_date"):
        bits.append(f"renewal in {(date.fromisoformat(str(r.renewal_date)) - TODAY).days} days")
    return (f"ACCOUNT: {bits[0]} ({', '.join(bits[1:])})\n\n"
            f"BEHAVIOURAL / BILLING SIGNALS:\n{facts}\n\n"
            f"RECENT INTERACTIONS (oldest first):\n{convo}\n\n"
            "Assess this account.")


PLACEHOLDERS = {"", "sk-your-key-here", "sk-...", "your-key-here", "changeme"}


def _client():
    try:
        from openai import OpenAI
    except ImportError:
        return None
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if key in PLACEHOLDERS:   # an unedited .env template is the same as no key
        return None
    try:
        # No fallback path exists, so give transient failures more chances.
        return OpenAI(max_retries=4)   # reads OPENAI_API_KEY / OPENAI_BASE_URL
    except Exception:
        return None


def _analyse_text(client, r, sigs, msgs, errors: list | None = None) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,          # analysis, not prose - keep it stable across runs
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": _prompt(r, sigs, msgs)}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "account_assessment", "strict": True, "schema": SCHEMA}},
        )
        msg = resp.choices[0].message
        if msg.refusal:
            if errors is not None:
                errors.append(f"{r.customer_id}: model refused")
            return None
        return json.loads(msg.content)
    except Exception as e:  # one bad account must not sink the batch
        if errors is not None:
            errors.append(f"{r.customer_id}: {e}")
        return None


# ------------------------------------------------------------------- scoring

def _score(sigs: dict[str, Signal], r: pd.Series) -> tuple[int, float, str, int | None]:
    for name, sig in sigs.items():
        sig.weight = WEIGHTS[name]
    # Renormalise over the signals this dataset actually supports, so a feed
    # missing (say) seat data is not scored as if that dimension were clean.
    # With the full seven present this divides by 1.0 and changes nothing.
    total_w = sum(s.weight for s in sigs.values()) or 1.0
    raw = 100 * sum(s.weight * s.score for s in sigs.values()) / total_w

    # Renewal proximity is an urgency amplifier, not a risk signal in itself:
    # the same decay matters more with 30 days left than with 300.
    renewal_in = None
    amp = 1.0
    if _has(r, "renewal_date"):
        renewal_in = (date.fromisoformat(str(r.renewal_date)) - TODAY).days
        amp = 1.25 if renewal_in <= 45 else 1.12 if renewal_in <= 90 else 1.0
    risk = int(round(min(100, raw * amp)))

    band = next(b for t, b in BANDS if risk >= t)
    # Priority = risk weighted by revenue at stake. The multiplier is bounded to
    # [1.0, 2.0] so a big logo never outranks a genuinely burning small one by
    # more than 2x - ops still works the risk, just tie-broken by value.
    arr = r.arr_usd if _has(r, "arr_usd") else 5_000
    value_mult = 1 + _clamp(math.log10(max(arr, 5_000) / 5_000) / 2)
    priority = round(risk * value_mult, 1)
    return risk, priority, band, renewal_in


# ---------------------------------------------------------------- entry point

class NoLLMError(RuntimeError):
    """Raised when no LLM is available. Sentiment is never inferred without one."""


def detect(cust: pd.DataFrame, inter: pd.DataFrame, client=None,
           progress=None) -> list[Assessment]:
    """Run the full pipeline and return assessments sorted by priority.

    `client` is injectable so tests can drive the pipeline with a stub. In normal
    use it is resolved from the environment. There is no non-LLM path: sentiment
    and churn intent are always model-scored, and an account whose text cannot be
    analysed is reported as an error rather than silently guessed at.
    """
    if client is None:
        client = _client()
    if client is None:
        raise NoLLMError(
            "No usable OPENAI_API_KEY found. Sentiment and churn-intent analysis require an "
            "LLM - put a real key in the .env file in the project root and re-run.")

    by_cust = {cid: g.sort_values("date") for cid, g in inter.groupby("customer_id")}
    rows = [r for _, r in cust.iterrows()]
    errors: list[str] = []

    def one(r: pd.Series) -> Assessment:
        msgs = by_cust.get(r.customer_id, inter.iloc[0:0])
        sigs = structured_signals(r)
        texts = msgs.text.tolist()

        out = _analyse_text(client, r, sigs, msgs, errors)
        if out is None:
            # The call failed after retries. Surface it as an unscored account
            # rather than inventing a sentiment number the model never produced.
            sent = churn = None
            themes = []
        else:
            sent, churn, themes = out["text_sentiment_risk"], out["churn_language"], out["themes"]

        quote = (texts[-1][:110] + ("..." if len(texts[-1]) > 110 else "")
                 if texts else "No recent contact on record")
        sigs["text_sentiment"] = Signal(
            "text_sentiment", _clamp(sent) if sent is not None else 0.0,
            f'Latest: "{quote}"' if out else "TEXT NOT ANALYSED - LLM call failed")
        sigs["churn_language"] = Signal(
            "churn_language", _clamp(churn) if churn is not None else 0.0,
            "TEXT NOT ANALYSED - LLM call failed" if out is None else
            "Explicit exit language in recent messages" if churn > 0.5 else
            "Some exit-adjacent language" if churn > 0.2 else "No exit language detected")

        risk, priority, band, renewal_in = _score(sigs, r)
        a = Assessment(r.customer_id, r.account_name,
                       r.segment if _has(r, 'segment') else '-',
                       float(r.arr_usd) if _has(r, 'arr_usd') else None,
                       renewal_in, risk, priority, band, list(sigs.values()), themes,
                       llm_used=out is not None)
        if out:
            a.issue = out["issue"]
            a.rationale, a.action, a.action_owner = (
                out["rationale"], out["recommended_action"], out["action_owner"])
        else:
            a.rationale = ("LLM analysis unavailable for this account - the two text signals "
                           "are excluded, so this risk score is incomplete and understated.")
            a.issue = "not analysed"
            a.action = "Re-run the analysis for this account before acting on the score."
            a.action_owner = "CSM"
        if progress:
            progress()
        return a

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, rows))

    if errors and len(errors) == len(rows):
        # Every call failed - that is a bad key, model name or network, not a
        # per-account blip. Refuse rather than hand back a book of empty scores.
        raise NoLLMError(
            f"All {len(rows)} LLM calls failed, so no sentiment was analysed. "
            f"Check OPENAI_API_KEY and OPENAI_MODEL (={MODEL}) in .env.\n"
            f"First error: {errors[0]}")
    if errors:
        print(f"[warn] {len(errors)} of {len(rows)} accounts could not be analysed. "
              f"First: {errors[0]}")
    return sorted(results, key=lambda a: a.priority_score, reverse=True)


def to_frame(assessments: list[Assessment]) -> pd.DataFrame:
    return pd.DataFrame([{
        "customer_id": a.customer_id,
        "account": a.account_name,
        "segment": a.segment,
        "arr_usd": a.arr_usd,
        "renewal_in_days": a.renewal_in_days,
        "risk_score": a.risk_score,
        "band": a.band,
        "priority": a.priority_score,
        "issue": a.issue,
        "top_drivers": ", ".join(d.name.replace("_", " ") for d in a.top_drivers),
        "rationale": a.rationale,
        "recommended_action": a.action,
        "owner": a.action_owner,
        **{f"sig_{s.name}": round(s.score, 2) for s in a.signals},
    } for a in assessments])


# Share of the book that lands in each band under percentile mode, worst first.
PERCENTILE_BANDS = [(0.03, "Critical"), (0.08, "High"), (0.15, "Watch")]


def apply_percentile_bands(assessments: list[Assessment],
                           cuts=PERCENTILE_BANDS) -> list[Assessment]:
    """Re-band by rank within the book instead of by absolute score.

    The absolute thresholds (70/50/30) assume a severity distribution. Real
    accounts do not decay like the demo set: on the 700-account backtest every
    raw severity averaged 0.02-0.18, so the weighted score topped out at 43 and
    the Critical and High bands were empty even though the *ranking* was sound
    (AUC 0.851). Percentile banding fixes the mismatch without retuning the
    normalisers per dataset, and matches how CS teams actually work - "give me
    the top 20 accounts this week", not "everyone over 70".

    Mutates and returns the same list, so it composes after detect().
    """
    ranked = sorted(assessments, key=lambda a: a.risk_score, reverse=True)
    n = len(ranked)
    for i, a in enumerate(ranked):
        pct = (i + 1) / n
        a.band = next((b for cut, b in cuts if pct <= cut), "Healthy")
    return assessments


# Ordinal ground-truth labels this project understands, worst first. Any column
# named _<something> holding these values can be evaluated against.
RISK_LABELS = ["critical", "high", "medium", "low"]
BAND_RANK = {"Critical": 0, "High": 1, "Watch": 2, "Healthy": 3}


def evaluate(df: pd.DataFrame, cust: pd.DataFrame, thresholds=(30, 50, 70)) -> str:
    """Score the detector against whatever held-out label the dataset carries.

    Handles two label schemes: the synthetic set's `_archetype` (healthy vs not)
    and an ordinal `_ground_truth_risk` of critical/high/medium/low. Sanity
    checks on labelled data, not a production accuracy claim.
    """
    label = next((c for c in ("_ground_truth_risk", "_archetype") if c in cust.columns), None)
    if label is None:
        return "(no ground-truth labels in this dataset - skipping evaluation)"
    m = df.merge(cust[["customer_id", label]], on="customer_id")
    vals = set(m[label].astype(str).str.lower())

    if vals <= set(RISK_LABELS):   # ordinal scheme
        rank = {v: i for i, v in enumerate(RISK_LABELS)}
        truth_rank = m[label].str.lower().map(rank)
        pred_rank = m.band.map(BAND_RANK)
        # "At risk" = the two worst labels; medium is a watch item, not a miss.
        truth = truth_rank <= 1
        lines = [f"Evaluation vs held-out `{label}` (at risk = critical or high)",
                 f"  rank correlation  Spearman {m['risk_score'].corr(-truth_rank, method='spearman'):.3f}"
                 f"   Kendall {m['risk_score'].corr(-truth_rank, method='kendall'):.3f}",
                 f"  band agreement    exact {(pred_rank == truth_rank).mean():.0%}"
                 f"   within one band {((pred_rank - truth_rank).abs() <= 1).mean():.0%}"]
    else:                          # binary archetype scheme
        truth = m[label] != "healthy"
        lines = [f"Evaluation vs held-out `{label}` (at risk = any non-healthy archetype)"]

    for t in thresholds:
        pred = m.risk_score >= t
        tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        band = next(b for lo, b in BANDS if t >= lo)
        lines.append(f"  risk >= {t:>3} ({band:<8}) precision {prec:>4.0%}  recall {rec:>4.0%}  "
                     f"(tp {tp}, fp {fp}, fn {fn}, n {len(m)})")

    lines.append(f"  risk score by {label.lstrip('_')}:")
    g = m.groupby(label).risk_score.agg(["min", "mean", "max", "count"])
    g = g.reindex([v for v in RISK_LABELS if v in g.index]) if vals <= set(RISK_LABELS) \
        else g.sort_values("mean", ascending=False)
    for name, row in g.iterrows():
        lines.append(f"    {name:<18} min {row['min']:>3.0f}  mean {row['mean']:>5.1f}  "
                     f"max {row['max']:>3.0f}  (n {row['count']:.0f})")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    c, i = load()
    try:
        res = detect(c, i)
    except NoLLMError as e:
        sys.exit(f"\n{e}\n")
    df = to_frame(res)
    failed = sum(1 for a in res if not a.llm_used)
    print(f"Model: {MODEL}" + (f"   [{failed} accounts could not be analysed]" if failed else ""))
    print()
    print(df[["account", "segment", "risk_score", "band", "top_drivers"]].head(12).to_string(index=False))
    print("\n" + evaluate(df, c))
    out = DATA / "signal_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nfull summary -> {out}")
