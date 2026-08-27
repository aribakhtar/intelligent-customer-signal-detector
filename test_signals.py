"""Self-check for the detection logic. Run: python test_signals.py

No API key needed and no network: the pipeline test injects a stub LLM client
that scores text from the archetype it can see in the transcript. That keeps the
maths, the wiring and the ranking under test without pretending a keyword
matcher is sentiment analysis - the real run always uses the model.
"""
import json
import tempfile
import types
from pathlib import Path

import pandas as pd

import make_data
import signals as sig

# The tests own their fixture: data/ holds whatever dataset is currently loaded
# for the demo, so generate a known synthetic set into a temp dir instead.
_FIXTURE = Path(tempfile.mkdtemp(prefix="signal-detector-fixture-"))
make_data.main(_FIXTURE)


def load_fixture():
    return sig.load(_FIXTURE)


def _stub_client(fail_for: set[str] = frozenset()):   # fail_for = account names
    """A fake OpenAI client. Reads the prompt, returns a plausible assessment.

    Stands in for the model only so the surrounding logic can be tested; it is
    never used by signals.py at runtime.
    """
    def create(**kw):
        prompt = kw["messages"][1]["content"]
        if any(name in prompt for name in fail_for):
            raise RuntimeError("simulated API failure")
        low = prompt.lower()
        exit_words = ("cancel", "notice period", "competitor", "evaluating alternatives",
                      "unlikely to renew", "data export")
        churn = 0.95 if any(w in low for w in exit_words) else 0.05
        upset = ("frustrating", "reopening", "charged twice", "blocking", "still waiting",
                 "not what we expected", "rocky start")
        quiet = ("pause seats", "not getting to it", "nobody has picked it back up")
        sent = 0.9 if churn > 0.5 else 0.75 if any(w in low for w in upset) \
            else 0.6 if any(w in low for w in quiet) else 0.1
        payload = {"text_sentiment_risk": sent, "churn_language": churn,
                   "issue": "stub issue", "themes": ["stub"],
                   "rationale": "Stub rationale.",
                   "recommended_action": "Stub action.", "action_owner": "CSM"}
        msg = types.SimpleNamespace(refusal=None, content=json.dumps(payload))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def test_units():
    assert sig._ratio_drop(50, 100) == 0.5
    assert sig._ratio_drop(120, 100) == 0.0, "growth is not a decline"
    assert sig._ratio_drop(0, 100) == 1.0
    assert sig._ratio_drop(10, 0) == 0.0, "no baseline -> no signal, not a divide-by-zero"
    assert abs(sum(sig.WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"


def test_structured_signals():
    r = pd.Series({
        "logins_prev_30d": 400, "logins_last_30d": 40,
        "tickets_prev_30d": 2, "tickets_last_30d": 8, "tickets_reopened_30d": 3,
        "open_p1_tickets": 2, "avg_first_response_hrs": 26.0,
        "failed_payments_90d": 3, "invoice_disputes_90d": 2, "days_payment_overdue": 45,
        "downgraded_last_90d": 1, "csat_current": 1.8, "csat_prev_quarter": 4.4, "nps_last": -50,
        "seats_active": 20, "seats_licensed": 200, "feature_adoption_pct": 0.1,
    })
    s = sig.structured_signals(r)
    assert all(0.0 <= v.score <= 1.0 for v in s.values()), "signals must stay normalised"
    assert s["usage_decline"].score > 0.9
    assert s["billing_friction"].score > 0.9
    assert "40" in s["usage_decline"].evidence, "evidence must cite the real numbers"

    healthy = r.copy()
    healthy.update(pd.Series({
        "logins_last_30d": 420, "tickets_last_30d": 1, "tickets_reopened_30d": 0,
        "open_p1_tickets": 0, "avg_first_response_hrs": 2.0, "failed_payments_90d": 0,
        "invoice_disputes_90d": 0, "days_payment_overdue": 0, "downgraded_last_90d": 0,
        "csat_current": 4.6, "csat_prev_quarter": 4.4, "nps_last": 60,
        "seats_active": 190, "feature_adoption_pct": 0.8}))
    h = sig.structured_signals(healthy)
    assert all(h[k].score < s[k].score for k in h), "healthy account must score lower everywhere"


def test_missing_columns_are_skipped_not_zeroed():
    """A feed without seat/NPS data must not read as 'healthy on that axis'."""
    full = pd.Series({
        "logins_prev_30d": 400, "logins_last_30d": 200,
        "tickets_prev_30d": 2, "tickets_last_30d": 6, "tickets_reopened_30d": 0,
        "open_p1_tickets": 0, "avg_first_response_hrs": 5.0,
        "failed_payments_90d": 0, "invoice_disputes_90d": 0, "days_payment_overdue": 0,
        "downgraded_last_90d": 0, "csat_current": 2.5, "csat_prev_quarter": 4.0,
        "nps_last": -20, "seats_active": 100, "seats_licensed": 100,
        "feature_adoption_pct": 0.9, "arr_usd": 50_000, "renewal_date": "2027-06-01",
    })
    sparse = full.drop(["seats_active", "seats_licensed", "feature_adoption_pct",
                        "nps_last", "invoice_disputes_90d", "downgraded_last_90d"])

    s_full, s_sparse = sig.structured_signals(full), sig.structured_signals(sparse)
    assert "engagement_breadth" in s_full and "engagement_breadth" not in s_sparse, \
        "a signal with no source data must be omitted, not emitted as 0"
    assert "NPS" in s_full["satisfaction_drop"].evidence
    assert "NPS" not in s_sparse["satisfaction_drop"].evidence, \
        "evidence must not cite a metric the feed does not carry"

    # The account is healthy on the dropped axes, so zeroing them would *lower*
    # the score. Renormalising must keep the two scores close instead.
    r_full = sig._score(s_full, full)[0]
    r_sparse = sig._score(s_sparse, sparse)[0]
    assert abs(r_full - r_sparse) <= 8, f"renormalisation drifted: {r_full} vs {r_sparse}"

    # Weights of present signals are renormalised, never silently summing to <1.
    assert sum(x.weight for x in s_full.values()) + sig.WEIGHTS["text_sentiment"] \
        + sig.WEIGHTS["churn_language"] == 1.0


def test_no_llm_is_an_error_not_a_fallback():
    cust, inter = load_fixture()
    real_client = sig._client
    sig._client = lambda: None          # simulate a missing / unusable key
    try:
        sig.detect(cust.head(1), inter)
    except sig.NoLLMError as e:
        assert "OPENAI_API_KEY" in str(e)
    else:
        raise AssertionError("detect() must refuse to run without an LLM, not degrade")
    finally:
        sig._client = real_client

    assert not hasattr(sig, "lexicon_text_signals"), "no non-LLM sentiment path may exist"
    assert not hasattr(sig, "_fallback_text"), "no template-rationale fallback may exist"


def test_llm_failure_is_reported_not_guessed():
    cust, inter = load_fixture()
    target = cust.account_name.iloc[0]
    res = sig.detect(cust.head(3), inter, client=_stub_client(fail_for={target}))
    bad = next(a for a in res if a.account_name == target)
    assert not bad.llm_used
    assert bad.rationale.startswith("LLM analysis unavailable")
    text_sig = next(s for s in bad.signals if s.name == "text_sentiment")
    assert text_sig.score == 0.0 and "NOT ANALYSED" in text_sig.evidence, \
        "a failed call must be surfaced, never filled in with a guessed number"


def test_percentile_bands_follow_rank_not_threshold():
    """Real books score low across the board; banding must still populate."""
    cust, inter = load_fixture()
    res = sig.detect(cust, inter, client=_stub_client())
    for a in res:                      # squash every score into the Healthy band
        a.risk_score = int(a.risk_score / 10)
        a.band = next(b for t, b in sig.BANDS if a.risk_score >= t)
    assert {a.band for a in res} == {"Healthy"}, "precondition: absolute bands collapse"

    sig.apply_percentile_bands(res)
    bands = [a.band for a in sorted(res, key=lambda x: x.risk_score, reverse=True)]
    assert bands[0] == "Critical" and bands[-1] == "Healthy"
    assert bands == sorted(bands, key=["Critical", "High", "Watch", "Healthy"].index),         "bands must be monotonic in risk score"
    flagged = sum(b != "Healthy" for b in bands)
    assert 0.10 * len(res) <= flagged <= 0.20 * len(res), f"flagged {flagged} of {len(res)}"


def test_pipeline_ranks_at_risk_first():
    cust, inter = load_fixture()
    res = sig.detect(cust, inter, client=_stub_client())
    df = sig.to_frame(res).merge(cust[["customer_id", "_archetype"]], on="customer_id")

    healthy = df[df._archetype == "healthy"]
    at_risk = df[df._archetype.isin(["churn_intent", "support_strain", "quiet_decline"])]
    assert at_risk.risk_score.mean() > healthy.risk_score.mean() + 20, \
        f"at-risk {at_risk.risk_score.mean():.0f} vs healthy {healthy.risk_score.mean():.0f}"
    assert healthy.risk_score.max() < 50, "no healthy account should reach the High band"
    assert df[df._archetype == "churn_intent"].risk_score.min() >= 50, \
        "every explicit-churn account must be flagged High or above"
    # The point of the whole thing: quiet decliners get caught without complaining.
    assert df[df._archetype == "quiet_decline"].risk_score.mean() > 40, \
        "silent decliners must surface, not hide behind the absence of complaints"
    assert res == sorted(res, key=lambda a: a.priority_score, reverse=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
