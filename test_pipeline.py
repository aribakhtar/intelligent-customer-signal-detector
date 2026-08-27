"""Self-check for the four-layer pipeline. Run: python test_pipeline.py

No API key and no network - the LLM steps are stubbed. What is under test is the
wiring: file identification, the merge, mapping validation, and bucketing.
"""
import json
import tempfile
import types
from pathlib import Path

import pandas as pd

import pipeline as pl
import signals as sig

TMP = Path(tempfile.mkdtemp(prefix="pipeline-test-"))


def _write(name, df):
    p = TMP / name
    df.to_csv(p, index=False)
    return p


def _map_client(payload):
    def create(**kw):
        msg = types.SimpleNamespace(refusal=None, content=json.dumps(payload))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def test_fingerprints_beat_the_llm():
    """A recognised file must never cost an API call."""
    p = _write("known.csv", pd.DataFrame({"customer_id": ["A"], "arr_usd": [1],
                                          "csat_current": [3.0]}))
    def boom(**kw):
        raise AssertionError("fingerprinted file must not reach the LLM")
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=boom)))
    spec = pl.classify(p, client)
    assert spec["kind"] == "accounts_wide" and spec["via"] == "fingerprint"

    p = _write("chat.csv", pd.DataFrame({"customer_id": ["A"], "text": ["hi"]}))
    assert pl.classify(p, client)["kind"] == "interactions"


def test_invalid_mappings_are_rejected():
    """Both ends of every mapping must be real, or the score reads wrong data."""
    p = _write("odd.csv", pd.DataFrame({"Ref": ["A"], "ScoreOutOfTen": [8.0]}))
    client = _map_client({
        "kind": "accounts_wide", "id_column": "Ref", "note": "",
        "mappings": [
            {"source_column": "ScoreOutOfTen", "canonical_field": "csat_current",
             "needs_halving": True},
            {"source_column": "ScoreOutOfTen", "canonical_field": "invented_field",
             "needs_halving": False},          # hallucinated target
            {"source_column": "NoSuchColumn", "canonical_field": "arr_usd",
             "needs_halving": False},          # hallucinated source
        ]})
    spec = pl.classify(p, client)
    assert [m["canonical_field"] for m in spec["mappings"]] == ["csat_current"]

    cust, _, _ = pl.consolidate([spec])
    assert cust.csat_current.iloc[0] == 4.0, "1-10 source must be halved to the 1-5 scale"


def test_consolidation_merges_without_blanking():
    """A partial feed adds to a profile; it never wipes what another feed filled."""
    a = pl.classify(_write("a.csv", pd.DataFrame({
        "customer_id": ["A", "B"], "account_name": ["Acme", "Beta"],
        "arr_usd": [1000, 2000], "csat_current": [2.0, 4.0]})))
    b = pl.classify(_write("b.csv", pd.DataFrame({
        "customer_id": ["A"], "days_payment_overdue": [30],
        "account_name": [None]})))          # null must not overwrite "Acme"
    c = pl.classify(_write("c.csv", pd.DataFrame({
        "customer_id": ["A"], "text": ["we are cancelling"], "date": ["2026-01-01"],
        "channel": ["email"]})))

    cust, inter, prov = pl.consolidate([a, b, c])
    row = cust.set_index("customer_id").loc["A"]
    assert row.account_name == "Acme", "a null in a later feed must not blank a filled field"
    assert row.days_payment_overdue == 30, "later feed must contribute its own fields"
    assert len(cust) == 2 and len(inter) == 1
    assert prov["A"] == ["a.csv", "b.csv", "c.csv"], "provenance must list every source"


def test_agent_text_is_excluded():
    spec = pl.classify(_write("msgs.csv", pd.DataFrame({
        "customer_id": ["A", "A"], "author_role": ["agent", "customer"],
        "text": ["Happy to help!", "This is unacceptable"],
        "date": ["2026-01-01", "2026-01-02"], "channel": ["chat", "chat"]})))
    _, inter, _ = pl.consolidate([spec])
    assert len(inter) == 1 and inter.text.iloc[0] == "This is unacceptable", \
        "agent replies would pollute the sentiment read"


def test_precedence_beats_filename_order():
    """A dedicated aggregation must win over a snapshot, whatever the filenames."""
    panel = pl.classify(_write("zz_panel.csv", pd.DataFrame({
        "customer_id": ["A"] * 4,
        "month": ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"],
        "support_tickets": [1, 1, 1, 99], "monthly_usage_pct": [80, 70, 60, 50],
        "csat_score": [8, 7, 6, 5]})))
    tix = pl.classify(_write("aa_tickets.csv", pd.DataFrame({
        "customer_id": ["A", "A"], "created_at": ["2025-04-20", "2025-04-25"],
        "resolved_at": [None, None], "severity": ["low", "critical"],
        "reopened": [False, True]})))

    # tickets sorts FIRST alphabetically, so without precedence the panel's 99
    # would overwrite the real aggregate.
    cust, _, _ = pl.consolidate([tix, panel])
    row = cust.set_index("customer_id").loc["A"]
    assert row.tickets_last_30d == 2, f"tickets table must win, got {row.tickets_last_30d}"
    assert row.open_p1_tickets == 1
    assert row.logins_last_30d == 50, "panel still supplies fields tickets do not"


def test_only_recent_messages_reach_the_model():
    """Sentiment is about how they sound now, not 18 months ago."""
    n = pl.MAX_MSGS_PER_CUSTOMER
    dates = [f"2025-{m:02d}-01" for m in range(1, 13)]
    spec = pl.classify(_write("many.csv", pd.DataFrame({
        "customer_id": ["A"] * 12, "date": dates, "channel": ["chat"] * 12,
        "text": [f"message {i}" for i in range(12)]})))
    _, inter, _ = pl.consolidate([spec])
    assert len(inter) == n, f"expected {n} most recent, got {len(inter)}"
    assert inter.date.min() == dates[12 - n], "must keep the newest, not the oldest"
    assert "message 11" in set(inter.text)


def test_buckets_are_ordered_and_sized():
    shares = [b["share"] for b in pl.BUCKETS]
    assert shares == sorted(shares), "bucket shares must be cumulative and ascending"
    assert shares[-1] == 1.0, "the last bucket must absorb the rest of the book"
    assert [b["rank"] for b in pl.BUCKETS] == [0, 1, 2, 3]

    made = [sig.Assessment(f"C{i}", f"Acct {i}", "SMB", 1000.0, None,
                           risk_score=100 - i, priority_score=100 - i,
                           band="Healthy", signals=[]) for i in range(100)]
    payload = pl.to_nodes(made, {})
    nodes = payload["nodes"]
    assert [n["bucket"] for n in nodes[:5]] == ["urgent"] * 5
    assert nodes[-1]["bucket"] == "stable"
    ranks = [n["bucket_rank"] for n in nodes]
    assert ranks == sorted(ranks), "buckets must be monotonic in risk"
    counts = {b["key"]: b["count"] for b in payload["buckets"]}
    assert counts == {"urgent": 5, "high": 10, "watch": 15, "stable": 70}


def test_movement_flags_accounts_getting_worse():
    """A score that moved is an alert; a static one is only a report."""
    def book(scores):
        made = [sig.Assessment(f"C{i}", f"A{i}", "SMB", 1000.0, None, sc, sc,
                               "Healthy", []) for i, sc in enumerate(scores)]
        return pl.to_nodes(made, {})

    first = book([90, 80, 70, 60, 50, 40, 30, 20, 10, 5])
    # C9 leaps from the bottom of the book to the top; C0 falls to the bottom.
    second = pl.to_nodes(
        [sig.Assessment(f"C{i}", f"A{i}", "SMB", 1000.0, None, sc, sc, "Healthy", [])
         for i, sc in enumerate([5, 80, 70, 60, 50, 40, 30, 20, 10, 95])],
        {}, previous=first)

    by_id = {n["id"]: n for n in second["nodes"]}
    assert by_id["C9"]["movement"]["status"] == "escalating"
    assert by_id["C9"]["movement"]["from_bucket"] == "stable"
    assert by_id["C9"]["movement"]["delta"] == 90
    assert by_id["C0"]["movement"]["status"] == "improving"
    assert by_id["C5"]["movement"]["status"] == "steady", "unchanged score must not alert"
    assert second["totals"]["escalating_since_last_run"] == 1

    fresh = pl.to_nodes([sig.Assessment("NEW", "N", "SMB", 1.0, None, 50, 50,
                                        "Healthy", [])], {}, previous=first)
    assert fresh["nodes"][0]["movement"]["status"] == "new"


def test_node_shape_is_stable():
    """The frontend contract - keep these keys stable."""
    a = sig.Assessment("C1", "Acme", "Enterprise", 50_000.0, 30, 80, 120.0,
                       "Critical", [sig.Signal("usage_decline", 0.9, "Usage 100 -> 20", 0.16)])
    a.issue, a.rationale, a.action, a.action_owner = "usage collapse", "Because.", "Call.", "CSM"
    node = pl.to_nodes([a], {"C1": ["crm.csv"]})["nodes"][0]
    assert set(node) == {"id", "label", "bucket", "bucket_rank", "urgency", "issue",
                         "summary", "action", "drivers", "meta", "sources",
                         "movement", "updated_at"}
    assert node["movement"]["status"] == "new", "no prior run means every node is new"
    assert node["urgency"] == 80 and node["issue"] == "usage collapse"
    assert node["action"] == {"text": "Call.", "owner": "CSM"}
    assert node["drivers"][0]["signal"] == "usage_decline"
    assert node["sources"] == ["crm.csv"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
