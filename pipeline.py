"""Continuous four-layer pipeline: drop files in, get ranked customer nodes out.

    python pipeline.py inbox/                 # process once
    python pipeline.py inbox/ --watch 60      # keep polling every 60s
    python pipeline.py inbox/ -o nodes.json   # choose the output file

    L1  document processing   what kind of file is this, and which of its
                              columns map onto our canonical fields?
    L2  consolidation         gather every file's contribution into one
                              profile per customer_id
    L3  sentiment + scoring   signals.detect - deterministic signals plus the
                              LLM text layer, fused into a risk score
    L4  bucketing             rank the book and drop each customer into a
                              bucket, worst first, as a frontend-ready node

Layers 1 and 3 both use the LLM. L1 uses it only for schemas the fingerprints
don't recognise - a known file costs nothing, an unknown one gets mapped rather
than dropped. L3 is the sentiment and reasoning layer.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import signals as sig

# The four buckets, worst first. `rank` is the sort key a frontend can rely on;
# `share` is the cumulative top fraction of the ranked book that lands in each.
#
# Why fractions rather than fixed score thresholds: how severe a 20% usage drop
# looks depends entirely on the book. Fixed cut-points tuned on one population
# leave the worst buckets empty on a calmer one. Sizing by share keeps the
# worklist workable whatever data arrives - the top 5% is always the top 5%.
# Override `share` per deployment to match team capacity.
BUCKETS = [
    {
        "key": "urgent", "label": "Needs attention now", "rank": 0, "share": 0.05,
        "colour": "#B3261E",
        "definition": "Top 5% of the book by risk. Typically explicit exit language "
                      "- cancellation, notice periods, a named competitor - or severe "
                      "multi-signal decay with a renewal close enough to matter.",
        "means": "Assume the customer is already leaving unless someone intervenes.",
        "sla": "Owner assigned and contact made within 48 hours.",
    },
    {
        "key": "high", "label": "At risk", "rank": 1, "share": 0.15,
        "colour": "#C2681A",
        "definition": "Next 10%. Deteriorating on more than one signal at once - "
                      "usage falling while tickets climb, or billing friction "
                      "alongside a satisfaction drop. No stated intent to leave yet.",
        "means": "The pattern that precedes churn, caught before it is voiced.",
        "sla": "Worked in this week's queue; root cause identified before renewal.",
    },
    {
        "key": "watch", "label": "Worth watching", "rank": 2, "share": 0.30,
        "colour": "#8A6D0B",
        "definition": "Next 15%. One signal drifting, usually quietly - a usage "
                      "decline or adoption stall with no complaint filed. The "
                      "silent cases that manual review misses.",
        "means": "Not urgent, but the trend is the wrong way.",
        "sla": "Reviewed at the next cycle; flag if it moves up a bucket.",
    },
    {
        "key": "stable", "label": "Likely to stay", "rank": 3, "share": 1.00,
        "colour": "#1E6F3C",
        "definition": "The remaining 70%. No meaningful deterioration across "
                      "behaviour, billing, satisfaction or language.",
        "means": "Healthy. Expansion territory rather than retention work.",
        "sla": "No action. Re-scored automatically each cycle.",
    },
]

# Canonical fields the detector understands. L1 maps incoming columns onto these.
CANONICAL = [
    "customer_id", "account_name", "segment", "plan", "arr_usd", "tenure_months",
    "renewal_date", "seats_licensed", "seats_active", "feature_adoption_pct",
    "logins_last_30d", "logins_prev_30d", "tickets_last_30d", "tickets_prev_30d",
    "tickets_reopened_30d", "open_p1_tickets", "avg_first_response_hrs",
    "csat_current", "csat_prev_quarter", "nps_last", "failed_payments_90d",
    "invoice_disputes_90d", "days_payment_overdue", "downgraded_last_90d",
]
TEXT_FIELDS = ["customer_id", "date", "channel", "text"]

# How many of a customer's most recent messages reach the model. Sentiment is a
# question about how they sound NOW; feeding 18 months of history lets a stale
# complaint carry the same weight as last week's, and dilutes the recent signal.
MAX_MSGS_PER_CUSTOMER = 6

# When two feeds write the same field, the higher number wins regardless of the
# order files happen to be read in. Without this, resolution falls to whichever
# filename sorts last - so renaming a file silently changes the score.
#
# Precedence is per FIELD, not per source kind, because no single source is best
# at everything. A monthly snapshot carries the system-of-record rollup, so it
# owns volume counts; a raw event export is often a filtered or partial extract
# and undercounts them - on this dataset the two disagreed 8x on tickets/month.
# But the event export is the only source with per-ticket detail, so it owns
# reopens and severity. Getting this backwards fed the support signal a
# near-empty count and cost real accuracy.
PRECEDENCE = {
    "monthly_panel": {"": 2, "tickets_last_30d": 4, "tickets_prev_30d": 4},
    "tickets":       {"": 3},
    "accounts_wide": {"": 1},
    "billing":       {"": 1},
    "unknown":       {"": 1},
}


def _priority(kind: str, field: str) -> int:
    table = PRECEDENCE.get(kind, {"": 1})
    return table.get(field, table[""])

# The column that dates each row, by file kind. Used to rewind a feed to a past
# date. `contract_end_date` is deliberately absent: it is set at signing and is
# therefore known in advance, not a record timestamp.
EVENT_DATE = {
    "monthly_panel": ("snapshot_date", "month"),
    "interactions": ("timestamp", "date", "sent_at"),
    "messages": ("timestamp", "date", "sent_at"),
    "tickets": ("created_at", "opened_at"),
}

# Column tokens that mean "this file contains the answer". A drop folder will
# sooner or later receive an outcomes export sitting beside the feature files,
# and scoring on it would produce a beautiful, meaningless result. Quarantine is
# structural rather than trusting L1 to decline the mapping.
LABEL_MARKERS = {
    "outcome", "churn", "churned", "decision_date", "is_censored", "renewed",
    "ground_truth", "label", "target", "arr_after", "eligible_for_prediction",
    "human_sentiment", "human_intent", "scenario_type",
}

# Source names we understand but that are spelled differently by each system.
ALIASES = {
    "contract_end_date": "renewal_date", "renewal_dt": "renewal_date",
    "arr": "arr_usd", "annual_recurring_revenue": "arr_usd",
    "tier": "segment", "active_users": "seats_active",
    "product_adoption_pct": "feature_adoption_pct",
    "nps_score": "nps_last", "invoice_overdue_days": "days_payment_overdue",
    "payment_failure_count_30d": "failed_payments_90d",
    "support_tickets_unresolved": "open_p1_tickets",
}


# ===================================================== L1 document processing

# Checked in order; the first match wins. Text and panel shapes are checked
# before the generic wide shape, because they need different handling.
FINGERPRINTS = [
    ("interactions", {"text"}),
    ("monthly_panel", {"month"}),
    ("monthly_panel", {"snapshot_date"}),
    # Ticket exports are one row per ticket and need aggregating, so they must
    # be recognised here rather than left to the LLM - a mapped ticket file
    # would otherwise be absorbed as one profile per row.
    ("tickets", {"created_at"}),
    ("tickets", {"ticket_id"}),
]
ID_COLUMNS = ("customer_id", "account_id")

MAP_SYSTEM = """You map columns from an unknown customer-data file onto a fixed set
of canonical field names used by a churn-risk scorer.

Return one entry per SOURCE column you can confidently map. Omit anything you
cannot map - a wrong mapping is far worse than a missing one, because the score
would then be computed from the wrong number. Never invent a source column name.

Notes on the canonical fields:
- logins_* are any recurring engagement measure (sessions, active users, usage %).
- tickets_* are support contact volume in the period.
- csat_current is expected on a 1-5 scale; flag `needs_halving` if the source
  looks like a 1-10 scale.
- Free-text customer messages map to the separate field `text`.

`canonical_field` must be one of the listed names exactly - do not invent new
ones. Use kind "unknown" only when the file holds no per-customer data worth
scoring; if you mapped any field at all, pick the kind that fits best."""

MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string",
                 "enum": ["accounts_wide", "monthly_panel", "interactions",
                          "tickets", "billing", "unknown"]},
        "id_column": {"type": "string"},
        "mappings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_column": {"type": "string"},
                    "canonical_field": {"type": "string"},
                    "needs_halving": {"type": "boolean"},
                },
                "required": ["source_column", "canonical_field", "needs_halving"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string"},
    },
    "required": ["kind", "id_column", "mappings", "note"],
    "additionalProperties": False,
}


def classify(path: Path, client=None) -> dict:
    """Layer 1. Identify a file and how its columns map onto canonical fields."""
    df = pd.read_csv(path, nrows=200)
    cols = set(df.columns)

    lowered = {c.lower() for c in cols}
    hits = sorted(m for m in LABEL_MARKERS if any(m in c for c in lowered))
    if hits:
        return {"path": str(path), "kind": "quarantined_labels", "columns": list(df.columns),
                "id_column": None, "mappings": None, "via": "label-guard",
                "note": f"outcome/label columns present: {', '.join(hits)}"}

    idc = next((c for c in ID_COLUMNS if c in cols), None)

    if idc:
        def known(kind):
            return {"path": str(path), "kind": kind, "columns": list(df.columns),
                    "id_column": idc, "mappings": None, "via": "fingerprint"}
        for kind, marker in FINGERPRINTS:
            if marker <= cols:
                return known(kind)
        # Anything else carrying an id and at least one field we understand is a
        # usable per-account feed - billing extracts, CSAT exports, usage dumps.
        if cols & set(CANONICAL) - {"customer_id"}:
            return known("accounts_wide")

    # Unrecognised - ask the model to map it rather than dropping the file.
    if client is None:
        return {"path": str(path), "kind": "unknown", "columns": list(df.columns),
                "id_column": None, "mappings": None, "via": "unmapped"}

    sample = df.head(3).to_csv(index=False)
    try:
        r = client.chat.completions.create(
            model=sig.MODEL, temperature=0.0,
            messages=[{"role": "system", "content": MAP_SYSTEM},
                      {"role": "user", "content":
                       f"CANONICAL FIELDS:\n{', '.join(CANONICAL + ['text', 'date'])}\n\n"
                       f"FILE: {path.name}\nSAMPLE ROWS:\n{sample}"}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "schema_map", "strict": True, "schema": MAP_SCHEMA}})
        out = json.loads(r.choices[0].message.content)
        # Validate both ends of every mapping. 
        allowed = set(CANONICAL) | {"text", "date", "channel"}
        clean, dropped = [], []
        for m in out["mappings"]:
            if m["source_column"] in df.columns and m["canonical_field"] in allowed:
                clean.append(m)
            else:
                dropped.append(f'{m["source_column"]}->{m["canonical_field"]}')
        if dropped:
            print(f"[L1] {path.name}: dropped invalid mapping(s) {', '.join(dropped)}")
        out["mappings"] = clean
        return {"path": str(path), "kind": out["kind"], "columns": list(df.columns),
                "id_column": out["id_column"] if out["id_column"] in df.columns else None,
                "mappings": out["mappings"], "note": out.get("note", ""), "via": "llm"}
    except Exception as e:
        print(f"[warn] could not classify {path.name}: {e}")
        return {"path": str(path), "kind": "unknown", "columns": list(df.columns),
                "id_column": None, "mappings": None, "via": "error"}


# ======================================================== L2 consolidation

def _apply_mappings(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Rename an LLM-mapped file into canonical columns."""
    out = pd.DataFrame()
    for m in spec["mappings"] or []:
        col = df[m["source_column"]]
        if m.get("needs_halving"):
            col = pd.to_numeric(col, errors="coerce") / 2
        out[m["canonical_field"]] = col
    if spec["id_column"] and "customer_id" not in out.columns:
        out["customer_id"] = df[spec["id_column"]]
    return out


def _rewind(df: pd.DataFrame, kind: str, as_of, path_name: str) -> pd.DataFrame:
    """Drop rows dated after `as_of`, so a past date can be scored honestly.

    A current-state export carries no row date and cannot be rewound - it is
    already "now". Scoring a past date with one of those silently uses future
    information, so we say so rather than let it pass quietly.
    """
    if as_of is None:
        return df
    for col in EVENT_DATE.get(kind, ()):
        if col in df.columns:
            when = pd.to_datetime(df[col], errors="coerce")
            kept = df[when <= as_of]
            print(f"[L2] {path_name}: {len(kept)}/{len(df)} rows at or before "
                  f"{as_of:%Y-%m-%d}")
            return kept
    print(f"[L2] {path_name}: no row date - cannot rewind. Values are current "
          f"state and may post-date {as_of:%Y-%m-%d}.")
    return df


def consolidate(specs: list[dict], as_of=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Layer 2. Fold every classified file into one profile per customer.

    Later files win on conflict, but only where they carry a non-null value -
    a partial feed adds to a profile, it never blanks fields another feed filled.

    `as_of` rewinds every dated feed to that point, so the pipeline can score a
    past date using only what was knowable then. That is what makes its output
    evaluable and lets history be backfilled; without it a run can only ever
    describe the present.
    """
    profiles: dict[str, dict] = {}
    claims: dict[str, dict[str, int]] = {}
    texts: list[dict] = []
    provenance: dict[str, set] = {}

    def absorb(cid: str, rec: dict, source: str, kind: str = "unknown"):
        cid = str(cid)
        p = profiles.setdefault(cid, {"customer_id": cid})
        held = claims.setdefault(cid, {})
        for k, v in rec.items():
            if k == "customer_id" or pd.isna(v):
                continue
            prio = _priority(kind, k)
            if prio >= held.get(k, 0):
                p[k] = v
                held[k] = prio
        provenance.setdefault(cid, set()).add(source)

    for spec in specs:
        path, kind = Path(spec["path"]), spec["kind"]
        # "unknown" only disqualifies a file if nothing was mapped from it - the
        # model is often cautious about the label while mapping columns correctly.
        if kind == "quarantined_labels":
            continue
        if kind == "unknown" and not spec.get("mappings"):
            continue
        df = pd.read_csv(path).rename(columns=ALIASES)
        src = path.name
        df = _rewind(df, kind, as_of, src)
        if df.empty:
            continue

        # Column mappings are for shapes with no dedicated handler. A per-row
        # event table needs aggregating, and letting a column map turn it into a
        # profile puts a ticket id into a ticket-count field.
        if spec.get("mappings") and kind in ("unknown", "accounts_wide",
                                            "interactions", "billing"):
            df = _apply_mappings(df, spec)
            kind = "interactions" if "text" in df.columns else "accounts_wide"

        idc = "customer_id" if "customer_id" in df.columns else (
            "account_id" if "account_id" in df.columns else None)
        if idc is None:
            continue
        df = df.rename(columns={idc: "customer_id"})

        if kind in ("interactions", "messages"):
            for r in df.itertuples():
                # Agent-side replies would pollute the sentiment read.
                if str(getattr(r, "author_role", "customer")).lower() == "agent":
                    continue
                t = getattr(r, "text", None)
                if isinstance(t, str) and t.strip():
                    texts.append({"customer_id": str(r.customer_id),
                                  "date": str(getattr(r, "date", None)
                                               or getattr(r, "timestamp", ""))[:10],
                                  "channel": str(getattr(r, "channel", "support")),
                                  "text": t.strip()})
                    provenance.setdefault(str(r.customer_id), set()).add(src)

        elif kind == "tickets":
            # One row per ticket, not per customer. Aggregating is the whole job:
            # absorbing these as profiles would leave a ticket id in a count field.
            # Every optional column defaults to a Series of the right length.
            # pd.to_datetime(None) returns a scalar NaT, and df.get(x, False) a
            # bare bool - either one turns the vectorised code below into an
            # AttributeError the moment a feed omits a column.
            blank = pd.Series(pd.NaT, index=df.index)
            created = pd.to_datetime(df.get("created_at", blank), errors="coerce")
            resolved = pd.to_datetime(df.get("resolved_at", blank), errors="coerce")
            ref = as_of if as_of is not None else created.max()
            last30 = created > ref - pd.Timedelta(days=30)
            prev30 = ((created <= ref - pd.Timedelta(days=30))
                      & (created > ref - pd.Timedelta(days=60)))
            unresolved = resolved.isna() | (resolved > ref)
            # Defaults must be Series, not scalars - a bare False has no .astype,
            # so a ticket export missing either column would crash the run.
            sev = df.get("severity", pd.Series("", index=df.index)).astype(str).str.lower()
            reopened = df.get("reopened", pd.Series(False, index=df.index)).fillna(False)
            agg = pd.DataFrame({"customer_id": df.customer_id.astype(str)})
            for name, mask in (("tickets_last_30d", last30), ("tickets_prev_30d", prev30),
                               ("tickets_reopened_30d", last30 & reopened.astype(bool)),
                               ("open_p1_tickets", unresolved & sev.isin(
                                   {"critical", "p1", "urgent"}))):
                agg[name] = mask.fillna(False).astype(int)
            for cid, g in agg.groupby("customer_id"):
                absorb(cid, {k: int(g[k].sum()) for k in agg.columns[1:]}, src, kind)

        elif kind == "monthly_panel":
            import adapt
            panel = df.rename(columns={"snapshot_date": "month",
                                       "support_tickets_opened": "support_tickets"})
            wide, inter = adapt.adapt(panel)
            for r in wide.to_dict("records"):
                absorb(r["customer_id"], r, src, kind)
            # adapt() computes current-vs-baseline deltas; anything else the panel
            # carries is still useful at its latest observed value.
            latest = panel.sort_values("month").groupby("customer_id").last()
            extra = [c for c in latest.columns if c in CANONICAL]
            for cid, row in latest[extra].iterrows():
                absorb(cid, row.to_dict(), src, kind)
            texts.extend(inter.to_dict("records"))

        else:                                          # accounts_wide / billing
            for r in df.to_dict("records"):
                absorb(r["customer_id"], r, src, kind)

    cust = pd.DataFrame(list(profiles.values()))
    if texts:
        inter = (pd.DataFrame(texts, columns=TEXT_FIELDS)
                 .sort_values("date")
                 .groupby("customer_id", group_keys=False)
                 .tail(MAX_MSGS_PER_CUSTOMER))
    else:
        inter = pd.DataFrame(columns=TEXT_FIELDS)
    return cust, inter, {k: sorted(v) for k, v in provenance.items()}


# ============================================================ L4 bucketing

def movement(nodes: list[dict], previous: dict | None) -> list[dict]:
    """Annotate each node with how it changed since the last run.

    A static score is a report; a score that moved is an alert. An account
    sliding from stable to at-risk in one cycle is more actionable than one that
    has sat in at-risk for months, and ops should be able to sort on that.
    """
    prev = {n["id"]: n for n in (previous or {}).get("nodes", [])}
    ranks = {b["key"]: b["rank"] for b in BUCKETS}
    for n in nodes:
        was = prev.get(n["id"])
        if was is None:
            n["movement"] = {"status": "new", "delta": None, "from_bucket": None}
            continue
        delta = n["urgency"] - was["urgency"]
        # Bucket ranks run worst-first, so a FALLING rank means getting worse.
        crossed = ranks[was["bucket"]] - ranks[n["bucket"]]
        n["movement"] = {
            "status": "escalating" if crossed > 0 else
                      "improving" if crossed < 0 else
                      "worsening" if delta >= 5 else
                      "recovering" if delta <= -5 else "steady",
            "delta": delta,
            "from_bucket": was["bucket"],
        }
    return nodes


def to_nodes(assessments: list[sig.Assessment], provenance: dict, as_of=None,
             previous: dict | None = None) -> dict:
    """Layer 4. Rank the book, bucket it, and emit frontend-ready nodes."""
    ranked = sorted(assessments, key=lambda a: a.risk_score, reverse=True)
    n = len(ranked) or 1
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    nodes = []
    for i, a in enumerate(ranked):
        pct = (i + 1) / n
        b = next(b for b in BUCKETS if pct <= b["share"])
        nodes.append({
            "id": a.customer_id,
            "label": a.account_name,
            "bucket": b["key"],
            "bucket_rank": b["rank"],
            "urgency": a.risk_score,
            "issue": a.issue or "not analysed",
            "summary": a.rationale,
            "action": {"text": a.action, "owner": a.action_owner},
            "drivers": [
                {"signal": s.name, "severity": round(s.score, 2),
                 "points": s.contribution, "evidence": s.evidence}
                for s in a.top_drivers
            ],
            "meta": {"segment": a.segment, "arr_usd": a.arr_usd,
                     "renewal_in_days": a.renewal_in_days,
                     "analysed": a.llm_used},
            "sources": provenance.get(a.customer_id, []),
            "updated_at": now,
        })

    movement(nodes, previous)
    counts = {b["key"]: sum(1 for x in nodes if x["bucket"] == b["key"]) for b in BUCKETS}
    escalating = [x for x in nodes if x["movement"]["status"] == "escalating"]
    return {
        "generated_at": now,
        "as_of": as_of.date().isoformat() if as_of is not None else None,
        "model": sig.MODEL,
        "buckets": [{**b, "count": counts[b["key"]]} for b in BUCKETS],
        "totals": {"customers": len(nodes),
                   "escalating_since_last_run": len(escalating),
                   "needs_attention": sum(1 for x in nodes if x["bucket_rank"] <= 1),
                   "arr_at_risk": round(sum(x["meta"]["arr_usd"] or 0 for x in nodes
                                            if x["bucket_rank"] <= 1), 2)},
        "nodes": nodes,
    }


# ================================================================ orchestration

def run_once(inbox: Path, out: Path, as_of=None) -> dict:
    client = sig._client()
    if client is None:
        raise sig.NoLLMError("pipeline needs OPENAI_API_KEY in .env")

    files = sorted(p for p in inbox.glob("*.csv"))
    if not files:
        print(f"L1  nothing to process in {inbox}")
        return {}

    specs = [classify(p, client) for p in files]
    for s in specs:
        print(f"L1  {Path(s['path']).name:<38} {s['kind']:<15} via {s['via']}")

    cust, inter, prov = consolidate(specs, as_of)
    print(f"L2  {len(cust)} customers consolidated from {len(files)} files, "
          f"{len(inter)} messages" + (f"  (as of {as_of:%Y-%m-%d})" if as_of else ""))
    if cust.empty:
        print("L2  no identifiable customers - nothing to score")
        return {}

    # Renewal proximity is measured from the scoring date, not from today.
    res = sig.detect(cust, inter, client=client,
                     as_of=as_of.date() if as_of is not None else None)
    print(f"L3  scored {len(res)} customers ({sum(a.llm_used for a in res)} analysed)")

    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    payload = to_nodes(res, prov, as_of, previous)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("L4  " + "  ".join(f"{b['key']} {b['count']}" for b in payload["buckets"]))
    esc = payload["totals"]["escalating_since_last_run"]
    if previous:
        print(f"    {esc} escalating into a worse bucket since the last run")
    print(f"    -> {out}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inbox", type=Path, help="directory of incoming CSV files")
    ap.add_argument("-o", "--out", type=Path, default=Path("data/nodes.json"))
    ap.add_argument("--watch", type=int, metavar="SECONDS",
                    help="keep polling the inbox every N seconds")
    ap.add_argument("--as-of", dest="as_of", metavar="YYYY-MM-DD",
                    help="score as this date, using only records dated on or before it")
    a = ap.parse_args()
    a.as_of = pd.Timestamp(a.as_of) if a.as_of else None
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if not a.watch:
        run_once(a.inbox, a.out, a.as_of)
        return

    print(f"watching {a.inbox} every {a.watch}s - ctrl-c to stop")
    seen: set[tuple] = set()
    while True:
        state = {(p.name, p.stat().st_mtime) for p in a.inbox.glob("*.csv")}
        if state and state != seen:
            print(f"\n--- {datetime.now():%H:%M:%S}  inbox changed ---")
            run_once(a.inbox, a.out, a.as_of)
            seen = state
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
