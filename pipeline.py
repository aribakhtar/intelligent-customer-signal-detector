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
        # Validate both ends of every mapping. A hallucinated source column reads
        # the wrong data; a hallucinated canonical field silently does nothing.
        # Either way the score would be computed from something we did not intend.
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


def consolidate(specs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Layer 2. Fold every classified file into one profile per customer.

    Later files win on conflict, but only where they carry a non-null value -
    a partial feed adds to a profile, it never blanks fields another feed filled.
    """
    profiles: dict[str, dict] = {}
    texts: list[dict] = []
    provenance: dict[str, set] = {}

    def absorb(cid: str, rec: dict, source: str):
        p = profiles.setdefault(str(cid), {"customer_id": str(cid)})
        for k, v in rec.items():
            if k != "customer_id" and pd.notna(v):
                p[k] = v
        provenance.setdefault(str(cid), set()).add(source)

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
            created = pd.to_datetime(df.get("created_at"), errors="coerce")
            resolved = pd.to_datetime(df.get("resolved_at"), errors="coerce")
            ref = created.max()
            last30 = created > ref - pd.Timedelta(days=30)
            prev30 = (created <= ref - pd.Timedelta(days=30)) &                      (created > ref - pd.Timedelta(days=60))
            unresolved = resolved.isna() | (resolved > ref)
            sev = df.get("severity", pd.Series("", index=df.index)).astype(str).str.lower()
            agg = pd.DataFrame({"customer_id": df.customer_id.astype(str)})
            for name, mask in (("tickets_last_30d", last30), ("tickets_prev_30d", prev30),
                               ("tickets_reopened_30d", last30 & df.get(
                                   "reopened", False).astype(bool)),
                               ("open_p1_tickets", unresolved & sev.isin(
                                   {"critical", "p1", "urgent"}))):
                agg[name] = mask.fillna(False).astype(int)
            for cid, g in agg.groupby("customer_id"):
                absorb(cid, {k: int(g[k].sum()) for k in agg.columns[1:]}, src)

        elif kind == "monthly_panel":
            import adapt
            panel = df.rename(columns={"snapshot_date": "month",
                                       "support_tickets_opened": "support_tickets"})
            wide, inter = adapt.adapt(panel)
            for r in wide.to_dict("records"):
                absorb(r["customer_id"], r, src)
            # adapt() computes current-vs-baseline deltas; anything else the panel
            # carries is still useful at its latest observed value.
            latest = panel.sort_values("month").groupby("customer_id").last()
            extra = [c for c in latest.columns if c in CANONICAL]
            for cid, row in latest[extra].iterrows():
                absorb(cid, row.to_dict(), src)
            texts.extend(inter.to_dict("records"))

        else:                                          # accounts_wide / billing
            for r in df.to_dict("records"):
                absorb(r["customer_id"], r, src)

    cust = pd.DataFrame(list(profiles.values()))
    inter = pd.DataFrame(texts, columns=TEXT_FIELDS) if texts else \
        pd.DataFrame(columns=TEXT_FIELDS)
    return cust, inter, {k: sorted(v) for k, v in provenance.items()}


# ============================================================ L4 bucketing

def to_nodes(assessments: list[sig.Assessment], provenance: dict) -> dict:
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

    counts = {b["key"]: sum(1 for x in nodes if x["bucket"] == b["key"]) for b in BUCKETS}
    return {
        "generated_at": now,
        "model": sig.MODEL,
        "buckets": [{**b, "count": counts[b["key"]]} for b in BUCKETS],
        "totals": {"customers": len(nodes),
                   "needs_attention": sum(1 for x in nodes if x["bucket_rank"] <= 1),
                   "arr_at_risk": round(sum(x["meta"]["arr_usd"] or 0 for x in nodes
                                            if x["bucket_rank"] <= 1), 2)},
        "nodes": nodes,
    }


# ================================================================ orchestration

def run_once(inbox: Path, out: Path) -> dict:
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

    cust, inter, prov = consolidate(specs)
    print(f"L2  {len(cust)} customers consolidated from {len(files)} files, "
          f"{len(inter)} messages")
    if cust.empty:
        print("L2  no identifiable customers - nothing to score")
        return {}

    res = sig.detect(cust, inter, client=client)
    print(f"L3  scored {len(res)} customers ({sum(a.llm_used for a in res)} analysed)")

    payload = to_nodes(res, prov)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("L4  " + "  ".join(f"{b['key']} {b['count']}" for b in payload["buckets"]))
    print(f"    -> {out}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inbox", type=Path, help="directory of incoming CSV files")
    ap.add_argument("-o", "--out", type=Path, default=Path("data/nodes.json"))
    ap.add_argument("--watch", type=int, metavar="SECONDS",
                    help="keep polling the inbox every N seconds")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if not a.watch:
        run_once(a.inbox, a.out)
        return

    print(f"watching {a.inbox} every {a.watch}s - ctrl-c to stop")
    seen: set[tuple] = set()
    while True:
        state = {(p.name, p.stat().st_mtime) for p in a.inbox.glob("*.csv")}
        if state and state != seen:
            print(f"\n--- {datetime.now():%H:%M:%S}  inbox changed ---")
            run_once(a.inbox, a.out)
            seen = state
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
