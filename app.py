"""Streamlit front end: feed data in, watch the four layers run, read the summary.

    streamlit run app.py

Scoring lives in pipeline.py / signals.py; this file only drives and renders it,
so the UI and a batch CLI run can never disagree.
"""
from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import pipeline as pl
import signals as sig

ROOT = Path(__file__).parent
DATA = ROOT / "data"

st.set_page_config(page_title="Customer Signal Detector", page_icon="\U0001F6A8",
                   layout="wide")

BUCKET_CSS = {"urgent": "#B3261E", "high": "#C2681A",
              "watch": "#8A6D0B", "stable": "#1E6F3C"}
MOVE_ICON = {"escalating": "▲▲", "worsening": "▲", "steady": "—",
             "recovering": "▼", "improving": "▼▼", "new": "•"}
CHUNK = 10          # customers per scoring batch, so the progress bar actually moves

st.session_state.setdefault("payload", None)
st.session_state.setdefault("source_label", None)


# ============================================================== the four layers

def run_pipeline(files: list[Path], as_of, limit: int | None, previous: dict | None):
    """Execute L1-L4 with the working shown."""
    client = sig._client()
    if client is None:
        st.error("**No usable `OPENAI_API_KEY`.** Sentiment and churn intent are scored "
                 "entirely by the LLM — there is no keyword fallback. Put a real key in "
                 "`.env` and restart.")
        return None

    with st.status("**L1 — Document processing**", expanded=True) as s:
        st.caption("What kind of file is each of these, and which columns do we understand?")
        specs, rows = [], []
        for p in files:
            spec = pl.classify(p, client)
            specs.append(spec)
            rows.append({"file": p.name, "recognised as": spec["kind"],
                         "how": spec["via"], "note": spec.get("note", "")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        quarantined = sum(r["recognised as"] == "quarantined_labels" for r in rows)
        if quarantined:
            st.warning(f"{quarantined} file(s) held back — they contain outcome or label "
                       "columns. Scoring on those would produce a meaningless result.")
        s.update(label=f"**L1 — Document processing** · {len(files)} files, "
                       f"{quarantined} quarantined", state="complete")

    with st.status("**L2 — Consolidation**", expanded=True) as s:
        st.caption("Gather every file's contribution into one profile per customer.")
        cust, inter, prov = pl.consolidate(specs, as_of)
        if cust.empty:
            s.update(label="**L2 — Consolidation** · no identifiable customers",
                     state="error")
            st.error("No file carried a usable customer id.")
            return None
        known = sorted(c for c in cust.columns if c in pl.CANONICAL)
        st.write(f"**{len(cust)} customers**, {len(inter)} customer-authored messages")
        st.write(f"**{len(known)} signal fields** recovered: `{'`, `'.join(known)}`")
        s.update(label=f"**L2 — Consolidation** · {len(cust)} customers, "
                       f"{len(inter)} messages", state="complete")

    if limit and len(cust) > limit:
        cust = cust.head(limit)
        inter = inter[inter.customer_id.isin(set(cust.customer_id))]

    with st.status("**L3 — Sentiment & scoring**", expanded=True) as s:
        st.caption(f"One {sig.MODEL} call per customer — sentiment, churn intent, the "
                   "issue, a rationale and a recommended action — fused with the "
                   "deterministic signals into a score.")
        bar = st.progress(0.0, text="scoring…")
        ref = as_of.date() if as_of is not None else None
        assessments = []
        chunks = [cust.iloc[i:i + CHUNK] for i in range(0, len(cust), CHUNK)]
        for n, part in enumerate(chunks, 1):
            sub = inter[inter.customer_id.isin(set(part.customer_id))]
            assessments += sig.detect(part, sub, client=client, as_of=ref)
            bar.progress(n / len(chunks),
                         text=f"scored {min(n * CHUNK, len(cust))} of {len(cust)}")
        failed = sum(1 for a in assessments if not a.llm_used)
        bar.empty()
        st.write(f"**{len(assessments)} scored**"
                 + (f" · ⚠ {failed} could not be analysed" if failed else " · no failures"))
        s.update(label=f"**L3 — Sentiment & scoring** · {len(assessments)} customers",
                 state="complete")

    with st.status("**L4 — Bucketing**", expanded=True) as s:
        st.caption("Rank all customers and drop each one into a bucket, worst first.")
        payload = pl.to_nodes(assessments, prov, as_of, previous)
        cols = st.columns(len(payload["buckets"]))
        for c, b in zip(cols, payload["buckets"]):
            c.metric(b["label"], b["count"])
        s.update(label="**L4 — Bucketing** · "
                       + "  ".join(f"{b['key']} {b['count']}" for b in payload["buckets"]),
                 state="complete")
    return payload


# ==================================================================== sidebar

st.sidebar.title("Signal Detector")
mode = st.sidebar.radio(
    "Data source", ["Upload files", "Bundled folder", "Open a saved run"],
    help="Upload your own CSVs, point at a folder in the repo, or reopen the "
         "output of an earlier run.")

files: list[Path] = []
run_label = ""

if mode == "Upload files":
    ups = st.sidebar.file_uploader("Customer data (CSV)", type="csv",
                                   accept_multiple_files=True)
    st.sidebar.caption("Any shape — accounts, monthly snapshots, support messages, "
                       "tickets, billing. Unfamiliar column names get mapped by the "
                       "model in L1.")
    if ups:
        tmp = Path(tempfile.mkdtemp(prefix="signal-inbox-"))
        for u in ups:
            (tmp / u.name).write_bytes(u.getvalue())
        files = sorted(tmp.glob("*.csv"))
        run_label = f"{len(files)} uploaded file(s)"

elif mode == "Bundled folder":
    folders = [d for d in (ROOT / "data_extensive", ROOT / "inbox", DATA) if d.is_dir()]
    chosen = st.sidebar.selectbox(
        "Folder", folders,
        format_func=lambda p: f"{p.name}/  ({len(list(p.glob('*.csv')))} csv)")
    files = sorted(chosen.glob("*.csv"))
    run_label = f"{chosen.name}/"

if mode == "Open a saved run":
    runs = sorted(DATA.glob("nodes*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        st.sidebar.warning("No saved runs yet — run the pipeline first.")
    else:
        pick = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
        if st.sidebar.button("Open", type="primary", width="stretch"):
            st.session_state.payload = json.loads(pick.read_text(encoding="utf-8"))
            st.session_state.source_label = pick.name
else:
    st.sidebar.divider()
    use_asof = st.sidebar.checkbox(
        "Score as of a past date", value=False,
        help="Uses only records dated on or before it — what a run on that day would "
             "have seen. Leave off to score with everything available.")
    as_of = pd.Timestamp(st.sidebar.date_input("As of", date(2025, 6, 30))) \
        if use_asof else None
    limit = st.sidebar.slider("Max customers to score", 5, 200, 25, 5,
                              help="Each customer is one API call. Keep it small for an "
                                   "interactive demo; the CLI has no cap.")
    compare = st.sidebar.checkbox(
        "Compare against the last run", value=True,
        help="Adds movement — who escalated since last time.")

    if st.sidebar.button("▶ Run pipeline", type="primary", width="stretch",
                         disabled=not files):
        prev = st.session_state.payload if compare else None
        st.session_state.payload = None
        st.subheader(f"Running on {run_label}")
        out = run_pipeline(files, as_of, limit, prev)
        if out:
            st.session_state.payload = out
            st.session_state.source_label = run_label
            (DATA / "nodes_ui.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if not files:
        st.sidebar.caption("Add files to enable the run button.")

# ===================================================================== results

payload = st.session_state.payload
st.title("Intelligent Customer Signal Detector")
st.caption("Support text, product usage, billing and satisfaction correlated into one "
           "ranked worklist — so intervention happens before the cancellation email.")

if payload is None:
    st.info("Choose a data source in the sidebar and run the pipeline, "
            "or open a saved run.")
    st.markdown(
        "**L1 Document processing** — identifies each file and maps unfamiliar columns.  \n"
        "**L2 Consolidation** — gathers every source into one profile per customer.  \n"
        "**L3 Sentiment & scoring** — one model call per customer, fused with the "
        "deterministic signals.  \n"
        "**L4 Bucketing** — ranks all customers and assigns a bucket, worst first.")
    st.stop()

nodes = pd.DataFrame(payload["nodes"])
buckets = payload["buckets"]
by_key = {b["key"]: b for b in buckets}
nodes["segment"] = nodes.meta.apply(lambda m: m.get("segment"))
nodes["arr"] = nodes.meta.apply(lambda m: m.get("arr_usd"))
nodes["renewal"] = nodes.meta.apply(lambda m: m.get("renewal_in_days"))
nodes["move"] = nodes.movement.apply(lambda m: m["status"])
nodes["delta"] = nodes.movement.apply(lambda m: m.get("delta"))

st.caption(f"Source: **{st.session_state.source_label}** · scored with "
           f"**{payload['model']}** · "
           + (f"as of **{payload['as_of']}**" if payload.get("as_of")
              else "all available history"))

t = payload["totals"]
k = st.columns(5)
k[0].metric("Customers", t["customers"])
for i, b in enumerate(buckets[:3], start=1):
    k[i].metric(b["label"], b["count"], help=b["definition"])
k[4].metric("Escalating", t.get("escalating_since_last_run", 0),
            help="Moved into a worse bucket since the previous run.")

st.sidebar.divider()
st.sidebar.subheader("Filter results")
show = st.sidebar.multiselect(
    "Buckets", [b["key"] for b in buckets],
    default=st.session_state.get("bucket_filter",
                                 [b["key"] for b in buckets if b["rank"] <= 2]),
    format_func=lambda k_: by_key[k_]["label"])
segs = sorted({s for s in nodes.segment.dropna()})
seg = st.sidebar.multiselect("Segment", segs, default=segs)
moving_only = st.sidebar.toggle("Only accounts that moved", value=False)

view = nodes[nodes.bucket.isin(show) & (nodes.segment.isin(seg) | nodes.segment.isna())]
if moving_only:
    view = view[view.move.isin(["escalating", "worsening"])]
view = view.sort_values("urgency", ascending=False)


def render_detail(r):
    c = st.columns(4)
    c[0].metric("Urgency", r.urgency, by_key[r.bucket]["label"])
    c[1].metric("Movement", r.move, f"{r.delta:+.0f}" if pd.notna(r.delta) else None)
    c[2].metric("ARR", f"${r.arr:,.0f}" if pd.notna(r.arr) else "n/a")
    c[3].metric("Renewal in", f"{r.renewal:.0f} d" if pd.notna(r.renewal) else "n/a")
    st.markdown(f"### {r.issue}")
    st.markdown(r.summary)
    st.success(f"**Recommended action — {r.action['owner']}:** {r.action['text']}")
    st.markdown("##### What drove the score")
    if r.drivers:
        st.dataframe(pd.DataFrame(r.drivers), hide_index=True, width="stretch",
                     column_config={
                         "signal": "Signal",
                         "severity": st.column_config.ProgressColumn(
                             "Severity", min_value=0.0, max_value=1.0, format="%.2f"),
                         "points": st.column_config.NumberColumn("Points of risk"),
                         "evidence": st.column_config.TextColumn("Evidence", width="large"),
                     })
    else:
        st.info("No signal exceeded the reporting threshold.")
    st.caption(f"Built from: {', '.join(r.sources) or 'unknown'} · scored {r.updated_at}"
               + ("" if r.meta.get("analysed", True)
                  else " · ⚠ text not analysed, score incomplete"))


tab_work, tab_buckets, tab_detail = st.tabs(
    ["Worklist", "Buckets & signals", "Account detail"])

with tab_work:
    st.subheader(f"{len(view)} accounts")
    st.caption("Click any row to open that account.")
    event = st.dataframe(
        view[["id", "label", "bucket", "urgency", "issue", "move", "delta",
              "segment", "arr", "renewal"]],
        hide_index=True, width="stretch", key="worklist",
        on_select="rerun", selection_mode="single-row",
        column_config={
            "id": "ID", "label": "Account", "bucket": "Bucket",
            "urgency": st.column_config.ProgressColumn(
                "Urgency", min_value=0, max_value=100, format="%d"),
            "issue": st.column_config.TextColumn("Issue", width="medium"),
            "move": "Movement",
            "delta": st.column_config.NumberColumn("Δ", format="%+d"),
            "segment": "Segment",
            "arr": st.column_config.NumberColumn("ARR", format="$%d"),
            "renewal": st.column_config.NumberColumn("Renewal (d)"),
        })
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download worklist (CSV)",
        view.drop(columns=["meta", "movement", "drivers", "sources"]).to_csv(index=False),
        "worklist.csv", "text/csv", width="stretch")
    c2.download_button("Download nodes (JSON)", json.dumps(payload, indent=2),
                       "nodes.json", "application/json", width="stretch")

    chosen = event.selection.rows if event and event.selection else []
    if chosen:
        r = view.iloc[chosen[0]]
        st.divider()
        st.markdown(f"## {r.label} · `{r.id}`")
        render_detail(r)
    else:
        st.subheader("Today's briefing")
        for _, r in view.head(6).iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([6, 1])
                bits = [r.segment or "—"]
                if pd.notna(r.arr):
                    bits.append(f"${r.arr:,.0f}")
                if pd.notna(r.renewal):
                    bits.append(f"renewal in {r.renewal:.0f}d")
                c1.markdown(f"**{r.label}** · {' · '.join(bits)}")
                c1.markdown(f"`{r.issue}`")
                c2.markdown(f"<div style='text-align:right'><span style='font-size:1.8rem;"
                            f"font-weight:700;color:{BUCKET_CSS.get(r.bucket, '#666')}'>"
                            f"{r.urgency}</span><br><span style='font-size:.75rem;"
                            f"color:#888'>{MOVE_ICON.get(r.move, '')} {r.move}</span></div>",
                            unsafe_allow_html=True)
                st.markdown(f"*{r.summary}*")
                st.markdown(f"**Next step ({r.action['owner']}):** {r.action['text']}")

with tab_buckets:
    st.subheader("How the customers are split")
    cols = st.columns(len(buckets))
    for col, b in zip(cols, buckets):
        with col:
            st.markdown(f"<div style='border-top:4px solid {BUCKET_CSS[b['key']]};"
                        f"padding-top:.6rem'><b>{b['label']}</b><br>"
                        f"<span style='font-size:2rem;font-weight:700'>{b['count']}</span>"
                        f"</div>", unsafe_allow_html=True)
            st.caption(b["definition"])
            st.caption(f"**SLA** — {b['sla']}")
            if st.button("Show only these", key=f"only_{b['key']}", width="stretch"):
                st.session_state.bucket_filter = [b["key"]]
                st.rerun()

    st.subheader("Where the risk is coming from")
    st.caption("Severity per signal, 0 (clean) to 1 (severe). Blank means the signal was "
               "not among that account's top drivers.")
    rows = []
    for _, r in view.iterrows():
        row = {"account": r.label}
        row.update({d["signal"].replace("_", " ").title(): d["severity"] for d in r.drivers})
        rows.append(row)
    if rows:
        heat = pd.DataFrame(rows).set_index("account")
        st.dataframe(heat.style.background_gradient(cmap="OrRd", vmin=0, vmax=1)
                     .format("{:.2f}", na_rep="–"), width="stretch")

    st.subheader("Movement since the previous run")
    st.bar_chart(nodes.move.value_counts())


# ------------------------------------------------------------- detail tab
with tab_detail:
    pool = view if len(view) else nodes
    pick_id = st.selectbox(
        "Account", pool.id.tolist(),
        format_func=lambda i: f"{nodes.loc[nodes.id == i, 'label'].iloc[0]} ({i})")
    render_detail(nodes[nodes.id == pick_id].iloc[0])
