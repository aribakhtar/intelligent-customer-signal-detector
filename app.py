"""Streamlit front end for the Intelligent Customer Signal Detector.

Run:  streamlit run app.py
"""
import io

import pandas as pd
import streamlit as st

import signals as sig

st.set_page_config(page_title="Customer Signal Detector", page_icon="🚨", layout="wide")

BAND_COLOR = {"Critical": "#b3261e", "High": "#c2681a", "Watch": "#8a6d0b", "Healthy": "#1e6f3c"}
# Derived from the frame, not from WEIGHTS: a dataset that lacks a signal
# simply has no column for it.
def sig_cols(d):
    return [f"sig_{k}" for k in sig.WEIGHTS if f"sig_{k}" in d.columns]


def fmt_money(v):
    return f"${v:,.0f}" if pd.notna(v) else "ARR n/a"


def fmt_renewal(v):
    return f"renewal in {v:.0f}d" if pd.notna(v) else "no renewal date"


@st.cache_data(show_spinner="Analysing accounts with the LLM...")
def run(cust_csv: bytes | None, inter_csv: bytes | None, pct_bands: bool) -> pd.DataFrame:
    if cust_csv and inter_csv:
        cust = pd.read_csv(io.BytesIO(cust_csv))
        inter = pd.read_csv(io.BytesIO(inter_csv))
    else:
        cust, inter = sig.load()
    res = sig.detect(cust, inter)
    if pct_bands:
        sig.apply_percentile_bands(res)
    return sig.to_frame(res)


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Signal Detector")
has_key = sig._client() is not None
st.sidebar.caption(f"Sentiment & reasoning: **{sig.MODEL}**")
if not has_key:
    st.error("**No usable `OPENAI_API_KEY` found.** Sentiment and churn-intent analysis are "
             "done entirely by the LLM - there is no keyword fallback. Put a real key in the "
             "`.env` file in the project root (`OPENAI_API_KEY=sk-...`) and restart.")
    st.stop()

st.sidebar.subheader("Data source")
up_c = st.sidebar.file_uploader("customers.csv", type="csv")
up_i = st.sidebar.file_uploader("interactions.csv", type="csv")
st.sidebar.caption("Leave empty to use the bundled sample dataset.")

st.sidebar.subheader("Filters")
min_risk = st.sidebar.slider("Minimum risk score", 0, 100, 30, 5)
sort_by = st.sidebar.radio("Rank by", ["Risk score", "Priority (risk x ARR)"], index=1)
pct_bands = st.sidebar.toggle(
    "Band by rank, not absolute score", value=False,
    help="Bands the top 3% / 8% / 15% of the book. Use this on real data, where "
         "severities rarely approach 1.0 and fixed thresholds leave the top bands empty.")

try:
    df = run(up_c.getvalue() if up_c else None, up_i.getvalue() if up_i else None, pct_bands)
except sig.NoLLMError as e:
    st.error(f"**Analysis could not run.**\n\n{e}")
    st.stop()
seg_pick = st.sidebar.multiselect("Segment", sorted(df.segment.unique()),
                                  default=sorted(df.segment.unique()))

view = df[(df.risk_score >= min_risk) & (df.segment.isin(seg_pick))].sort_values(
    "risk_score" if sort_by == "Risk score" else "priority", ascending=False)

# --------------------------------------------------------------------- header
st.title("Intelligent Customer Signal Detector")
st.caption("Correlates support text, product usage, billing and satisfaction into a "
           "prioritised watchlist - so intervention happens before the cancellation email.")

k = st.columns(5)
k[0].metric("Accounts monitored", len(df))
k[1].metric("Critical", int((df.band == "Critical").sum()))
k[2].metric("High", int((df.band == "High").sum()))
k[3].metric("ARR at risk", fmt_money(df.loc[df.risk_score >= 50, "arr_usd"].sum()),
            help="Combined ARR of accounts scoring 50+")
if df.renewal_in_days.notna().any():
    k[4].metric("Renewals <60d at risk",
                int(((df.renewal_in_days < 60) & (df.risk_score >= 50)).sum()))
else:
    k[4].metric("Watch band", int((df.band == "Watch").sum()),
                help="No renewal dates in this dataset")

tab_list, tab_heat, tab_detail = st.tabs(["Prioritised watchlist", "Signal heatmap", "Account detail"])

# ------------------------------------------------------------------ watchlist
with tab_list:
    st.subheader(f"{len(view)} accounts need attention")
    st.dataframe(
        view[["account", "segment", "arr_usd", "renewal_in_days", "risk_score", "band",
              "top_drivers", "recommended_action", "owner"]],
        hide_index=True, width="stretch",
        column_config={
            "account": "Account",
            "segment": "Segment",
            "arr_usd": st.column_config.NumberColumn("ARR", format="$%d"),
            "renewal_in_days": st.column_config.NumberColumn("Renewal (d)"),
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100,
                                                          format="%d"),
            "band": "Band",
            "top_drivers": "Top drivers",
            "recommended_action": st.column_config.TextColumn("Recommended action", width="large"),
            "owner": "Owner",
        })
    st.download_button("Download signal summary (CSV)", view.to_csv(index=False),
                       "signal_summary.csv", "text/csv")

    st.subheader("Today's briefing")
    for _, r in view.head(5).iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{r.account}** · {r.segment} · {fmt_money(r.arr_usd)} · "
                        f"{fmt_renewal(r.renewal_in_days)}")
            c2.markdown(f"<h3 style='text-align:right;color:{BAND_COLOR[r.band]};margin:0'>"
                        f"{r.risk_score}</h3>", unsafe_allow_html=True)
            st.markdown(f"*{r.rationale}*")
            st.markdown(f"**Next step ({r.owner}):** {r.recommended_action}")

# -------------------------------------------------------------------- heatmap
with tab_heat:
    st.subheader("Where the risk is coming from")
    st.caption("Each cell is that signal's severity, 0 (clean) to 1 (severe). "
               "Reading across a row tells you which lever to pull.")
    heat = view.set_index("account", verify_integrity=False)[sig_cols(view)].rename(
        columns=lambda c: c[4:].replace("_", " ").title())
    st.dataframe(heat.style.background_gradient(cmap="OrRd", vmin=0, vmax=1).format("{:.2f}"),
                 width="stretch")

    st.subheader("Risk distribution by segment")
    st.bar_chart(df.pivot_table(index="band", columns="segment", values="customer_id",
                                aggfunc="count").reindex(["Critical", "High", "Watch", "Healthy"]))

    st.subheader("Average signal severity across the whole book")
    st.bar_chart(df[sig_cols(df)].mean().rename(lambda c: c[4:].replace("_", " ").title()))

# --------------------------------------------------------------------- detail
with tab_detail:
    pool = view if len(view) else df
    pick = st.selectbox("Account", pool.customer_id.tolist(),
                        format_func=lambda cid: f"{df.loc[df.customer_id == cid, 'account'].iloc[0]} ({cid})")
    r = df[df.customer_id == pick].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score", r.risk_score, r.band)
    c2.metric("ARR", fmt_money(r.arr_usd))
    c3.metric("Renewal in", f"{r.renewal_in_days:.0f} d" if pd.notna(r.renewal_in_days) else "n/a")

    st.markdown(f"### Why this account is flagged\n{r.rationale}")
    st.success(f"**Recommended action - {r.owner}:** {r.recommended_action}")

    st.markdown("### Signal breakdown")
    cols = sig_cols(df)
    br = pd.DataFrame({
        "signal": [c[4:].replace("_", " ").title() for c in cols],
        "severity": [r[c] for c in cols],
        "weight": [sig.WEIGHTS[c[4:]] for c in cols],
    })
    # Weights are renormalised over the signals this dataset supports.
    br["weight"] = br.weight / br.weight.sum()
    br["points_of_risk"] = (br.severity * br.weight * 100).round(1)
    st.dataframe(br.sort_values("points_of_risk", ascending=False), hide_index=True,
                 width="stretch",
                 column_config={"severity": st.column_config.ProgressColumn(
                     "Severity", min_value=0.0, max_value=1.0, format="%.2f")})

    _, inter = sig.load()
    msgs = inter[inter.customer_id == r.customer_id].sort_values("date")
    st.markdown("### Source interactions")
    if msgs.empty:
        st.info("No interactions on record - silence is itself a signal here.")
    for m in msgs.itertuples():
        st.markdown(f"**{m.date}** · `{m.channel}` — {m.text}")
