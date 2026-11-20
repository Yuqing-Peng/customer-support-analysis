# Click-through version for demo in the browser

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from support_analysis.config import PipelineConfig  
from support_analysis.insights import DEFAULT_MODEL, generate_insights  
from support_analysis.pipeline import dataset_context, step_clean, step_cluster  
from support_analysis.sample_data import make_sample  

st.set_page_config(page_title="Support Ticket Analysis", layout="wide")
st.title("Customer Support Ticket Analysis")
st.caption("Ingest → clean → batch K-means → Claude API insights for the support team")

with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Source", ["Sample tickets", "Upload export"], index=0)
    uploaded = st.file_uploader("Ticket export (CSV/XLSX)", type=["csv", "xlsx", "xls"]) if source == "Upload export" else None
    n_sample = st.slider("Sample size", 300, 5000, 2000, 100) if source == "Sample tickets" else None
    st.header("2 · Batch K-means")
    k_min, k_max = st.slider("K range", 2, 15, (3, 10))
    batch_size = st.select_slider("Batch size", [64, 128, 256, 512, 1024], value=256)
    n_epochs = st.slider("Epochs over batches", 1, 5, 3)
    st.header("3 · Cloud insights")
    use_api = st.toggle("Call Claude API", value=False, help="Needs ANTHROPIC_API_KEY or an `ant auth login` profile. Off = offline heuristic.")
    model = st.text_input("Model", DEFAULT_MODEL, disabled=not use_api)

cfg = PipelineConfig(k_min=k_min, k_max=k_max, batch_size=batch_size, n_epochs=n_epochs, use_api=use_api, model=model)


@st.cache_data(show_spinner="Loading tickets…")
def load_raw(n: int | None, file_bytes: bytes | None, name: str | None) -> pd.DataFrame:
    if file_bytes is None:
        return make_sample(n=n)
    import io
    buf = io.BytesIO(file_bytes)
    return pd.read_csv(buf) if name.lower().endswith(".csv") else pd.read_excel(buf)


@st.cache_data(show_spinner="Cleaning tickets…")
def cached_clean(raw: pd.DataFrame, _cfg: PipelineConfig, key: tuple):
    return step_clean(raw, _cfg)


@st.cache_data(show_spinner="Vectorizing and clustering in batches…")
def cached_cluster(tickets: pd.DataFrame, _cfg: PipelineConfig, key: tuple):
    vec, X, scores, seg, profiles, trend, samples = step_cluster(tickets, _cfg)
    return X.shape[1], scores, seg.k, seg.silhouette, seg.labels, seg.batch_log, profiles, trend, samples


if source == "Upload export" and uploaded is None:
    st.info("Upload a ticket export, or switch the source to **Sample tickets**.")
    st.stop()

raw = load_raw(n_sample, uploaded.getvalue() if uploaded else None, uploaded.name if uploaded else None)

# ----------Step 1-2: ingest + clean
st.header("Ingest  →  Clean")
try:
    tickets, rep = cached_clean(raw, cfg, (cfg.id_col, cfg.text_cols, cfg.min_text_chars))
except ValueError as e:
    st.error(str(e))
    st.stop()

c = st.columns(6)
c[0].metric("Raw rows", f"{len(raw):,}")
c[1].metric("Tickets kept", f"{rep.n_rows_out:,}")
c[2].metric("Duplicates removed", rep.basic.duplicates_removed + rep.content_duplicates_removed)
c[3].metric("Empty dropped", rep.empty_or_short_dropped)
c[4].metric("PII redacted (tickets)", rep.tickets_with_pii)
c[5].metric("HTML / signatures fixed", f"{rep.tickets_with_html} / {rep.signatures_removed}")

with st.expander("Before vs after cleaning (example tickets)"):
    raw_view = raw.copy()
    raw_view.columns = [str(col).strip().lower().replace(" ", "_") for col in raw_view.columns]
    messy = pd.DataFrame()
    if {"body", cfg.id_col} <= set(raw_view.columns):
        messy = raw_view[raw_view["body"].astype(str).str.contains(r"<p>|@|Thanks,|wrote:", regex=True)].head(5)
    for _, row in messy.iterrows():
        after = tickets.loc[tickets[cfg.id_col] == str(row[cfg.id_col]).strip(), "text"]
        l, r = st.columns(2)
        l.code(str(row["body"])[:600], language=None)
        r.success(after.iloc[0] if len(after) else "(dropped)")
    st.json(rep.to_dict(), expanded=False)

# --------- Step 3-4: vectorize + batch K-means
st.header(" Vectorize  →  Batch K-means")
dims, scores, k, sil, labels, batch_log, profiles, trend, samples = cached_cluster(
    tickets, cfg, (len(tickets), k_min, k_max, batch_size, n_epochs)
)
st.write(f"TF-IDF on cleaned text → **{dims}-dim** vectors. Selected **K = {k}** (silhouette {sil:.3f}); "
         f"MiniBatchKMeans trained on **{max(b['batch'] for b in batch_log)} batches × {n_epochs} epochs**.")
l, r = st.columns(2)
score_df = pd.DataFrame({"k": list(scores), "silhouette": list(scores.values())})
l.plotly_chart(px.line(score_df, x="k", y="silhouette", markers=True, title="Silhouette score by K"), use_container_width=True)
blog = pd.DataFrame(batch_log)
blog["step"] = range(1, len(blog) + 1)
r.plotly_chart(px.line(blog, x="step", y="batch_inertia", color=blog["epoch"].astype(str),
                       markers=True, title="Batch inertia while streaming batches",
                       labels={"color": "epoch", "step": "batch update"}), use_container_width=True)
st.dataframe(profiles, use_container_width=True, hide_index=True)
if len(trend):
    st.plotly_chart(px.area(trend, x="week", y="n_tickets", color=trend["cluster"].astype(str),
                            title="Weekly tickets per cluster", labels={"color": "cluster"}), use_container_width=True)

# --------- Step 5: AI insights
st.header("AI insights")
if st.button("Generate insights", type="primary"):
    ctx = dataset_context(raw, tickets, k, sil, cfg)
    with st.spinner("Calling Claude…" if use_api else "Building offline insights…"):
        st.session_state["insights"] = generate_insights(profiles, samples, ctx, use_api=use_api, model=model,
                                                         log=lambda m: st.warning(m.strip(" !")))

ins = st.session_state.get("insights")
if ins:
    st.subheader("Summary")
    st.write(ins["overall_summary"])
    st.markdown("**Top priorities**\n" + "\n".join(f"- {p}" for p in ins["top_priorities"]))
    st.caption(f"Source: {ins['source']} · {ins['generated_at']}")
    order = {"high": 0, "medium": 1, "low": 2}
    for c_ in sorted(ins["clusters"], key=lambda x: order.get(x["priority"], 3)):
        badge = {"high": "🔴", "medium": "🟠", "low": "🟢"}.get(c_["priority"], "")
        m = c_["metrics"]
        with st.expander(f"{badge} Cluster {c_['cluster']} — {c_['theme']}  ({m.get('n_tickets')} tickets)", expanded=c_["priority"] == "high"):
            st.write(c_["summary"])
            st.markdown(f"**Likely root cause:** {c_['likely_root_cause']}")
            st.markdown("**Recommended actions**\n" + "\n".join(f"- {a}" for a in c_["recommended_actions"]))
            st.caption("Top terms: " + ", ".join(c_["top_terms"]))
            for t in c_["sample_tickets"][:3]:
                st.text(t)
