"""Minimal Streamlit UI. No styling. Just the pipeline output."""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="PS105 Prototype", layout="wide")

ARTIFACTS = "outputs/artifacts.pkl"


@st.cache_data
def load():
    with open(ARTIFACTS, "rb") as f:
        return pickle.load(f)


if not os.path.exists(ARTIFACTS):
    st.error("Run `python pipeline.py` first to generate artifacts.")
    st.stop()

A = load()
scenarios = A["scenarios"]
graphs = A["graphs"]
labels = A["labels"]
mc_results = A["mc_results"]
history = A["history"]

st.title("PS105 Prototype — Cyber Risk Quantification")
st.caption("Layer 1 (Ingestion) → Layer 2 (Graph) → Layer 3 (GNN) → Layer 4 (Monte Carlo + XGBoost)")

tab1, tab2, tab3, tab4 = st.tabs(["Data (L1)", "Graph (L2)", "Models (L3/L4)", "Inference"])

# ---------- Tab 1: Layer 1 ----------
with tab1:
    st.subheader("Layer 1 — Ingested Scenarios")
    st.write(f"Total scenarios: **{len(scenarios)}**")

    idx = st.selectbox("Pick scenario", range(len(scenarios)))
    s = scenarios[idx]

    st.text(f"industry={s['industry']}  attack_vector={s['attack_vector']}")
    st.write("**Assets**")
    st.dataframe(s["assets"], use_container_width=True)
    st.write("**Vulnerabilities (with TEF)**")
    st.dataframe(s["vulnerabilities"], use_container_width=True)

# ---------- Tab 2: Layer 2 ----------
with tab2:
    st.subheader("Layer 2 — Attack Graph")

    idx = st.selectbox("Pick scenario", range(len(graphs)), key="g")
    G = graphs[idx]

    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", G.number_of_nodes())
    col2.metric("Edges", G.number_of_edges())
    col3.metric("Path danger score", f"{labels[idx]:.4f}")

    st.write("**Adjacency (edges with ATT&CK labels)**")
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "from": u, "to": v,
            "technique": d["technique"],
            "name": d["technique_name"],
            "tactic": d["tactic"],
            "epss": round(d["epss"], 3),
            "cvss": d["cvss"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------- Tab 3: Layer 3/4 ----------
with tab3:
    st.subheader("Layer 3 — GNN Training")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(history["train"], label="train MSE")
    ax.plot(history["test"], label="test MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Layer 4a — Monte Carlo distribution")
    mc_idx = st.selectbox("Pick scenario", range(len(mc_results)), key="mc")
    r = mc_results[mc_idx]
    col1, col2, col3 = st.columns(3)
    col1.metric("EAL", f"₹{r['EAL']/1e7:.2f} Cr")
    col2.metric("VaR 95", f"₹{r['VaR_95']/1e7:.2f} Cr")
    col3.metric("CVaR 95", f"₹{r['CVaR_95']/1e7:.2f} Cr")

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.hist(r["losses"] / 1e7, bins=50, color="steelblue")
    ax2.set_xlabel("Annual loss (₹ Cr)")
    ax2.set_ylabel("frequency")
    st.pyplot(fig2)

# ---------- Tab 4: Inference ----------
with tab4:
    st.subheader("End-to-end inference: GNN embedding → XGBoost prediction")
    st.write("XGBoost features = 32-dim GNN embedding + 4 tabular features")
    st.write(f"Total feature dimension: **{A['embeddings'].shape[1] + A['tabular'].shape[1]}**")

    idx = st.selectbox("Pick scenario", range(len(scenarios)), key="inf")
    emb = A["embeddings"][idx]
    st.write("**GNN embedding (32-d)**")
    st.write(np.round(emb, 3))
    st.write("**Monte Carlo ground truth EAL**")
    st.write(f"₹{mc_results[idx]['EAL']/1e7:.3f} Cr")

st.sidebar.markdown("---")
st.sidebar.caption("Prototype scope: Layers 1–4 (no dashboard polish)")