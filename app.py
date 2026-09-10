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
# ---------- Tab 2: Layer 2 ----------
# ---------- Tab 2: Layer 2 ----------
# ---------- Tab 2: Layer 2 ----------
with tab2:
    st.subheader("Layer 2 — Attack Graph")

    idx = st.selectbox("Pick scenario", range(len(graphs)), key="g")
    G = graphs[idx]

    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", G.number_of_nodes())
    col2.metric("Edges", G.number_of_edges())
    col3.metric("Path danger score", f"{labels[idx]:.4f}")

    st.write("**Attack graph** — nodes laid out by hop depth from entry")

    import networkx as nx

    # ---------- Compute hop depth for each node via BFS ----------
    entry_nodes = [n for n in G.nodes if G.nodes[n].get("is_entry")]
    depth = {}
    for e in entry_nodes:
        depth[e] = 0
        for node, d in nx.single_source_shortest_path_length(G, e).items():
            if node not in depth or d < depth[node]:
                depth[node] = d

    # nodes not reachable from any entry -> put at max depth + 1
    max_d = max(depth.values()) if depth else 0
    for n in G.nodes:
        if n not in depth:
            depth[n] = max_d + 1

    # ---------- Group nodes by depth ----------
    layers = {}
    for n, d in depth.items():
        layers.setdefault(d, []).append(n)

    # ---------- Position nodes ----------
    pos = {}
    n_layers = max(layers.keys()) + 1
    for d, nodes in layers.items():
        nodes_sorted = sorted(nodes)          # stable ordering
        k = len(nodes_sorted)
        # spread horizontally, centered
        for i, n in enumerate(nodes_sorted):
            x = (i - (k - 1) / 2) * 1.4
            y = -d * 1.2                      # top-to-bottom
            pos[n] = (x, y)

    # ---------- Figure ----------
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # --- node styling ---
    node_colors, node_sizes, node_edges = [], [], []
    for n in G.nodes:
        crit = G.nodes[n]["criticality"]
        if G.nodes[n].get("is_entry"):
            node_colors.append("blue")
            node_sizes.append(1400)
            node_edges.append("black")
        else:
            if crit >= 0.75:
                node_colors.append("red")
            elif crit >= 0.5:
                node_colors.append("orange")
            else:
                node_colors.append("green")
            node_sizes.append(700 + crit * 800)
            node_edges.append("black")

    # --- edge styling ---
    TACTIC_COLORS = {
        "InitialAccess":  "blue",
        "Execution":      "orange",
        "Persistence":    "green",
        "PrivEsc":        "red",
        "DefenseEvasion": "purple",
        "CredAccess":     "brown",
    }
    edge_colors = [TACTIC_COLORS.get(G.edges[u, v]["tactic"], "gray")
                   for u, v in G.edges]
    edge_widths = [1.2 + G.edges[u, v]["epss"] * 3.5 for u, v in G.edges]

    # --- draw ---
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        edgecolors=node_edges, linewidths=1.0,
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowsize=14, alpha=0.7,
        connectionstyle="arc3,rad=0.08",
        node_size=900,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={n: n.split("_")[-1] for n in G.nodes},
        font_size=8,
    )

    # ---------- Draw depth labels on the left ----------
    for d in range(n_layers):
        y = -d * 1.2
        label = "Entry" if d == 0 else f"Hop {d}"
        ax.text(
            ax.get_xlim()[0] - 0.8, y, label,
            fontsize=9, fontweight="bold", color="gray",
            ha="right", va="center",
        )

    # ---------- Draw faint horizontal separators ----------
    x_min = ax.get_xlim()[0] - 0.3
    x_max = ax.get_xlim()[1] + 0.3
    for d in range(n_layers - 1):
        y_mid = -d * 1.2 - 0.6
        ax.plot([x_min, x_max], [y_mid, y_mid],
                color="lightgray", linewidth=0.6, linestyle="--", zorder=0)

    ax.axis("off")
    ax.set_title(
        f"Scenario {idx} — {len(entry_nodes)} entry, {len(G.nodes) - len(entry_nodes)} assets, "
        f"{n_layers - 1} hop depth",
        fontsize=10, pad=10,
    )

    st.pyplot(fig)

    # ---------- Legend ----------
    st.caption(
        "Node: blue = entry, red = high crit, orange = medium crit, green = low crit  •  "
        "Edge: color = ATT&CK tactic, thickness = EPSS  •  "
        "Rows show hop depth (Entry → Hop 1 → Hop 2 …)"
    )

    with st.expander("Tactic color legend"):
        st.dataframe(
            pd.DataFrame([{"Tactic": k, "Color": v} for k, v in TACTIC_COLORS.items()]),
            use_container_width=True, hide_index=True,
        )

    # ---------- Edge detail ----------
    st.write("**Edge details**")
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "from": u, "to": v,
            "hop": d.get("hop", "—"),
            "technique": d["technique"],
            "name": d["technique_name"],
            "tactic": d["tactic"],
            "epss": round(d["epss"], 3),
            "cvss": d["cvss"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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