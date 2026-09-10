"""End-to-end pipeline: Layer 1 -> Layer 2 -> Layer 3 -> Layer 4."""

import os
import pickle
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, DATA_DIR, SEED
import data_layer
import graph_layer
import gnn_model
import monte_carlo
import xgb_model

np.random.seed(SEED)


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n=== LAYER 1: Data Ingestion ===")
    scenarios = data_layer.build_all()

    print("\n=== LAYER 2: Attack Graphs ===")
    graphs, labels = graph_layer.build_all(scenarios)

    print("\n=== LAYER 3: GNN Training ===")
    model, history = gnn_model.train(graphs, labels, verbose=True)

    # ---- embed each graph ----
    embeddings = []
    tabular = []
    from graph_layer import graph_to_pyg
    for g, s in zip(graphs, scenarios):
        data = graph_to_pyg(g)
        emb = gnn_model.embed(model, data)
        embeddings.append(emb)

        assets = s["assets"]
        vulns = s["vulnerabilities"]
        tabular.append([
            assets["criticality"].mean(),
            assets["cms"].mean(),
            vulns["tef"].mean() if len(vulns) else 0.0,
            len(assets),
        ])
    embeddings = np.array(embeddings)
    tabular = np.array(tabular)

    print("\n=== LAYER 4a: Monte Carlo ===")
    mc_results = []
    for emb, tab in zip(embeddings, tabular):
        # Use TEF-based LEF as lambda estimate
        lef = float(tab[2]) * 0.5 + 0.1        # crude proxy
        result = monte_carlo.simulate(lef=lef)
        mc_results.append(result)
    eals = np.array([r["EAL"] for r in mc_results])
    print(f"[MC] Mean EAL = ₹{eals.mean()/1e7:.2f} Cr")
    print(f"[MC] Mean VaR_95 = ₹{np.mean([r['VaR_95'] for r in mc_results])/1e7:.2f} Cr")

    print("\n=== LAYER 4b: XGBoost ===")
    X = xgb_model.build_features(embeddings, tabular)
    y = eals
    xgb = xgb_model.train(X, y)

    # save everything for the Streamlit app
    artifacts = {
        "scenarios": scenarios,
        "graphs": graphs,
        "labels": labels,
        "embeddings": embeddings,
        "tabular": tabular,
        "mc_results": mc_results,
        "history": history,
    }
    with open(os.path.join(OUTPUT_DIR, "artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    print("\n=== Pipeline complete ===\n")


if __name__ == "__main__":
    run()