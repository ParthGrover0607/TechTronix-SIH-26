"""Layer 2: Build MITRE ATT&CK attack graphs with realistic multi-hop paths."""

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from config import TACTICS, TECHNIQUES, NUM_TACTICS, SEED

np.random.seed(SEED)


# Tactic assignment per hop depth:
#   hop 0 (entry -> first asset)     : InitialAccess
#   hop 1 (first -> second asset)    : Execution or CredAccess
#   hop 2 (second -> third asset)    : PrivEsc or Persistence
#   hop 3+ (deeper lateral movement) : DefenseEvasion or CredAccess
HOP_TACTICS = {
    0: ["InitialAccess"],
    1: ["Execution", "CredAccess"],
    2: ["PrivEsc", "Persistence"],
    3: ["DefenseEvasion", "CredAccess"],
}


def _pick_tactic(depth):
    """Pick a tactic appropriate for this hop depth."""
    if depth in HOP_TACTICS:
        return np.random.choice(HOP_TACTICS[depth])
    return np.random.choice(HOP_TACTICS[3])


def _pick_technique(tactic):
    """Pick a specific technique within a tactic."""
    options = TECHNIQUES[tactic]
    tech_id, tech_name = options[np.random.randint(len(options))]
    return tech_id, tech_name


def _pick_edge_features(dst, vulns):
    """Attach EPSS/CVSS from a vulnerability on the destination asset."""
    dst_vulns = vulns[vulns["asset_id"] == dst]
    if len(dst_vulns) == 0:
        return 0.1, 3.0
    v = dst_vulns.sample(1).iloc[0]
    return float(v["epss"]), float(v["cvss"])


def build_graph(scenario):
    """
    Build a realistic multi-hop attack graph.

    Strategy:
      - Start at the entry node.
      - Do random walks of depth 2-4.
      - Each hop consumes a new (unvisited) asset.
      - Assign tactic based on hop depth so paths read like real attack chains.
      - Allow converging paths: multiple walks can reach the same asset.
    """
    G = nx.DiGraph()
    assets = scenario["assets"]
    vulns = scenario["vulnerabilities"]

    # --- 1. add all nodes ---
    for _, a in assets.iterrows():
        G.add_node(
            a["asset_id"],
            name=a["asset_name"],
            criticality=float(a["criticality"]),
            cms=float(a["cms"]),
            is_entry=bool(a["is_entry"]),
        )

    entry_nodes = assets[assets["is_entry"]]["asset_id"].tolist()
    all_assets = assets["asset_id"].tolist()
    non_entry = [a for a in all_assets if a not in entry_nodes]

    if not entry_nodes or not non_entry:
        return G

    # --- 2. generate several random walks from entry ---
    # More walks => denser graph => more multi-hop paths for the GNN to learn from
    num_walks = max(3, len(non_entry) // 2)

    for _ in range(num_walks):
        src = np.random.choice(entry_nodes)
        depth = 0
        max_depth = np.random.randint(2, 5)  # 2-4 hops

        visited = {src}
        current = src

        while depth < max_depth:
            # pick next asset (not the current, not already visited on this walk)
            candidates = [a for a in all_assets if a != current and a not in visited]
            if not candidates:
                break
            nxt = np.random.choice(candidates)

            # skip if edge already exists (keep the first assignment)
            if G.has_edge(current, nxt):
                visited.add(nxt)
                current = nxt
                depth += 1
                continue

            # pick tactic based on hop depth, then a technique within it
            tactic = _pick_tactic(depth)
            tech_id, tech_name = _pick_technique(tactic)
            epss, cvss = _pick_edge_features(nxt, vulns)

            G.add_edge(
                current, nxt,
                tactic=tactic,
                technique=tech_id,
                technique_name=tech_name,
                epss=epss,
                cvss=cvss,
                hop=depth,
            )

            visited.add(nxt)
            current = nxt
            depth += 1

    return G


def compute_path_danger(G):
    """
    PathDangerScore = Π(EPSS along path) × Criticality(target)
    Return max over all entry→target paths up to depth 5.
    Multi-hop paths naturally have smaller EPSS products — so this
    gives the model a real signal about path length and structure.
    """
    entry_nodes = [n for n in G.nodes if G.nodes[n].get("is_entry")]
    if not entry_nodes:
        return 0.0

    max_score = 0.0
    for src in entry_nodes:
        for tgt in G.nodes:
            if tgt == src or G.nodes[tgt].get("is_entry"):
                continue
            if not nx.has_path(G, src, tgt):
                continue
            try:
                for path in nx.all_simple_paths(G, src, tgt, cutoff=5):
                    score = 1.0
                    for i in range(len(path) - 1):
                        score *= G.edges[path[i], path[i + 1]]["epss"]
                    score *= G.nodes[tgt]["criticality"]
                    max_score = max(max_score, score)
            except nx.NetworkXNoPath:
                continue
    return float(max_score)


def graph_to_pyg(G, label=None):
    """Convert NetworkX DiGraph -> PyG Data object."""
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    # node features: [criticality, cms, is_entry]
    x = torch.tensor(
        [[G.nodes[n]["criticality"], G.nodes[n]["cms"], float(G.nodes[n]["is_entry"])]
         for n in nodes],
        dtype=torch.float,
    )

    # edges
    edges, edge_feats = [], []
    for u, v, d in G.edges(data=True):
        edges.append([idx[u], idx[v]])
        onehot = [1.0 if d["tactic"] == t else 0.0 for t in TACTICS]
        edge_feats.append(onehot + [d["epss"], d["cvss"] / 10.0])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, NUM_TACTICS + 2), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data


def build_all(scenarios):
    """Build all graphs and compute labels. Prints path-length stats."""
    graphs, labels = [], []

    for s in scenarios:
        G = build_graph(s)
        score = compute_path_danger(G)
        graphs.append(G)
        labels.append(score)

    # --- diagnostic: path length distribution ---
    path_lengths = []
    for G in graphs:
        entry_nodes = [n for n in G.nodes if G.nodes[n].get("is_entry")]
        for src in entry_nodes:
            for tgt in G.nodes:
                if tgt == src or G.nodes[tgt].get("is_entry"):
                    continue
                try:
                    for path in nx.all_simple_paths(G, src, tgt, cutoff=5):
                        path_lengths.append(len(path) - 1)  # hop count
                except nx.NetworkXNoPath:
                    continue

    print(f"[Layer 2] Built {len(graphs)} attack graphs")
    print(f"[Layer 2] Danger scores -> min={min(labels):.3f}  max={max(labels):.3f}  mean={np.mean(labels):.3f}")
    if path_lengths:
        unique, counts = np.unique(path_lengths, return_counts=True)
        dist = "  ".join(f"{u}hop={c}" for u, c in zip(unique, counts))
        print(f"[Layer 2] Path-length distribution: {dist}")

    return graphs, labels