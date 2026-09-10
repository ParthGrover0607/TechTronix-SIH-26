"""Layer 2: Build MITRE ATT&CK attack graphs."""

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from config import TACTICS, TECHNIQUES, NUM_TACTICS


def build_graph(scenario):
    """
    Directed attack graph:
      nodes = entry + assets
      edges = ATT&CK techniques (from entry to assets, and laterally)
    """
    G = nx.DiGraph()
    assets = scenario["assets"]
    vulns = scenario["vulnerabilities"]

    # nodes
    for _, a in assets.iterrows():
        G.add_node(
            a["asset_id"],
            name=a["asset_name"],
            criticality=a["criticality"],
            cms=a["cms"],
            is_entry=a["is_entry"],
        )

    entry_ids = assets[assets["is_entry"]]["asset_id"].tolist()
    non_entry = assets[~assets["is_entry"]]["asset_id"].tolist()

    # edges: entry -> each non-entry asset (lateral movement)
    # label each edge with an ATT&CK technique sampled from tactics
    for src in entry_ids:
        for dst in non_entry:
            tactic = np.random.choice(TACTICS)
            tech_id, tech_name = TECHNIQUES[tactic][np.random.randint(len(TECHNIQUES[tactic]))]

            # attach a vuln from dst to this edge (if any)
            dst_vulns = vulns[vulns["asset_id"] == dst]
            if len(dst_vulns) == 0:
                epss, cvss = 0.1, 3.0
            else:
                v = dst_vulns.sample(1).iloc[0]
                epss, cvss = v["epss"], v["cvss"]

            G.add_edge(
                src, dst,
                tactic=tactic, technique=tech_id, technique_name=tech_name,
                epss=epss, cvss=cvss,
            )

    return G


def compute_path_danger(G):
    """
    PathDangerScore = Π(EPSS along path) × Criticality(target)
    Return the maximum over all entry->target paths (depth-limited).
    """
    entry_ids = [n for n in G.nodes if G.nodes[n].get("is_entry")]
    if not entry_ids:
        return 0.0

    max_score = 0.0
    for src in entry_ids:
        for tgt in G.nodes:
            if tgt == src or G.nodes[tgt].get("is_entry"):
                continue
            if not nx.has_path(G, src, tgt):
                continue
            try:
                for path in nx.all_simple_paths(G, src, tgt, cutoff=4):
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
    graphs = []
    labels = []
    for s in scenarios:
        G = build_graph(s)
        score = compute_path_danger(G)
        graphs.append(G)
        labels.append(score)

    print(f"[Layer 2] Built {len(graphs)} attack graphs")
    print(f"[Layer 2] Danger scores -> min={min(labels):.3f}  max={max(labels):.3f}  mean={np.mean(labels):.3f}")
    return graphs, labels