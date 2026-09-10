"""Layer 3: Graph Attention Network for attack-path risk."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader

from config import (
    NODE_FEATURES, EDGE_FEATURES, GNN_HIDDEN, GNN_HEADS,
    GNN_EMBED_DIM, GNN_EPOCHS, GNN_LR, GNN_DROPOUT, MODEL_DIR, SEED,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


class GATRisk(torch.nn.Module):
    """2-layer Graph Attention Network -> 32-dim graph embedding + scalar head."""

    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(NODE_FEATURES, GNN_HIDDEN, heads=GNN_HEADS,
                             edge_dim=EDGE_FEATURES, dropout=GNN_DROPOUT)
        self.conv2 = GATConv(GNN_HIDDEN * GNN_HEADS, GNN_EMBED_DIM, heads=1,
                             edge_dim=EDGE_FEATURES, dropout=GNN_DROPOUT)
        self.head = torch.nn.Linear(GNN_EMBED_DIM, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=GNN_DROPOUT, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        emb = global_mean_pool(x, batch)       # [B, 32]
        out = self.head(emb)                   # [B, 1]
        return emb, out.squeeze(-1)


def train(graphs, labels, epochs=GNN_EPOCHS, verbose=True):
    """Train the GAT on graph-level regression of path danger score."""
    data_list = []
    for g, y in zip(graphs, labels):
        from graph_layer import graph_to_pyg
        data_list.append(graph_to_pyg(g, label=y))

    n = len(data_list)
    split = int(0.7 * n)
    train_data = data_list[:split]
    test_data = data_list[split:]

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=4)

    model = GATRisk()
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    best_loss = float("inf")
    history = {"train": [], "test": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            _, pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_data)

        # eval
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                _, pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                test_loss += loss_fn(pred, batch.y).item() * batch.num_graphs
        test_loss /= max(len(test_data), 1)

        history["train"].append(train_loss)
        history["test"].append(test_loss)

        if verbose and epoch % 10 == 0:
            print(f"[GNN] epoch {epoch:3d}  train_mse={train_loss:.5f}  test_mse={test_loss:.5f}")

        best_loss = min(best_loss, test_loss)

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gat.pt"))
    print(f"[GNN] Training done. Best test MSE = {best_loss:.5f}")
    return model, history


@torch.no_grad()
def embed(model, data):
    """Get the 32-dim graph embedding for a single PyG Data object."""
    model.eval()
    batch = torch.zeros(data.x.size(0), dtype=torch.long)
    emb, _ = model(data.x, data.edge_index, data.edge_attr, batch)
    return emb.squeeze(0).numpy()