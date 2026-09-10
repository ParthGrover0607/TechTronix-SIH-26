"""Layer 4b: XGBoost fast surrogate trained on GNN embeddings + Monte Carlo EAL."""

import os
import numpy as np
import xgboost as xgb

from config import XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LR, MODEL_DIR


def build_features(embeddings, tabular):
    """
    Concatenate GNN embedding (32-d) with tabular features:
    [criticality_avg, cms_avg, tef_avg, num_nodes]
    """
    return np.hstack([embeddings, tabular])


def train(X, y):
    model = xgb.XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LR,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, "xgb.json"))
    print(f"[XGBoost] Trained on {X.shape[0]} samples, {X.shape[1]} features")
    return model


def predict(model, X):
    return model.predict(X)