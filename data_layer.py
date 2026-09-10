"""Layer 1: Synthetic data generation + TEF calculation."""

import os
import numpy as np
import pandas as pd

from config import SEED, DATA_DIR

np.random.seed(SEED)


INDUSTRIES = ["Finance", "Healthcare", "Retail", "Manufacturing", "Education", "IT"]
ATTACK_VECTORS = ["Phishing", "Ransomware", "DDoS", "SQLi", "Insider", "SupplyChain"]


def generate_vulnerabilities(n=40):
    """Generate a pool of vulnerabilities with EPSS + KEV flags."""
    cve_ids = [f"CVE-2024-{1000 + i}" for i in range(n)]
    return pd.DataFrame({
        "cve_id": cve_ids,
        "cvss": np.round(np.random.uniform(4.0, 10.0, n), 1),
        "epss": np.round(np.random.uniform(0.02, 0.95, n), 3),
        "in_kev": np.random.choice([True, False], n, p=[0.3, 0.7]),
    })


def compute_tef(vulns: pd.DataFrame) -> pd.DataFrame:
    """TEF = EPSS × (2.0 if in_kev else 1.0)"""
    v = vulns.copy()
    v["kev_multiplier"] = np.where(v["in_kev"], 2.0, 1.0)
    v["tef"] = v["epss"] * v["kev_multiplier"]
    return v


def generate_scenarios(n_scenarios, vuln_pool):
    """
    Each scenario = one small enterprise with assets + assigned vulns.
    """
    scenarios = []
    for sid in range(n_scenarios):
        n_assets = np.random.randint(8, 15)
        industry = np.random.choice(INDUSTRIES)
        attack_vector = np.random.choice(ATTACK_VECTORS)

        # assets
        assets = []
        for i in range(n_assets):
            assets.append({
                "asset_id": f"S{sid}_A{i}",
                "asset_name": f"Asset-{i}",
                "criticality": round(float(np.random.uniform(0.3, 1.0)), 2),
                "cms": round(float(np.random.uniform(0.3, 0.95)), 2),
                "is_entry": i == 0,   # first asset is the internet-facing entry
            })
        assets_df = pd.DataFrame(assets)

        # assign 1-3 vulns to each asset
        assignments = []
        for a in assets:
            n_v = np.random.randint(1, 4)
            sample = vuln_pool.sample(n_v)
            for _, v in sample.iterrows():
                assignments.append({
                    "asset_id": a["asset_id"],
                    "cve_id": v["cve_id"],
                    "cvss": v["cvss"],
                    "epss": v["epss"],
                    "in_kev": v["in_kev"],
                    "tef": v["epss"] * (2.0 if v["in_kev"] else 1.0),
                })

        scenarios.append({
            "scenario_id": sid,
            "industry": industry,
            "attack_vector": attack_vector,
            "assets": assets_df,
            "vulnerabilities": pd.DataFrame(assignments),
        })
    return scenarios


def build_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    pool = compute_tef(generate_vulnerabilities())
    pool.to_csv(os.path.join(DATA_DIR, "vulnerability_pool.csv"), index=False)

    scenarios = generate_scenarios(40, pool)
    print(f"[Layer 1] Generated {len(scenarios)} scenarios")
    print(f"[Layer 1] Vulnerability pool: {len(pool)} CVEs")
    return scenarios