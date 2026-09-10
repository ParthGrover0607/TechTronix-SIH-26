"""All hyperparameters in one place."""

# --- Data ---
NUM_SCENARIOS = 40              # total graphs to generate
NUM_ASSETS_PER_GRAPH = (8, 15)  # min, max nodes per graph
SEED = 42

# --- MITRE ATT&CK ---
TACTICS = [
    "InitialAccess", "Execution", "Persistence",
    "PrivEsc", "DefenseEvasion", "CredAccess"
]
NUM_TACTICS = len(TACTICS)

# Tactic -> list of (technique_id, technique_name, base_difficulty)
TECHNIQUES = {
    "InitialAccess":  [("T1190", "Exploit Public-Facing App"), ("T1566", "Phishing"), ("T1133", "External Remote Services")],
    "Execution":      [("T1059", "Command & Scripting"), ("T1204", "User Execution")],
    "Persistence":    [("T1547", "Boot Autostart"), ("T1543", "Create System Process")],
    "PrivEsc":        [("T1068", "Exploit for PrivEsc"), ("T1548", "Abuse Elevation Control")],
    "DefenseEvasion": [("T1562", "Impair Defenses"), ("T1070", "Indicator Removal")],
    "CredAccess":     [("T1078", "Valid Accounts"), ("T1003", "Credential Dumping")],
}

# --- GNN ---
NODE_FEATURES = 3               # criticality, cms, compromised
EDGE_FEATURES = NUM_TACTICS + 2 # tactic one-hot + epss + cvss = 8
GNN_HIDDEN = 32
GNN_HEADS = 4
GNN_EMBED_DIM = 32
GNN_EPOCHS = 60
GNN_LR = 0.001
GNN_DROPOUT = 0.3

# --- Monte Carlo ---
MC_ITERATIONS = 5000
MC_LAMBDA = 2.5                 # avg attacks/year (Poisson)
MC_MU = 14.5                    # log-mean of loss (Lognormal)
MC_SIGMA = 2.0                  # log-std of loss

# --- XGBoost ---
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 4
XGB_LR = 0.1

# --- Paths ---
OUTPUT_DIR = "outputs"
DATA_DIR = "outputs/data"
MODEL_DIR = "outputs/models"