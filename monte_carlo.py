"""Layer 4a: Monte Carlo simulation of annual loss."""

import numpy as np

from config import MC_ITERATIONS, MC_LAMBDA, MC_MU, MC_SIGMA


def simulate(lef=None, iterations=MC_ITERATIONS,
             lam=MC_LAMBDA, mu=MC_MU, sigma=MC_SIGMA):
    """
    Frequency  ~ Poisson(lambda)
    Severity   ~ Lognormal(mu, sigma)
    Returns distribution + summary metrics.
    """
    if lef is not None:
        lam = max(float(lef), 0.01)

    losses = np.zeros(iterations)
    for i in range(iterations):
        n_attacks = np.random.poisson(lam)
        if n_attacks == 0:
            losses[i] = 0.0
        else:
            per_attack = np.random.lognormal(mean=mu, sigma=sigma, size=n_attacks)
            losses[i] = per_attack.sum()

    eal = float(np.mean(losses))
    var_95 = float(np.percentile(losses, 95))
    cvar_95 = float(np.mean(losses[losses >= var_95]))

    return {
        "losses": losses,
        "EAL": eal,
        "VaR_95": var_95,
        "CVaR_95": cvar_95,
        "max": float(losses.max()),
    }


def summarize(result):
    return (
        f"EAL=₹{result['EAL']/1e7:.2f}Cr  "
        f"VaR95=₹{result['VaR_95']/1e7:.2f}Cr  "
        f"CVaR95=₹{result['CVaR_95']/1e7:.2f}Cr"
    )