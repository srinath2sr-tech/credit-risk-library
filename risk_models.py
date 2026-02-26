"""
credit-risk-library
====================
Market Risk VaR Analytics Library
Author: Srinath Reddy
"""

import numpy as np
from scipy.stats import norm


class RiskModel:
    """Base class for all risk models."""

    def __init__(self, name: str):
        self.name = name

    def compute(self) -> float:
        raise NotImplementedError("Subclasses must implement compute()")

    def report(self):
        print(f"Model: {self.name} | VaR: {self.compute():.4f}")


class HistoricalVaR(RiskModel):
    """
    Historical Simulation VaR.
    Uses actual past returns to estimate loss at a given confidence level.
    """

    def __init__(self, returns: list, confidence: float = 0.99):
        super().__init__(name="Historical VaR")
        self.returns = returns
        self.confidence = confidence

    def compute(self) -> float:
        return float(np.percentile(self.returns, (1 - self.confidence) * 100))


class ParametricVaR(RiskModel):
    """
    Parametric (Variance-Covariance) VaR.
    Assumes returns are normally distributed.
    """

    def __init__(self, mean: float, std: float, confidence: float = 0.99):
        super().__init__(name="Parametric VaR")
        self.mean = mean
        self.std = std
        self.confidence = confidence

    def compute(self) -> float:
        return float(self.mean - self.std * norm.ppf(self.confidence))


class MonteCarloVaR(RiskModel):
    """
    Monte Carlo VaR.
    Simulates thousands of return scenarios to estimate loss distribution.
    """

    def __init__(self, mean: float, std: float,
                 confidence: float = 0.99, simulations: int = 10000):
        super().__init__(name="Monte Carlo VaR")
        self.mean = mean
        self.std = std
        self.confidence = confidence
        self.simulations = simulations

    def compute(self) -> float:
        np.random.seed(42)
        simulated = np.random.normal(self.mean, self.std, self.simulations)
        return float(np.percentile(simulated, (1 - self.confidence) * 100))


# ── Quick test when file is run directly ──────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 500).tolist()
    mean = float(np.mean(returns))
    std  = float(np.std(returns))

    models = [
        HistoricalVaR(returns=returns, confidence=0.99),
        ParametricVaR(mean=mean, std=std, confidence=0.99),
        MonteCarloVaR(mean=mean, std=std, confidence=0.99),
    ]

    print("===== Daily VaR Report (99% Confidence) =====")
    for model in models:
        model.report()