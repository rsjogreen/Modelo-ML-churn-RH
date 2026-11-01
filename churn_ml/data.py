"""Synthetic employee data for the initial churn exercise."""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "tenure_months",
    "satisfaction",
    "monthly_hours",
    "overtime",
    "promotions",
]
TARGET_COLUMN = "churn"


def generate_employee_data(
    n_samples: int = 1_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate reproducible, fictional employee records and churn labels."""
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10.")

    rng = np.random.default_rng(random_state)
    tenure_months = rng.integers(1, 121, size=n_samples)
    satisfaction = rng.uniform(0.1, 1.0, size=n_samples)
    monthly_hours = np.clip(rng.normal(160, 25, size=n_samples), 80, 260)
    overtime = rng.binomial(1, 0.3, size=n_samples)
    promotions = rng.poisson(0.7, size=n_samples)

    churn_log_odds = (
        1.4
        - 3.2 * satisfaction
        - 0.012 * tenure_months
        + 0.85 * overtime
        + 0.008 * (monthly_hours - 160)
        - 0.25 * promotions
    )
    churn_probability = 1 / (1 + np.exp(-churn_log_odds))

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "satisfaction": satisfaction,
            "monthly_hours": monthly_hours,
            "overtime": overtime,
            "promotions": promotions,
            "churn": rng.binomial(1, churn_probability),
        }
    )
