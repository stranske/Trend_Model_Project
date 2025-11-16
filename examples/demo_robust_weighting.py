#!/usr/bin/env python3
"""Integration example demonstrating robust weighting methods."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from trend_analysis.logging_setup import setup_logging


def create_test_scenarios():
    """Create various covariance matrix scenarios for testing."""
    scenarios = {}

    # Scenario 1: Well-conditioned matrix
    scenarios["well_conditioned"] = pd.DataFrame(
        [[0.04, 0.002, 0.001], [0.002, 0.09, 0.003], [0.001, 0.003, 0.16]],
        index=["Asset_A", "Asset_B", "Asset_C"],
        columns=["Asset_A", "Asset_B", "Asset_C"],
    )

    # Scenario 2: Ill-conditioned (near-singular) matrix
    ill_cond = (
        np.array(
            [[1.0, 0.99999, 0.99998], [0.99999, 1.0, 0.99999], [0.99998, 0.99999, 1.0]]
        )
        * 0.04
    )
    scenarios["ill_conditioned"] = pd.DataFrame(
        ill_cond,
        index=["Asset_A", "Asset_B", "Asset_C"],
        columns=["Asset_A", "Asset_B", "Asset_C"],
    )

    # Scenario 3: Singular matrix (perfectly correlated)
    singular = np.ones((3, 3)) * 0.04
    scenarios["singular"] = pd.DataFrame(
        singular,
        index=["Asset_A", "Asset_B", "Asset_C"],
        columns=["Asset_A", "Asset_B", "Asset_C"],
    )

    # Scenario 4: Matrix with extreme variance differences
    extreme_var = np.diag([1e-8, 1.0, 100.0])
    scenarios["extreme_variances"] = pd.DataFrame(
        extreme_var,
        index=["Low_Vol", "Med_Vol", "High_Vol"],
        columns=["Low_Vol", "Med_Vol", "High_Vol"],
    )

    return scenarios


def demonstrate_robust_weighting():
    """Demonstrate robust weighting with different scenarios."""
    print("=== Robust Portfolio Weighting Demonstration ===\n")

    # Import after setting up logging to capture initialization messages
    try:
        from trend_analysis.plugins import create_weight_engine
    except ImportError as e:
        print(f"Cannot import trend_analysis modules: {e}")
        print("This demo requires the trend_analysis package dependencies.")
        return

    scenarios = create_test_scenarios()

    # Test different robust engines
    engines_to_test = [
        (
            "robust_mv",
            {
                "shrinkage_method": "ledoit_wolf",
                "condition_threshold": 1e10,
                "safe_mode": "hrp",
            },
        ),
        (
            "robust_mv",
            {
                "shrinkage_method": "oas",
                "condition_threshold": 1e8,
                "safe_mode": "risk_parity",
            },
        ),
        ("robust_risk_parity", {"condition_threshold": 1e10}),
    ]

    for engine_name, params in engines_to_test:
        print(f"\n--- Testing {engine_name} with params: {params} ---")

        try:
            engine = create_weight_engine(engine_name, **params)

            for scenario_name, cov_matrix in scenarios.items():
                print(f"\nScenario: {scenario_name}")
                print(
                    f"Matrix condition number: {np.linalg.cond(cov_matrix.values):.2e}"
                )

                try:
                    weights = engine.weight(cov_matrix)
                    print(f"Weights: {weights.round(4).to_dict()}")
                    print(f"Sum: {weights.sum():.6f}")
                    print(f"Min: {weights.min():.6f}, Max: {weights.max():.6f}")

                    if not np.isclose(weights.sum(), 1.0):
                        print("WARNING: Weights do not sum to 1!")

                except Exception as e:
                    print(f"ERROR in weight calculation: {e}")

        except Exception as e:
            print(f"ERROR creating engine {engine_name}: {e}")


def demonstrate_config_usage():
    """Show how robustness config would be used."""
    print("\n=== Configuration Usage Example ===\n")

    robust_config = {
        "portfolio": {
            "weighting_scheme": "robust_mv",
            "robustness": {
                "shrinkage": {"enabled": True, "method": "ledoit_wolf"},
                "condition_check": {
                    "enabled": True,
                    "threshold": 1e10,
                    "safe_mode": "hrp",
                    "diagonal_loading_factor": 1e-6,
                },
                "logging": {
                    "log_method_switches": True,
                    "log_shrinkage_intensity": True,
                    "log_condition_numbers": True,
                },
            },
        }
    }

    print("Sample robust portfolio configuration:")
    import yaml

    print(yaml.dump(robust_config, default_flow_style=False))

    print("\nThis configuration would:")
    print("1. Use robust mean-variance optimization as primary method")
    print("2. Apply Ledoit-Wolf shrinkage to stabilize covariance estimation")
    print("3. Monitor condition numbers and switch to HRP when threshold exceeded")
    print("4. Log all robustness decisions for transparency")


def main() -> None:
    log_path = setup_logging()
    logging.getLogger(__name__).info(
        "Robust weighting demo logs stored at %s", log_path
    )

    demonstrate_robust_weighting()
    demonstrate_config_usage()

    print("\n=== Summary ===")
    print("The robust weighting system provides:")
    print("• Automatic shrinkage (Ledoit-Wolf/OAS) for covariance stabilization")
    print("• Condition number monitoring with configurable thresholds")
    print(
        "• Safe mode fallback (HRP/Risk Parity/Diagonal Loading) for ill-conditioned matrices"
    )
    print("• Comprehensive logging of all robustness decisions")
    print("• Backwards compatibility with existing configurations")
    print("• Enhanced numerical stability for pathological inputs")


if __name__ == "__main__":
    main()
