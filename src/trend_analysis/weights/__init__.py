"""Risk-based weight engine implementations.

The risk-oriented engines are deliberately distinct:

* :class:`RiskParity` is the inexpensive inverse-volatility policy (also
  registered as ``vol_inverse``); it ignores off-diagonal covariance.
* :class:`HierarchicalRiskParity` clusters correlated assets and recursively
  bisects cluster variance.
* :class:`RobustRiskParity` uses inverse volatility after explicit covariance
  diagnostics and diagonal-loading repairs.

They share an objective family, not an interchangeable implementation
contract. Keeping separate registry entries preserves the choice between
speed, correlation-aware clustering, and observable numerical repair.
"""

from .equal_risk_contribution import EqualRiskContribution, EqualRiskContributionPolicy
from .hierarchical_risk_parity import HierarchicalRiskParity
from .risk_parity import RiskParity
from .robust_weighting import RobustMeanVariance, RobustRiskParity

__all__ = [
    "RiskParity",
    "HierarchicalRiskParity",
    "EqualRiskContribution",
    "EqualRiskContributionPolicy",
    "RobustMeanVariance",
    "RobustRiskParity",
]
