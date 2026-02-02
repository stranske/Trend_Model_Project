# Monte Carlo Strategy Packs

This directory contains curated strategy packs intended for Monte Carlo scenario use.

## HF Equity Curated (hf_equity_curated.yml)

| Strategy | Intent | Rationale |
| --- | --- | --- |
| Rank_8_Equal_TightTurnover | Concentrated, low-turnover equity sleeve focused on top-ranked managers. | Use a small top-N set with tight turnover to represent a capacity-aware, high-conviction book. |
| Rank_12_RiskParity_ModerateTurnover | Diversified equity sleeve with risk parity weighting and moderate turnover. | Adds risk-based diversification while allowing some turnover to refresh the signal set. |
| Rank_16_HRP_LooseTurnover | Broad, diversified equity sleeve using HRP with looser turnover. | Captures a larger opportunity set while relying on HRP for stability. |
| TopPct_20_ScoreProp_TightTurnover | Percentile-based selection with score-proportional weights and tight turnover. | Keeps exposure proportional to conviction while limiting churn in the top quintile. |
| Threshold_ZScore_Bayes_ModerateTurnover | Z-score threshold selection with Bayesian score-proportional weighting. | Enforces a minimum signal quality while smoothing weights via shrinkage. |
| Rank_10_AdaptiveBayes_ModerateTurnover | Mid-sized top-N selection with adaptive Bayesian weighting. | Blends concentration with adaptive shrinkage to stabilize weights over time. |
| Random_12_Equal_LooseTurnover | Randomized selection baseline with equal weights and loose turnover. | Acts as a stress-test/control strategy for selection bias and turnover effects. |
| Manual_8_Equal_TightTurnover | Manual core list with equal weights and tight turnover. | Represents a discretionary or constrained core book with stable holdings. |
| Rank_20_ERC_LooseTurnover | Large top-N selection with ERC weighting and loose turnover. | Emphasizes diversification across a wider set while tolerating rebalancing. |
| Rank_14_RobustMV_TightTurnover | Top-N selection using robust mean-variance with tight turnover. | Targets risk-adjusted optimization while controlling trading activity. |
| Rank_12_RobustRiskParity_ModerateTurnover | Top-N selection using robust risk parity with moderate turnover. | Provides risk-balanced weights with added robustness to estimation noise. |
| All_Equal_LowTurnover_LowVol | Full-universe equal weight with low turnover and vol targeting. | Serves as a defensive baseline that limits trading and targets lower volatility. |
