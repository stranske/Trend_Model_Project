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

## HF Macro Curated (hf_macro_curated.yml)

| Strategy | Intent | Rationale |
| --- | --- | --- |
| Macro_Rank_6_Equal_TightTurnover | Concentrated macro sleeve with equal weights and tight turnover. | Keeps exposure focused on the strongest signals while limiting churn. |
| Macro_TopPct_25_RiskParity_ModerateTurnover | Top-quartile macro selection with risk parity weights. | Balances risk across the highest conviction set with moderate turnover. |
| Macro_Threshold_ZScore_HRP_ModerateTurnover | Threshold-based macro sleeve using HRP weighting. | Filters for signal strength while relying on HRP for stability. |
| Macro_Rank_10_ERC_LooseTurnover | Broader top-N macro selection with ERC weighting. | Adds diversification while allowing higher refresh rates. |
| Macro_Rank_12_RobustMV_TightTurnover | Robust mean-variance macro sleeve with tight turnover. | Uses robust optimization while capping trading activity. |
| Macro_Rank_8_RobustRiskParity_ModerateTurnover | Top-N macro selection with robust risk parity. | Controls risk balance with moderate turnover. |
| Macro_Threshold_BayesScoreProp_ModerateTurnover | Threshold selection with Bayesian score-proportional weights. | Smooths weights while enforcing signal quality. |
| Macro_Random_8_Equal_LooseTurnover | Randomized macro baseline with equal weights. | Provides a stress-test/control strategy for selection bias. |
| Macro_Manual_6_Equal_TightTurnover | Manual macro core list with equal weights and tight turnover. | Represents a discretionary core macro book with stable holdings. |
| Macro_All_Equal_LowTurnover_LowVol | Full-universe macro sleeve with low turnover and vol targeting. | Defensive baseline that limits trading and volatility. |
