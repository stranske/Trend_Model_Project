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

| Identifier | Name | Intent | Rationale |
| --- | --- | --- | --- |
| macro_rank_8_equal_tight_turnover | Rank_8_Equal_TightTurnover | Concentrated macro sleeve selecting top-ranked managers with equal weights and tight turnover. | Preserves high conviction while reducing implementation drift from frequent rebalances. |
| macro_toppct_25_riskparity_moderate_turnover | TopPct_25_RiskParity_ModerateTurnover | Top-quartile macro selection with risk parity weighting and moderate turnover limits. | Balances concentration in stronger signals with risk-based diversification and practical trading cadence. |
| macro_threshold_zscore_hrp_bayes_moderate_turnover | Threshold_ZScore_HRP_Bayes_ModerateTurnover | Threshold-based macro selection with Bayesian score-proportional weighting and HRP-style diversification intent. | Enforces a minimum signal bar while shrinkage dampens noisy estimates in regime-sensitive macro universes. |
| macro_rank_16_hrp_loose_turnover | Rank_16_HRP_LooseTurnover | Broad rank-based macro sleeve using HRP and looser turnover. | Expands breadth across macro opportunities while preserving clustering-aware risk control. |
| macro_rank_20_erc_loose_turnover | Rank_20_ERC_LooseTurnover | Diversified top-N macro sleeve with equal risk contribution and loose turnover. | Distributes risk across a broader universe to avoid concentration in correlated macro themes. |
| macro_rank_14_robustmv_tight_turnover | Rank_14_RobustMV_TightTurnover | Robust mean-variance long/short macro sleeve with tight turnover controls. | Targets risk-adjusted optimization while allowing controlled short exposure under estimation uncertainty. |
| macro_rank_12_robustriskparity_moderate_turnover | Rank_12_RobustRiskParity_ModerateTurnover | Top-N macro sleeve with robust risk parity and moderate turnover constraints. | Prioritizes stable risk budgets under noisy covariance estimates and changing macro regimes. |
| macro_random_12_equal_loose_turnover | Random_12_Equal_LooseTurnover | Randomly selected macro baseline with equal weights and loose turnover. | Provides a stress-test benchmark to separate true signal value from selection luck. |
| macro_manual_8_equal_tight_turnover | Manual_8_Equal_TightTurnover | Fixed manual macro core book with equal sizing and tight turnover. | Represents a discretionary anchor sleeve where holdings are policy-driven rather than score-driven. |
| macro_all_equal_low_turnover_low_vol | All_Equal_LowTurnover_LowVol | Full-universe defensive macro sleeve with low turnover and volatility targeting. | Acts as a conservative reference allocation emphasizing stability and broad diversification. |
