# LangChain Integration Plan for Trend Model Project

> **Status**: Planning  
> **Created**: 2026-01-03  
> **Last Updated**: 2026-01-03

## Executive Summary

This document outlines a phased approach to integrating LangChain capabilities into the Trend Model Project. The goal is to enhance user interaction through natural language while maintaining the deterministic, auditable nature of the simulation pipeline.

---

## Phase Overview

| Phase | Feature | Feasibility | Priority | Effort |
|-------|---------|-------------|----------|--------|
| 1 | QC Documentation Generation | 🟢 High | High | 1-2 days |
| 1.5 | Statistical Validation Layer | 🟢 High | High | 2-3 days |
| 2 | Simulation-Specific Q&A | 🟢 High | High | 3-5 days |
| 3 | NL Configuration | 🟠 Low-Medium | Low | 1-2 weeks |

---

## Design Principles

### Provider Agnosticism

The implementation will abstract LLM provider details behind a common interface:

```python
# Proposed: src/trend_analysis/llm/providers.py
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        """Generate completion from messages."""
        pass
    
    @abstractmethod
    def structured_output(self, messages: list[dict], schema: type) -> object:
        """Generate structured output conforming to Pydantic schema."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI implementation (used for testing with existing secret)."""
    pass

class AnthropicProvider(LLMProvider):
    """Anthropic Claude implementation."""
    pass

class OllamaProvider(LLMProvider):
    """Local Ollama implementation for offline use."""
    pass
```

Environment variable `TREND_LLM_PROVIDER` controls which provider is used, defaulting to OpenAI when `OPENAI_API_KEY` is present.

---

## Complementary Analytical Tools

LLMs excel at interpretation and communication but are not the right tool for statistical validation. A robust analysis pipeline should use specialized tools **before** or **alongside** LLM evaluation.

### Pre-LLM Statistical Validation

Run these deterministic checks first—they catch issues an LLM might miss or hallucinate about:

| Tool Category | Purpose | Libraries | Example Use |
|---------------|---------|-----------|-------------|
| **Distribution Testing** | Verify returns match expected statistical properties | `scipy.stats` | Jarque-Bera normality test, Kolmogorov-Smirnov against target distribution |
| **Anomaly Detection** | Flag outlier periods/funds before interpretation | `scipy.stats`, `sklearn.ensemble.IsolationForest` | Z-score > 3 on period returns, isolation forest on multi-period patterns |
| **Stationarity Testing** | Check for structural breaks in time series | `statsmodels.tsa.stattools` | ADF test, KPSS test on rolling windows |
| **Correlation Analysis** | Validate diversification claims | `scipy.stats.spearmanr`, `numpy.corrcoef` | Correlation stability across regimes, correlation breakdown detection |

### Portfolio-Specific Analytics

These tools provide domain-specific validation:

| Tool | Purpose | Implementation |
|------|---------|----------------|
| **Risk Decomposition** | Validate risk attribution sums correctly | Already in codebase (`risk.py`) |
| **Factor Exposure** | Check unintended factor tilts | `statsmodels.regression.linear_model.OLS` for factor regression |
| **Drawdown Analysis** | Structured drawdown decomposition | Custom (extend existing `metrics.py`) |
| **Turnover Validation** | Verify turnover calculations match constraints | Already tested in wiring tests |

### Proposed Tool Pipeline

```
Simulation Output
       │
       ▼
┌──────────────────┐
│ Statistical      │  ← scipy.stats, statsmodels
│ Validation       │  ← Deterministic pass/fail checks
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Anomaly          │  ← Flag unusual patterns
│ Detection        │  ← Structured anomaly report
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Economic         │  ← Rule-based sanity checks
│ Sanity Checks    │  ← "Sharpe > 3 is suspicious"
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LLM              │  ← Receives pre-validated data
│ Interpretation   │  ← PLUS validation flags/warnings
└──────────────────┘
```

### Economic Sanity Check Rules (Pre-LLM)

Hard-coded rules that flag implausible results before LLM sees them:

```python
# Proposed: src/trend_analysis/validation/sanity_checks.py
SANITY_RULES = {
    "sharpe_too_high": lambda s: s.sharpe > 3.0,  # Likely data issue
    "sharpe_negative_with_positive_return": lambda s: s.sharpe < 0 and s.return_ann > 0,
    "volatility_implausibly_low": lambda s: s.volatility < 0.01,  # < 1% annual vol
    "volatility_implausibly_high": lambda s: s.volatility > 1.0,  # > 100% annual vol
    "max_drawdown_exceeds_100pct": lambda s: s.max_drawdown < -1.0,
    "turnover_exceeds_constraint": lambda s, cfg: s.turnover > cfg.max_turnover * 1.01,
    "weights_dont_sum_to_one": lambda w: abs(sum(w.values()) - 1.0) > 0.01,
    "negative_weights_with_long_only": lambda w, cfg: cfg.long_only and any(v < -0.001 for v in w.values()),
}
```

### Benefits of Tool Pipeline Approach

| Benefit | Without Pre-Processing | With Pre-Processing |
|---------|----------------------|---------------------|
| **Accuracy** | LLM might miss statistical issues | Statistical tests catch them deterministically |
| **Cost** | Full context sent to LLM every time | Only anomalies/summaries sent |
| **Auditability** | "The LLM said it was fine" | Deterministic test logs + LLM interpretation |
| **Speed** | LLM latency on every check | Fast local checks, LLM only for interpretation |
| **Reproducibility** | LLM outputs vary | Statistical checks are deterministic |

### Additional Tools to Consider

| Tool | Use Case | When to Add |
|------|----------|-------------|
| **QuantLib** | Option pricing validation if derivatives added | Future scope |
| **PyPortfolioOpt** | Cross-validate optimization results | If optimization complexity increases |
| **empyrical** | Alternative risk metrics implementation | Cross-validation of existing metrics |
| **arch** | GARCH modeling for volatility validation | If regime switching becomes complex |

### Recommendation

**Phase 1.5**: Before implementing LLM Q&A (Phase 2), add a structured validation layer:
1. Statistical validation module (`src/trend_analysis/validation/`)
2. Sanity check rules (economic intuition encoded as code)
3. Anomaly detection for multi-period results
4. Validation report structure that feeds into LLM context

This ensures LLM receives pre-validated data with flags, rather than raw results it might misinterpret.

---

## Phase 1: QC Documentation Generation

### Objective

Generate comprehensive documentation proving that each app setting produces meaningful, economically-intuitive output changes.

### Current State

- ✅ `test_settings_wiring.py` tests 33 settings at 100% effectiveness
- ✅ `comparison_export.py` provides full Excel workbook generation
- ⚠️ 37 settings lack wiring tests (see Appendix A)

### Deliverables

1. **Excel workbooks** (human consumption): One per setting category with:
   - Baseline vs. test configuration comparison
   - Winner indicators with conditional formatting
   - Period-by-period breakdown
   - Config snapshots

2. **JSON reports** (machine/LLM consumption):
   - Structured metrics with deltas
   - Economic intuition validation flags
   - Anomaly indicators

3. **Markdown summary** (documentation):
   - Category-level pass rates
   - Individual setting results
   - Known gaps and limitations

### Implementation

Extend existing infrastructure—no new dependencies required for Phase 1.

---

## Phase 2: Simulation-Specific Q&A

### Objective

Enable natural language interaction with simulation results to understand *why* the simulation produced specific outcomes.

### Simulation-Appropriate Questions

Unlike generic finance Q&A, questions must relate to the simulation mechanics:

| Question Type | Example | Data Source |
|---------------|---------|-------------|
| **Selection Logic** | "Why was Manager X excluded in Period 3?" | z-scores, ranking, threshold comparisons |
| **Constraint Impact** | "Which constraint was binding on the final weights?" | pre/post-constraint weight deltas |
| **Turnover Analysis** | "What's driving the high turnover this period?" | selection changes, weight adjustments |
| **Volatility Scaling** | "Why did Manager Y receive 2.3x scaling?" | vol estimates, target vol, floor |
| **Performance Attribution** | "How much did the max_weight cap cost in returns?" | constrained vs unconstrained backtest |
| **Config Sensitivity** | "Would selecting 8 instead of 10 funds have changed outcomes?" | requires re-simulation |

### Questions That Don't Fit (Avoid These)

- Generic market analysis ("What drove equity returns in Q3?")
- Forward-looking predictions ("Will this strategy work next year?")
- External data questions ("What's the current risk-free rate?")
- Questions requiring data not in the simulation context

### Architecture

```python
# Proposed: src/trend_analysis/llm/simulation_qa.py
class SimulationQASession:
    """Q&A session grounded in simulation results."""
    
    def __init__(self, result: RunResult, config: dict, provider: LLMProvider):
        self.result = result
        self.config = config
        self.provider = provider
        self.context = self._build_context()
    
    def _build_context(self) -> str:
        """Build structured context from simulation outputs.
        
        Includes:
        - Configuration parameters used
        - Period-by-period selection decisions with z-scores
        - Constraint binding indicators
        - Weight allocation pre/post constraints
        - Turnover calculations
        - Performance metrics
        """
        pass
    
    def ask(self, question: str) -> SimulationAnswer:
        """Answer a question about the simulation.
        
        Returns structured answer with:
        - Natural language response
        - Supporting data references
        - Confidence indicator
        - Suggested follow-up questions
        """
        pass
    
    def generate_summary(self) -> str:
        """Generate executive summary of simulation run."""
        pass
```

### Context Window Management

Simulation results can be large. Strategy:
1. Always include: config, summary metrics, selected funds
2. On-demand retrieval: period details, individual fund metrics
3. Tool-calling pattern: LLM requests specific data as needed

---

## Phase 3: Natural Language Configuration

### ⚠️ Feasibility Assessment: LOW-MEDIUM

This phase carries meaningful implementation risk and should only proceed after Phases 1-2 prove successful.

### Why This Is Harder

| Concern | Mitigation |
|---------|------------|
| LLM may generate invalid parameter values | Pydantic validation with strict schemas |
| Parameter combinations may be invalid | Cross-parameter validation rules |
| User intent may be ambiguous | Confirmation step showing interpreted changes |
| Changes may produce unexpected results | Mandatory preview with comparison to baseline |

### Failure Modes and Prevention

**For deterministic config (simulation parameters):**

Failures should *never* be silent. The simulation is deterministic—given valid config, it produces deterministic output. Invalid config must be caught at validation time:

```python
class ConfigChange(BaseModel):
    """Validated configuration change."""
    parameter: str
    old_value: Any
    new_value: Any
    
    @model_validator(mode='after')
    def validate_parameter(self):
        if self.parameter not in ALLOWED_PARAMETERS:
            raise ValueError(f"Unknown parameter: {self.parameter}")
        spec = PARAMETER_SPECS[self.parameter]
        if not spec.validate(self.new_value):
            raise ValueError(f"Invalid value for {self.parameter}: {self.new_value}")
        return self
```

**For Q&A interaction config (non-deterministic):**

This requires more careful design because the LLM's interpretation of questions/answers is inherently variable. Approach:
1. Store conversation templates as structured JSON
2. Validate template structure, not content
3. Allow human review before execution
4. Log all interactions for audit

### Required Research (Before Implementation)

Before implementing Phase 3, conduct research into LangChain patterns that have shown success:

#### Research Questions

1. **Structured Output Patterns**: How do successful implementations ensure LLM outputs conform to schemas?
   - LangChain's `with_structured_output()` reliability
   - Pydantic integration best practices
   - Retry strategies for malformed outputs

2. **Tool-Calling for Config**: How do production systems use tool-calling to modify application state?
   - Guard rails preventing invalid operations
   - Rollback mechanisms
   - Audit logging patterns

3. **Similar Domain Implementations**: What patterns exist for:
   - Financial simulation configuration via NL
   - Parameter optimization with human-in-the-loop
   - Conversational configuration builders

4. **Failure Recovery**: How do robust implementations handle:
   - Ambiguous user intent
   - Partial config modifications
   - Conflicting parameter changes

#### Research Sources

- LangChain documentation and cookbook
- LangSmith traces from production deployments
- Academic papers on LLM-driven configuration
- Open-source projects with similar patterns (identified during research)

#### Research Deliverable

A findings document addressing each question with:
- Pattern description
- Success/failure examples
- Applicability to Trend Model Project
- Recommended approach or "do not proceed" determination

---

## Dependencies

```toml
# pyproject.toml [project.optional-dependencies]
llm = [
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-anthropic>=0.2.0",  # Optional
    "tiktoken>=0.8.0",
]

# Statistical validation tools (most already available via scipy/numpy)
# These are already in core dependencies:
# - scipy (distribution tests, correlation analysis)
# - numpy (numerical operations)
# - pandas (data manipulation)
# Optional additions for advanced validation:
validation = [
    "statsmodels>=0.14.0",  # Time series tests, regression
    "arch>=7.0.0",          # GARCH models (optional, for regime validation)
]
```

---

## Testing Strategy

### Phase 1 Testing
- Unit tests for workbook generation
- Snapshot tests for JSON structure
- CI job to regenerate QC docs on release

### Phase 1.5 Testing
- Unit tests for each sanity check rule
- Property-based tests for statistical validation (hypothesis)
- Known-anomaly detection tests (inject known issues, verify detection)
- Performance benchmarks (validation should be fast)

### Phase 2 Testing
- Mock LLM responses for deterministic tests
- Integration tests with real LLM (marked slow, optional)
- Golden-file tests for context building

### Phase 3 Testing
- Extensive validation testing
- Fuzzing for edge cases
- Human review gate for production

---

## Appendix A: Settings Without Wiring Tests

The following 37 settings were identified as lacking systematic wiring validation:

```
bottom_k, condition_threshold, cooldown_periods, date_mode, end_date,
info_ratio_benchmark, long_only, max_active_positions, max_changes_per_period,
min_history_periods, min_tenure_periods, min_weight_strikes, multi_period_enabled,
preset, rebalance_freq, regime_enabled, regime_proxy, report_attribution,
report_benchmark_comparison, report_concentration, report_factor_exposures,
report_regime_analysis, report_rolling_metrics, rf_override_enabled, safe_mode,
shrinkage_enabled, shrinkage_method, start_date, sticky_add_periods,
sticky_drop_periods, trend_lag, trend_min_periods, trend_vol_adjust,
trend_vol_target, trend_window, trend_zscore, z_entry_hard, z_exit_hard
```

Phase 1 should prioritize adding wiring tests for settings that affect simulation outcomes (excluding `report_*` toggles which only affect output format).

---

## Appendix B: Environment Configuration

```bash
# Required for Phase 2+
export OPENAI_API_KEY="sk-..."  # Existing secret

# Optional provider override
export TREND_LLM_PROVIDER="openai"  # openai | anthropic | ollama

# For local development with Ollama
export OLLAMA_BASE_URL="http://localhost:11434"
```

---

## Next Steps

1. ✅ Document plan (this document)
2. ⏳ Address other pending issues
3. ⏳ Implement Phase 1 (QC documentation)
4. ⏳ Implement Phase 2 (Simulation Q&A)
5. ⏳ Conduct Phase 3 research
6. ⏳ Decision gate: proceed with Phase 3 or defer
