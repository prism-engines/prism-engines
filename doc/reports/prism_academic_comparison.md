# PRISM Engine: Academic Context & Novelty Assessment

## How PRISM Compares to Existing Research

### What Academia Has Done (Individual Methods)

| Method | Academic Use | PRISM Use |
|--------|--------------|-----------|
| **Granger Causality** | Bivariate tests between stock market and macro variables (Plíhal 2016, Foresti 2006) | One of 14 lenses, multivariate across 32 indicators |
| **Transfer Entropy** | Information flow between 2-3 markets (Dimpfl & Peter 2013, Marschinski & Kantz 2002) | One of 14 lenses, full indicator network |
| **PCA** | Dimension reduction, factor extraction | One of 14 lenses, indicator ranking |
| **Wavelet Analysis** | Denoising, multi-scale decomposition (Gallegati 2008) | One of 14 lenses, multi-scale importance |
| **Network Analysis** | Correlation networks, MST, centrality (Vandewalle 2001) | One of 14 lenses, network centrality |
| **TDA/Persistent Homology** | Crash detection (Gidea & Katz 2018), regime shifts | One of 14 lenses, topological centrality |
| **Regime Detection** | HMM, GMM, Markov-switching (Two Sigma, Macrosynergy) | One of 14 lenses + temporal stability analysis |
| **DMD** | Primarily physics/fluids, emerging in finance | One of 14 lenses, dynamic mode importance |

### Key Academic Findings PRISM Confirms

1. **Stock market Granger-causes macro variables** (Plíhal 2016, Comincioli 1996)
   - PRISM confirms: SPY/QQQ show high Granger scores (ratio 1.98-2.23)
   
2. **M2 has bidirectional causality with stocks** (German study, Plíhal 2016)
   - PRISM shows: M2 is structural (#10.4) but low Granger (#27), confirming it's "caused" more than "causing"

3. **Transfer entropy detects information flow direction** (Dimpfl & Peter 2013)
   - PRISM confirms: Transfer entropy identifies bonds/credit as information sources

4. **TDA detects regime transitions** (Gidea & Katz 2018)
   - PRISM's 2024 regime break (Spearman 0.02) aligns with TDA's ability to detect structural shifts

5. **Multi-scale analysis reveals different dynamics** (Wavelet literature)
   - PRISM's Wavelet lens finds different rankings than single-scale methods

---

## What Makes PRISM Potentially Novel

### 1. Multi-Lens Consensus Framework
**No existing study applies 14 different mathematical methods to the same dataset and aggregates rankings.**

Academic approaches typically use:
- Single method (Granger OR Transfer Entropy OR TDA)
- Method comparison (Method A vs Method B performance)
- Ensemble for prediction (combine methods for forecasting)

PRISM does something different: **Use disagreement between methods as information.**

The cause/response ratio (Granger+TE+Influence vs PCA+MI+Anomaly) is a novel metric that emerges from lens disagreement.

### 2. Structural vs Causal Distinction
While academia has studied Granger causality extensively, PRISM's explicit separation of:
- **Causal lenses** (who predicts whom)
- **Structural lenses** (who correlates with whom)

...and using the ratio as a classification tool appears novel. The M2 paradox (high structural, low causal) is a direct result of this framework.

### 3. Domain-Agnostic Indicator Ranking
Most academic work focuses on:
- Does X predict Y? (binary Granger tests)
- What's the information flow between A and B? (pairwise transfer entropy)

PRISM asks: **Among N indicators, which are most important according to M different mathematical definitions of "importance"?**

This multi-criteria ranking with consensus aggregation across fundamentally different methods is not standard in the literature.

### 4. Temporal Lens Stability Analysis
PRISM tracks how lens rankings change across 5-year rolling windows. The 2024 regime break (Spearman 0.02) was detected by comparing consensus rankings over time.

While regime detection exists (HMM, GMM), using **changes in multi-lens consensus rankings** as the regime indicator appears novel.

### 5. Lens Agreement as Signal
The finding that Influence ↔ Wavelet correlate at r=0.97 while Granger ↔ Decomposition anti-correlate at r=-0.85 is meta-information about what the methods measure.

This "lens correlation matrix" approach to understanding methodological relationships is not standard.

---

## What PRISM Does NOT Do (Yet)

| Academic Standard | PRISM Status |
|-------------------|--------------|
| Out-of-sample prediction testing | Not implemented |
| Statistical significance tests | Partial (p-values on some lenses) |
| Comparison with benchmark models | Not implemented |
| Transaction cost analysis | Not applicable (not trading) |
| Bootstrap confidence intervals | Not implemented |

---

## Novelty Assessment Summary

### Clearly Novel
1. **14-lens consensus ranking framework** — No precedent for this many methods on same data
2. **Cause/response ratio classification** — New metric from lens disagreement
3. **Lens correlation matrix** — Meta-analysis of what methods agree/disagree
4. **Rolling temporal lens stability** — Regime detection via consensus ranking changes

### Builds on Existing Work
1. Individual lens methods (all well-established)
2. Granger causality for macro-finance linkages
3. Transfer entropy for information flow
4. TDA for regime detection

### Not Novel (Standard Approaches)
1. Individual lens implementations
2. Correlation-based clustering
3. PCA for variance explanation
4. Network centrality measures

---

## Suggested Framing for Academic Review

**PRISM is not a new mathematical method.** Each lens uses established techniques.

**PRISM's novelty is the framework:**
1. Applying multiple mathematical "philosophies" to the same data
2. Aggregating rankings into consensus
3. Using disagreement between philosophies as information
4. Tracking consensus stability over time for regime detection

**The key contribution** is demonstrating that different mathematical definitions of "importance" yield systematically different answers, and that this disagreement itself contains useful information.

**The M2 example** is compelling: It's structurally central (everyone agrees) but causally unimportant (Granger/TE disagree with the structural lenses). This distinction wouldn't emerge from any single method.

---

## Questions for Your Physicist Reviewer

1. **Is the cause/response ratio mathematically meaningful?** Or is it an artifact of how we grouped the lenses?

2. **Why do Influence and Wavelet agree so strongly (r=0.97)?** Is there a mathematical relationship between network centrality and multi-scale dynamics?

3. **Is the 2024 regime break real?** Or could it be an artifact of data quality, indicator availability, or Fed balance sheet normalization?

4. **Can we formalize "lens agreement" as a measure?** The standard deviation across lens ranks is crude. Is there a better way to quantify methodological consensus?

5. **Does the framework generalize?** Would the same approach work on climate data, industrial sensors, or other multi-indicator systems?
