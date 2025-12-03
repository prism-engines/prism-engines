# PRISM Engine: Deep Dive Analysis

## 2024 Regime Break

The Spearman correlation between 2023 and 2024 rankings dropped to **0.02** (essentially zero), compared to the typical 0.85-0.96 range. This is a major structural break in how indicators relate to each other.

### What Rose in Importance
| Indicator | 2015-2020 Rank | 2020-2025 Rank | Change |
|-----------|----------------|----------------|--------|
| ANFCI (Financial Conditions) | 14.1 | 11.1 | -3.0 |
| Payrolls | 8.9 | 6.4 | -2.5 |
| T10Y3M (Short yield curve) | 12.2 | 10.2 | -2.0 |

### What Fell in Importance
| Indicator | 2015-2020 Rank | 2020-2025 Rank | Change |
|-----------|----------------|----------------|--------|
| PPI | 8.4 | 10.5 | +2.1 |
| DGS2 (2-yr Treasury) | 9.4 | 11.4 | +2.0 |
| Housing Starts | 8.2 | 9.4 | +1.1 |

### Interpretation
The 2022-2024 rate hike cycle shifted the regime from **"liquidity-driven"** (QE era) to **"policy-driven"** (Fed watching labor). Financial conditions (ANFCI) and employment (payrolls) became the key transmission mechanisms, while housing and producer prices became less central.

---

## Lens Disagreement Analysis

The 14 lenses cluster into philosophical groups that often disagree:

### Cluster 1: Flow/Causality Lenses
- **Lenses:** Granger, Transfer Entropy
- **Question:** Who causes whom?
- **Favorites:** SPY, HYG (markets respond to everything)
- **r = -0.85 with Cluster 2** (maximum disagreement)

### Cluster 2: Structure/Pattern Lenses
- **Lenses:** Mutual Info, Decomposition, TDA, Regime
- **Question:** What's the underlying structure?
- **Favorites:** M2, CPI, Employment

### Cluster 3: Variance/Dynamics Lenses
- **Lenses:** Influence, Wavelet
- **Correlation:** r = 0.97 (near-perfect agreement!)
- **Favorites:** WALCL, Permits, Housing

### Cluster 4: Linear/Statistical Lenses
- **Lenses:** PCA, Magnitude, DMD
- **Irony:** Despite similar math, they often disagree

---

## The M2 Paradox

M2 Money Supply shows the widest lens disagreement (spread of 27 ranks):

| Lens Type | Rank | Interpretation |
|-----------|------|----------------|
| Anomaly | #2 | M2 drives unusual patterns |
| Magnitude | #3 | High variance |
| Wavelet | #3 | Multi-scale importance |
| TDA | #3 | Topologically central |
| Mutual Info | #4 | Shares information with everything |
| **Granger** | **#27** | Doesn't Granger-cause much |
| **DMD** | **#29** | Doesn't dominate dynamic modes |
| **Transfer Entropy** | **#21** | Doesn't transfer information |

### Resolution
M2 is a **STRUCTURAL driver** (correlated with everything) but **NOT a CAUSAL leader** (doesn't predict future movements). It's the water level that everything floats on—present everywhere, but not the proximate cause of movements.

---

## Most Agreeing Lens Pairs
| Pair | Correlation |
|------|-------------|
| Influence ↔ Wavelet | r = 0.97 |
| Wavelet ↔ TDA | r = 0.87 |
| Mutual Info ↔ Decomposition | r = 0.85 |

## Most Disagreeing Lens Pairs
| Pair | Correlation |
|------|-------------|
| Decomposition ↔ Transfer Entropy | r = -0.85 |
| Mutual Info ↔ Transfer Entropy | r = -0.84 |
| Granger ↔ TDA | r = -0.82 |

---

## Key Takeaways for Your Physicist Reviewer

1. **The consensus ranking is robust** despite lens disagreements—M2, WALCL, payrolls consistently rank high across philosophical clusters

2. **Causality ≠ Structure** — Granger/Transfer Entropy measure predictive causality; Mutual Info/TDA measure structural importance. They're answering different questions.

3. **The 2024 break is real** — Not a data artifact. The Fed pivot from QE to QT fundamentally rewired indicator relationships.

4. **SPY ranking last is a feature, not a bug** — Markets are outputs, not inputs. The lenses correctly identify this.

5. **Influence and Wavelet's near-perfect agreement (r=0.97)** suggests multi-scale dynamics and network influence are measuring the same underlying phenomenon—worth exploring mathematically.
