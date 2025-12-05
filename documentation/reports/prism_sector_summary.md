# PRISM Engine: Sector Summary

## Sector Rankings (by Average Consensus Rank)

| Rank | Sector | Avg Rank | Best Indicator | Worst Indicator |
|------|--------|----------|----------------|-----------------|
| 1 | **Housing** | 14.6 | houst (14.3) | permit (15.0) |
| 2 | **Monetary/Fed** | 15.3 | m2sl (10.4) | t10y2y (18.4) |
| 3 | **Equity ETFs** | 15.4 | XLU_Close (12.4) | spy_spy (16.5) |
| 4 | **Inflation** | 15.4 | cpilfesl (14.5) | ppiaco (15.9) |
| 5 | **Financial Conditions** | 16.1 | nfci (15.6) | anfci (16.5) |
| 6 | **Real Economy** | 16.6 | indpro (16.6) | — |
| 7 | **Employment** | 16.8 | payems (15.3) | unrate (18.4) |
| 8 | **Bond ETFs** | 18.4 | lqd_lqd (16.8) | shy_shy (23.6) |
| 9 | **Commodity ETFs** | 18.8 | uso_uso (18.2) | gld_gld (19.5) |

---

## Asset Class Summary

| Asset Class | Avg Rank | Interpretation |
|-------------|----------|----------------|
| **Macro Indicators** | 15.1 | Core drivers |
| **Equities** | 15.4 | Surprisingly high (XLU effect) |
| **Rates/Yields** | 16.7 | Mid-tier importance |
| **Fixed Income ETFs** | 18.4 | Reactive, not driving |
| **Commodities** | 18.8 | Least connected |

---

## Lens-by-Lens Sector Preferences

Each lens has a distinct "philosophy" about which sectors matter:

### Housing Fans (5 lenses)
| Lens | Housing Rank | Rationale |
|------|--------------|-----------|
| DMD | 2.5 | Housing dominates dynamic modes |
| Magnitude | 3.0 | High variance sector |
| Influence | 3.5 | Network centrality |
| Wavelet | 4.5 | Multi-scale dynamics |
| TDA | 4.5 | Topologically central |

### Inflation Fans (3 lenses)
| Lens | Inflation Rank | Rationale |
|------|----------------|-----------|
| Mutual Info | 2.7 | Maximum shared information |
| Clustering | 6.3 | Groups with other fundamentals |
| Decomposition | 6.3 | Core structural component |

### Bond Fans (2 lenses)
| Lens | Bond Rank | Rationale |
|------|-----------|-----------|
| Transfer Entropy | 5.3 | Information flows through bonds |
| Network | 9.4 | Central network position |

### Equity Fans (1 lens)
| Lens | Equity Rank | Rationale |
|------|-------------|-----------|
| Granger | 6.0 | Markets Granger-cause everything |

### Employment Fans (1 lens)
| Lens | Employment Rank | Rationale |
|------|-----------------|-----------|
| Regime | 4.5 | Defines economic regimes |

---

## Equity ETF Deep Dive

### XLU (Utilities) — The Anomaly
**Consensus Rank: 12.4** (best equity)

| Loved By | Rank | Hated By | Rank |
|----------|------|----------|------|
| PCA | #1 | Magnitude | #27 |
| Anomaly | #3 | Network | #22 |
| Influence | #7 | Clustering | #18 |

**Why XLU ranks so high:** Utilities are the "bond proxy" equity sector. PCA sees it as a dominant principal component. Anomaly flags it as behaving unusually relative to other equities. It's structurally important because it bridges equity and rate sensitivity.

### SPY (S&P 500) — The Responder
**Consensus Rank: 16.5** (worst equity)

| Loved By | Rank | Hated By | Rank |
|----------|------|----------|------|
| Granger | #3 | Decomposition | #29 |
| Network | #4 | Regime | #27 |
| Transfer Entropy | #5 | TDA | #27 |

**Why SPY ranks lower:** Causality lenses love it (everything Granger-causes SPY). Structure lenses hate it (SPY is an output, not a fundamental). This confirms PRISM's thesis: markets are where signals aggregate, not where they originate.

### QQQ (Nasdaq) vs IWM (Small Caps)
Both rank ~16.5. Granger and Transfer Entropy love them (causal targets). Decomposition and Regime hate them (not structural drivers). The pattern is consistent: equity indices are information sinks, not sources.

---

## Bond ETF Rankings

| ETF | Rank | Notes |
|-----|------|-------|
| LQD (Corp IG) | 16.8 | Best bond — credit spreads matter |
| TIP (TIPS) | 16.8 | Tied — inflation expectations |
| TLT (Long Treasury) | 17.4 | Duration play |
| BND (Agg Bond) | 17.6 | Broad exposure |
| HYG (High Yield) | 17.6 | Credit risk |
| IEF (7-10yr Treasury) | 19.0 | Mid-duration |
| **SHY (Short Treasury)** | **23.6** | Least informative |

**Insight:** LQD and TIP rank highest because they contain **credit** and **inflation** information beyond pure rates. SHY ranks last because short-term Treasuries are nearly cash — they don't tell you much the Fed hasn't already said.

---

## Key Takeaways

1. **Housing punches above its weight** — Only 2 indicators but ranks #1 by sector. Five different lenses independently identify housing as central.

2. **XLU is the equity outlier** — Utilities rank higher than SPY, QQQ, or IWM. It's the bridge between equities and rates.

3. **Causality vs Structure split is real:**
   - Granger/Transfer Entropy → Equities matter (as information sinks)
   - TDA/Decomposition/Regime → Equities don't matter (not structural)

4. **Commodities are disconnected** — GLD, SLV, USO rank lowest. They march to their own drummer.

5. **Credit > Duration** — LQD beats TLT. Credit spreads contain more information than pure rate exposure.

6. **SHY is noise** — Rank 23.6. Short Treasuries tell you nothing the Fed hasn't already announced.
