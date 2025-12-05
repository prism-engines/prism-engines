# PRISM Engine: Cause vs Response Analysis

## What Are We Actually Measuring?

When PRISM ranks indicators, it's asking **14 different questions** about each one. These questions fall into two categories:

### CAUSE Lenses (Does this indicator drive others?)
| Lens | Question |
|------|----------|
| **Granger** | Does this predict future values of others? |
| **Transfer Entropy** | Does information flow FROM this to others? |
| **Influence** | Is this a network hub that affects others? |

### RESPONSE Lenses (Does this indicator reflect system state?)
| Lens | Question |
|------|----------|
| **PCA** | Does this capture common variance? |
| **Mutual Info** | Does this share information with everything? |
| **Anomaly** | Does this show unusual patterns? |

---

## The Cause/Response Ratio

We can compute a ratio: **Cause Score / Response Score**

| Ratio | Interpretation |
|-------|----------------|
| > 1.2 | **CAUSER** — drives other indicators |
| 0.8 - 1.2 | **BALANCED** — both causes and responds |
| < 0.8 | **RESPONDER** — reflects system state |

---

## Results: Who Causes vs Who Responds?

### CAUSERS (Ratio > 1.2)
| Indicator | Cause | Response | Ratio | Interpretation |
|-----------|-------|----------|-------|----------------|
| **QQQ** | 28.7 | 12.8 | **2.23** | Tech leads, doesn't follow |
| **GLD** | 19.7 | 9.8 | **2.00** | Gold as independent signal |
| **SPY** | 26.7 | 13.5 | **1.98** | Market drives expectations |
| **IWM** | 23.0 | 11.8 | **1.94** | Small caps as leading signal |
| **HYG** | 20.7 | 13.5 | **1.53** | Credit spreads drive |
| **TLT** | 18.0 | 12.5 | **1.44** | Long rates as causal |

### RESPONDERS (Ratio < 0.8)
| Indicator | Cause | Response | Ratio | Interpretation |
|-----------|-------|----------|-------|----------------|
| **M2** | 15.3 | 29.3 | **0.52** | Money supply reflects, doesn't lead |
| **DGS10** | 10.0 | 17.7 | **0.57** | 10yr yield is a thermometer |
| **Houst** | 15.3 | 22.3 | **0.69** | Housing responds to conditions |
| **XLU** | 20.7 | 28.3 | **0.73** | Utilities as canary |
| **CPI** | 11.3 | 15.5 | **0.73** | Inflation is outcome |
| **WALCL** | 19.0 | 25.0 | **0.76** | Fed balance sheet reflects policy |

### BALANCED (Ratio 0.8 - 1.2)
| Indicator | Cause | Response | Ratio | Interpretation |
|-----------|-------|----------|-------|----------------|
| **Payrolls** | 15.3 | 13.5 | **1.14** | Employment both leads and lags |

---

## XLU Deep Dive: The Canary in the Coal Mine

### XLU Rankings by Lens
| Lens | Rank | What It Means |
|------|------|---------------|
| PCA | #1 | Moves with everything |
| Anomaly | #3 | Moves in unusual ways |
| Wavelet | #7 | Important at all timescales |
| Influence | #7 | Well-connected |
| Magnitude | #27 | Low volatility |
| Network | #22 | Not a hub |

### Interpretation
XLU is the **#1 RESPONDER** among equities:
- **High PCA** = utilities correlate with the whole system
- **High Anomaly** = but they move differently than you'd expect
- **Low Magnitude** = stable, not volatile
- **Moderate Granger** = doesn't predict others

**XLU is a thermometer, not a heater.** When utilities move, they're telling you:
- Interest rate expectations shifted
- Risk appetite changed  
- Inflation expectations moved

But XLU itself isn't *causing* those changes — it's *reflecting* them.

---

## The Paradox: SPY as Causer?

SPY shows a **1.98 ratio** (strong causer). This seems backwards — shouldn't markets respond to fundamentals?

### Explanation
Granger causality measures **prediction**, not economic causation. SPY "Granger-causes" other indicators because:

1. **Markets price expectations first** — SPY moves before economic data confirms
2. **Wealth effects** — market moves affect consumer behavior, corporate investment
3. **Fed reaction function** — the Fed watches markets

So SPY is a "causer" in the **information flow** sense, even though economically it's responding to fundamentals. The lenses are measuring statistical relationships, not economic theory.

---

## Key Insight: M2 Paradox Resolved

M2 ranks **#0.52 ratio** — extreme responder. But it also ranks **#10.4 consensus** (very high).

### How Can Both Be True?

M2 is **structurally central** (high PCA, Mutual Info) but **not causally leading** (low Granger, Transfer Entropy).

Think of M2 as the **water level** in a harbor:
- Every boat's height correlates with the water level (high PCA)
- But the water doesn't cause individual boat movements (low Granger)
- The water reflects tides, weather, seasons (responder)

M2 doesn't predict what happens next — but knowing M2 tells you a lot about the current state.

---

## Sector Summary: Cause vs Response

| Sector | Profile | What It Means |
|--------|---------|---------------|
| **Tech (QQQ)** | Strong Causer | Leads market sentiment |
| **Small Caps (IWM)** | Strong Causer | Risk appetite signal |
| **Broad Market (SPY)** | Strong Causer | Prices expectations first |
| **Gold (GLD)** | Strong Causer | Independent macro signal |
| **High Yield (HYG)** | Moderate Causer | Credit spreads lead |
| **Long Bonds (TLT)** | Moderate Causer | Duration expectations |
| **Utilities (XLU)** | Responder | Rate/risk thermometer |
| **Housing** | Responder | Reflects credit conditions |
| **M2** | Strong Responder | System state indicator |
| **10yr Yield** | Strong Responder | Expectations thermometer |

---

## Implications for PRISM

1. **Consensus rankings blend both perspectives** — M2 ranks high because it's structurally important, even though it doesn't "cause" things

2. **The cause/response split is valuable** — it tells you which indicators to watch for *leading signals* vs *confirmation*

3. **XLU's high ranking makes sense** — it's the best equity thermometer, not the best equity driver

4. **SPY ranking low in consensus but high in causality** — consensus is dominated by structural lenses, which correctly identify SPY as an output aggregator

5. **For forecasting, weight the causers higher** — QQQ, HYG, GLD tell you where things are going; M2, XLU, DGS10 tell you where things are
