# PRISM Engine Complement Addendum: HMM + Koopman/DMD

Date: 2025-12-17

This addendum specifies two math engines to include in the PRISM engine complement:
(1) Hidden Markov Models (HMM) for regime inference, and
(2) Koopman / Dynamic Mode Decomposition (DMD) for dynamic mode structure.

This is not a mandate to run dozens of engines on every dataset.
The goal is to provide a modular, on-demand engine set so AI agents can select and
rank the most relevant tools for the research objective and the observed data.

---

## 1. Design Intent (Scope and Philosophy)

PRISM should support a growing set of analysis engines, but with the following rules:

1) Modular, on-demand execution
- Engines run one-at-a-time or as a small selected subset.
- No "compute everything" runs by default.

2) Minimal persistence, maximal reproducibility
- Do not bury Python calculations into the database purely for bulk production.
- Store only what is necessary to reproduce findings, verify claims, and support downstream steps.

3) Agent-guided selection
- AI agents evaluate engine outputs and recommend which engines matter for the current question:
  - Indicator alone (unbounded)
  - System geometry (lens outputs)
  - Indicator behavior inside system geometry (bounded)

4) Observability and auditability
- Every engine run must produce a clear run log and a compact "run summary" record:
  engine, params, data slice, windowing, runtime, success/failure, key metrics.

---

## 2. Engine: Hidden Markov Model (HMM)

### 2.1 Purpose
HMM provides probabilistic regime inference from time series behavior. It is useful for:
- detecting discrete latent states (regimes)
- estimating regime membership over time
- comparing "statistical regimes" vs "geometric regimes" produced by PRISM lenses

HMM is especially valuable as a cross-check engine:
- If PRISM geometry indicates regime change, HMM can provide an independent regime signal.
- If HMM indicates regime change without PRISM confirmation, that divergence is itself informative.

### 2.2 Inputs
HMM can operate on:
A) Indicator series (per-indicator)
- returns, differences, z-scores (for HMM only), volatility proxies, etc.

B) Lens outputs (system-level)
- geometry metrics over time (coherence, angles, distance measures, etc.)

C) Residual / bounded features
- indicator features conditioned on geometry state, or deviations from expected geometry

### 2.3 Outputs (minimum viable set)
Per run (and per window if windowed):
- n_states
- state probabilities over time (gamma)
- most likely state path (Viterbi)
- transition matrix
- log likelihood and/or information criteria (AIC/BIC)
- regime duration stats (mean duration, switching frequency)

### 2.4 Persistence Guidance
Store:
- run metadata (engine name/version, parameters, data slice, features used)
- transition matrix + summary metrics
- regime path (compressed) or probability summary
Avoid:
- storing full intermediate training artifacts unless needed for reproducibility

### 2.5 Notes on Implementation
- Start with Gaussian HMM (diagonal covariance) as baseline.
- Support multiple feature sets (univariate and multivariate).
- Include a deterministic seed control for repeatability.

---

## 3. Engine: Koopman / Dynamic Mode Decomposition (DMD)

### 3.1 Purpose
Koopman and DMD analyze a system's dynamics via modes and eigenvalues:
- oscillatory behavior (frequencies)
- growth/decay behavior (stability)
- low-rank dynamic structure
- reconstruction error as a stability or regime indicator

This aligns with PRISM's "waves / vibrations" framing and provides a compact
dynamic characterization of both indicators and system geometry over time.

### 3.2 Inputs
DMD/Koopman can operate on:
A) Indicator series (unbounded)
- transformed into state vectors using time-delay embedding if needed

B) Lens outputs (system geometry dynamics)
- geometry metrics time series per window (or combined features)

C) Bounded indicator-in-geometry features
- features of an indicator conditioned on geometry state, or deviations from system modes

### 3.3 Outputs (minimum viable set)
Per run (and per window if windowed):
- eigenvalues (lambda): real part (growth/decay), imag part (frequency)
- mode amplitudes / energies
- rank used and SVD singular values (energy capture)
- reconstruction error (normalized)
- dominant frequencies and their stability over time

### 3.4 Persistence Guidance
Store:
- run metadata (engine name/version, parameters, data slice, embedding/windowing)
- compact eigenvalue summary (top-k modes) per window
- reconstruction error series per window
Avoid:
- storing full mode matrices unless needed for later visualizations or paper figures

### 3.5 Notes on Implementation
- Implement baseline DMD first (exact DMD).
- Add time-delay embedding options as a parameterized extension.
- Keep window sizes and ranks explicit and logged.

---

## 4. Execution Model: Run One Engine at a Time

PRISM should provide a consistent runner interface so engines can be executed on demand.

### 4.1 Desired CLI Shape (conceptual)
- Run one engine on a domain or indicator set:
  - prism run-engine --engine HMM --domain finance --indicator SPX --window 252
  - prism run-engine --engine DMD --domain climate --indicator ENSO --window 3650

- Run an agent-guided selection pass:
  - prism agent-select --scope lens_outputs --objective "boundary stability" --top 3
  - prism agent-select --scope indicators --objective "bounded behavior" --top 5

### 4.2 Caching (optional but recommended)
- Cache engine outputs keyed by:
  engine + params + data slice + windowing
- Agents can re-use cached results instead of recomputing.

---

## 5. Agent Strategy: Broad Swatch, Narrow Execution

The goal is not to run 30 engines blindly.
The goal is to give agents a broad engine complement so they can:
- propose candidate engines for a question
- compare a small shortlist
- justify recommendations with measurable criteria

### 5.1 What agents should produce
For each question/scope, agents should output:
- recommended engines (ranked)
- why those engines fit the objective
- which metrics support selection
- what to compute next (minimal additional runs)

### 5.2 Expected compute pressure
Agent compute cost is unknown today.
To prevent runaway cost:
- keep engines on-demand
- use caching
- standardize small "probe runs" (short windows, small indicator sets) before scaling up

---

## 6. Definition of Done for Adding These Engines

HMM engine:
- deterministic seed control
- supports at least univariate + multivariate features
- produces transition matrix + regime path + likelihood metrics
- logs a run summary record

Koopman/DMD engine:
- supports baseline DMD with explicit rank
- outputs eigenvalue summaries + reconstruction error
- windowed execution support
- logs a run summary record

Both:
- integrate into the same runner interface
- are runnable one-at-a-time
- do not require bulk DB production to be useful

---

## 7. Why These Two Engines Matter for PRISM

HMM:
- provides discrete latent-state baselines to compare against geometric regimes.

Koopman/DMD:
- provides dynamic mode signatures that align with PRISM's vibration/field framing and
  can quantify stability, oscillations, and regime deformation over time.

Together they expand PRISM's ability to:
- detect regimes (HMM)
- explain dynamical structure (DMD/Koopman)
- and let agents decide when each is relevant.

