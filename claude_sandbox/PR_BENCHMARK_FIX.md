# PR: Fix Benchmark Generator Signal-to-Noise Ratios

## Summary

The benchmark synthetic data generator has inverted signal-to-noise ratios that make validation meaningless. This PR fixes all 6 benchmarks and adjusts the `structure_detected` logic in `benchmark_suite.py`.

## Problem

### Benchmark 01 (clear_leader) - Critical Bug
The "leader" indicator A had **lower** variance than followers:
```python
# OLD (broken)
leader = np.cumsum(np.random.randn(n) * 0.02)      # vol = 0.02
follower1 = np.roll(leader, 3) + np.random.randn(n) * 0.5  # noise = 0.5 (25x larger!)
```

Result: Variance showed D > C > B > A, making the "leader" actually the quietest signal. After normalization, all indicators appeared equal.

### Benchmark 06 (pure_noise) - False Positive
`structure_detected` returned `True` for pure noise because the MRF threshold was too permissive.

## Solution

### 1. Replace `data/benchmark/benchmark_generator.py`

Replace entire file with the attached `benchmark_generator.py`.

Key changes:
- **clear_leader**: Leader vol 0.02→0.5, follower noise 0.5→0.05 (SNR inverted from 1:25 to 10:1)
- **two_regimes**: Added correlation structure change between regimes, increased vol contrast to 5x
- **clusters**: Reduced within-cluster noise from 0.3 to 0.03 (cleaner separation)
- **periodic**: Increased cycle amplitude relative to noise (SNR ~10:1)
- **anomalies**: Scaled anomaly magnitude to be clearly detectable
- **pure_noise**: No changes (already correct), but added documentation

### 2. Update `workflows/benchmark_suite.py` lines 448-452

Replace:
```python
result.structure_detected = (
    result.lens_agreement > 0.3 or
    (result.pca_explained_variance and result.pca_explained_variance[0] > 0.4) or
    result.mrf_score < 2.0
)
```

With:
```python
result.structure_detected = (
    result.lens_agreement > 0.3 or
    (result.pca_explained_variance and result.pca_explained_variance[0] > 0.5)
    # Removed MRF criterion - too permissive, causes false positives on noise
)
```

### 3. Regenerate benchmark data

After updating the generator:
```bash
cd data/benchmark
python benchmark_generator.py
```

Then reload into database:
```bash
python -m data.sql.load_benchmarks
```

## Verification

After applying changes, run:
```bash
python prism_run.py --benchmarks
```

Expected results:
| Dataset | Lens Agreement | Structure Detected |
|---------|---------------|-------------------|
| clear_leader | > 0.5 | Yes |
| two_regimes | > 0.3 | Yes |
| clusters | > 0.3 | Yes |
| periodic | > 0.2 | Yes |
| anomalies | > 0.2 | Yes |
| pure_noise | < 0.2 | **No** (critical!) |

Also verify clear_leader shows A as #1:
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/benchmark/benchmark_01_clear_leader.csv', index_col=0)
print('Variance per column:')
print(df.var().sort_values(ascending=False))
"
```

Should show: A >> B ≈ C ≈ D >> E ≈ F

## Files Changed

1. `data/benchmark/benchmark_generator.py` - Full replacement
2. `workflows/benchmark_suite.py` - Lines 448-452 only

## Testing

- [ ] All 6 benchmarks generate without error
- [ ] clear_leader: A has highest variance (10x+ over E, F)
- [ ] pure_noise: structure_detected = False
- [ ] Lens agreement for clear_leader > 0.5
