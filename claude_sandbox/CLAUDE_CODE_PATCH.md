# Claude Code: Apply this patch to workflows/benchmark_suite.py

## File: workflows/benchmark_suite.py

### Change 1: Fix structure_detected logic (lines 447-452)

FIND THIS CODE:
```python
        # Determine if structure was detected
        result.structure_detected = (
            result.lens_agreement > 0.3 or
            (result.pca_explained_variance and result.pca_explained_variance[0] > 0.4) or
            result.mrf_score < 2.0
        )
```

REPLACE WITH:
```python
        # Determine if structure was detected
        # Note: Removed MRF criterion - too permissive, causes false positives on pure noise
        result.structure_detected = (
            result.lens_agreement > 0.3 or
            (result.pca_explained_variance and result.pca_explained_variance[0] > 0.5)
        )
```

RATIONALE:
- MRF (Mean Rank Fluctuation) < 2.0 was triggering true even for pure noise
- Increased PCA threshold from 0.4 to 0.5 for stricter detection
- Pure noise benchmark should return structure_detected=False

---

## File: data/benchmark/benchmark_generator.py

REPLACE ENTIRE FILE with the contents of the attached benchmark_generator.py

RATIONALE:
- Original clear_leader had inverted SNR (noise 25x larger than signal)
- After normalization, all indicators appeared identical
- Fixed all 6 benchmarks with proper signal-to-noise ratios

---

## After applying changes, run:

```bash
# 1. Regenerate benchmark CSVs
cd data/benchmark
python benchmark_generator.py

# 2. Reload into database  
cd ../..
python -m data.sql.load_benchmarks

# 3. Verify fix
python prism_run.py --benchmarks
```

## Expected outcome:
- clear_leader: lens_agreement > 0.5, A ranks #1
- pure_noise: structure_detected = False
