# PRISM Cleaning Rules

This document defines the data cleaning rules applied to raw indicator data.
The cleaner module (`prism/cleaning/cleaner.py`) implements these rules.

## Overview

**Data Flow:**
```
Fetcher → Cleaner → DB
           ↓
    raw.indicators (archive, inert)
    clean.indicators (active, used by engines)
```

Cleaning transforms raw fetched data into engine-ready data.
Cleaning does NOT include normalization (engines handle that internally).

---

## Cleaning Steps

### 1. Date Standardization
- Convert all dates to `datetime64`
- Remove time component (normalize to date only)
- Sort ascending by date

### 2. Duplicate Handling
- Remove duplicate dates
- Keep the **last** value for any duplicate date
- Rationale: Later values often reflect corrections

### 3. Gap Filling (Forward Fill)
- Forward-fill missing values up to **5 consecutive days**
- Covers weekends (2 days) plus typical holidays (3 days)
- Gaps larger than 5 days are NOT filled

### 4. NaN Removal
- After forward-fill, drop any remaining NaN values
- This removes:
  - Leading NaNs (before first data point)
  - Large gaps that couldn't be filled

---

## Frequency-Specific Notes

| Frequency   | Gap Fill Limit | Notes                                    |
|-------------|----------------|------------------------------------------|
| Daily       | 5 days         | Handles weekends + holidays              |
| Weekly      | 5 days         | Same limit, rarely triggered             |
| Monthly     | 5 days         | Rarely triggered                         |
| Quarterly   | 5 days         | Rarely triggered                         |

All data is stored at its **native frequency**. 
Alignment to daily (if needed) happens at query time or in engines.

---

## Cleaning Method Codes

The `cleaning_method` column in `clean.indicators` records what was done:

| Code                      | Meaning                                |
|---------------------------|----------------------------------------|
| `sorted_deduped`          | Sorted and removed duplicates only     |
| `sorted_deduped_ffill_5`  | Also forward-filled gaps (limit 5)     |
| `sorted_deduped_ffill_5_dropped_nulls` | Also dropped remaining NaNs |
| `none_empty_input`        | Input was empty, no cleaning done      |

---

## What Cleaning Does NOT Do

- **Normalization** — Engines handle this (z-score, min-max, etc.)
- **Outlier removal** — Preserved; engines can filter if needed
- **Frequency alignment** — Data stays at native frequency
- **Interpolation** — Only forward-fill, no linear interpolation
- **Smoothing** — Raw values preserved

---

## Validation Checks

Cleaned data must pass:

1. ✓ Has columns: `date`, `value`
2. ✓ No NaN values in `value`
3. ✓ Dates are sorted ascending
4. ✓ No duplicate dates

If validation fails, the indicator is marked as failed in the fetch run.

---

## Examples

### Example 1: Weekend Gap
```
Raw:
  2024-01-05 (Fri)  100.0
  2024-01-06 (Sat)  NaN     ← Missing (weekend)
  2024-01-07 (Sun)  NaN     ← Missing (weekend)
  2024-01-08 (Mon)  101.0

Clean:
  2024-01-05  100.0
  2024-01-06  100.0  ← Forward-filled
  2024-01-07  100.0  ← Forward-filled
  2024-01-08  101.0
```

### Example 2: Large Gap (NOT filled)
```
Raw:
  2024-01-01  100.0
  2024-01-02  NaN
  2024-01-03  NaN
  2024-01-04  NaN
  2024-01-05  NaN
  2024-01-06  NaN
  2024-01-07  NaN     ← 6th consecutive NaN, exceeds limit
  2024-01-08  105.0

Clean:
  2024-01-01  100.0
  2024-01-02  100.0  ← Filled (1)
  2024-01-03  100.0  ← Filled (2)
  2024-01-04  100.0  ← Filled (3)
  2024-01-05  100.0  ← Filled (4)
  2024-01-06  100.0  ← Filled (5, limit reached)
  2024-01-07  NaN    ← NOT filled, then dropped
  2024-01-08  105.0
```

### Example 3: Duplicate Dates
```
Raw:
  2024-01-05  100.0
  2024-01-05  100.5   ← Duplicate (correction)
  2024-01-06  101.0

Clean:
  2024-01-05  100.5   ← Kept last value
  2024-01-06  101.0
```

---

## Version History

| Version | Date       | Changes                        |
|---------|------------|--------------------------------|
| 1.0     | 2024-12-17 | Initial cleaning rules defined |
