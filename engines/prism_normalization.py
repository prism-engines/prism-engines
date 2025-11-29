'''PRISM Normalization Module'''

import numpy as np
import pandas as pd
from typing import Optional, Dict

class PRISMNormalizer:
    def __init__(self, ma_window: int = 12, roc_window: int = 1):
        self.ma_window = ma_window
        self.roc_window = roc_window
        
    def dual_input_transform(self, series: pd.Series, name: str):
        if series.isna().sum() > len(series) * 0.3:
            raise ValueError(f"Series {name} has >30% missing data")
            
        series = series.ffill().bfill()
        
        # POSITION: Ratio to moving average
        ma = series.rolling(window=self.ma_window, min_periods=1).mean()
        position = series / ma
        position = position.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        # MOMENTUM: Rate of change
        momentum = series.pct_change(periods=self.roc_window)
        momentum = momentum.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return pd.DataFrame({
            f'{name}_position': position,
            f'{name}_momentum': momentum
        }, index=series.index)
    
    def batch_normalize(self, df: pd.DataFrame, method: str = 'dual_input'):
        if method == 'dual_input':
            results = []
            for col in df.columns:
                try:
                    normalized = self.dual_input_transform(df[col], col)
                    results.append(normalized)
                except Exception as e:
                    print(f"Warning: Failed to normalize {col}: {e}")
                    continue
            return pd.concat(results, axis=1)
        elif method == 'standard_zscore':
            return (df - df.mean()) / df.std(ddof=0)
        else:
            raise ValueError(f"Unknown method: {method}")

def create_state_matrix(market_data: Dict[str, pd.Series], 
                       normalizer: Optional[PRISMNormalizer] = None):
    if normalizer is None:
        normalizer = PRISMNormalizer()
    
    df = pd.DataFrame(market_data)
    state = normalizer.batch_normalize(df, method='dual_input')
    state = state.sort_index().ffill().bfill()
    state = state.dropna(axis=1, how='all').dropna(axis=0, how='all')
    
    return state
