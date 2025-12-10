"""
PRISM Benchmark Generator v1
============================
Creates 6 synthetic datasets with known ground-truth patterns for lens validation.

Run: python benchmark_generator_v1.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


class BenchmarkGenerator:
    """Generate synthetic data with known patterns for lens validation."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.output_dir = Path('.')
        
        self.benchmarks = {
            'clear_leader': {
                'name': 'Clear Leader',
                'description': 'A drives B, C, D with known lags',
                'ground_truth': {
                    'leader': 'A',
                    'followers': ['B', 'C', 'D'],
                    'lags': [1, 2, 3],
                },
                'expected_results': {
                    'granger': 'A should rank #1, Granger A→B,C,D',
                    'transfer_entropy': 'A is top source',
                }
            },
            'two_regimes': {
                'name': 'Two Regimes',
                'description': 'Clear regime change at day 500',
                'ground_truth': {
                    'break_point': 500,
                    'regime_1': {'mean': 100, 'vol': 1},
                    'regime_2': {'mean': 120, 'vol': 3},
                },
                'expected_results': {
                    'regime': 'Should detect breakpoint near day 500',
                }
            },
            'clusters': {
                'name': 'Three Clusters',
                'description': 'Clear cluster structure',
                'ground_truth': {
                    'cluster_1': ['A', 'B', 'C'],
                    'cluster_2': ['D', 'E', 'F'],
                    'cluster_3': ['G', 'H'],
                },
                'expected_results': {
                    'clustering': 'Should find 3 clusters',
                }
            },
            'periodic': {
                'name': 'Hidden Periodicity',
                'description': 'Different cycle lengths in different columns',
                'ground_truth': {
                    'cycles': {
                        'A': 20,
                        'B': 50,
                        'C': 100,
                        'D': None,
                        'E': [20, 50],
                    }
                },
                'expected_results': {
                    'wavelet': 'Should detect 20/50/100 day frequencies',
                }
            },
            'anomalies': {
                'name': 'Anomaly Injection',
                'description': 'Specific columns have injected anomalies',
                'ground_truth': {
                    'clean': ['A', 'D'],
                    'point_anomalies': {
                        'B': [100, 300, 500, 700, 900],
                        'E': 'random_10',
                    },
                    'collective_anomaly': {
                        'C': (400, 420),
                    }
                },
                'expected_results': {
                    'anomaly': 'B, C, E should rank highest',
                }
            },
            'pure_noise': {
                'name': 'Pure Noise (Control)',
                'description': 'No patterns - control test',
                'ground_truth': {
                    'pattern': None,
                },
                'expected_results': {
                    'all': 'Should find NO strong patterns',
                }
            },
        }
    
    def create_clear_leader(self) -> pd.DataFrame:
        """
        GROUND TRUTH: A drives B, C, D with lags 1, 2, 3.
        E, F are independent noise.
        """
        np.random.seed(self.seed)
        n = 1000
        
        A = np.cumsum(np.random.randn(n) * 2) + 100
        
        B = np.zeros(n)
        B[1:] = 0.8 * A[:-1] + np.random.randn(n-1) * 0.5
        B[0] = 100
        
        C = np.zeros(n)
        C[2:] = 0.7 * A[:-2] + np.random.randn(n-2) * 0.6
        C[:2] = 100
        
        D = np.zeros(n)
        D[3:] = 0.6 * A[:-3] + np.random.randn(n-3) * 0.7
        D[:3] = 100
        
        E = np.cumsum(np.random.randn(n)) + 100
        F = np.cumsum(np.random.randn(n)) + 100
        
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F
        })
    
    def create_two_regimes(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Regime 1 (0-499) low vol, Regime 2 (500-999) high vol.
        """
        np.random.seed(self.seed + 1)
        n = 1000
        break_point = 500
        
        regime1 = np.random.randn(break_point) * 1 + 100
        regime2 = np.random.randn(n - break_point) * 3 + 120
        
        A = np.concatenate([np.cumsum(regime1[:break_point])/10 + 100,
                           np.cumsum(regime2)/(n-break_point)*50 + 150])
        
        B = A + np.random.randn(n) * 2
        C = A * 1.1 + np.random.randn(n) * 2
        D = A * 0.9 + np.random.randn(n) * 2
        E = A + np.random.randn(n) * 3
        F = A * 1.05 + np.random.randn(n) * 2
        
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F
        })
    
    def create_clusters(self) -> pd.DataFrame:
        """
        GROUND TRUTH: 3 clusters with high intra-cluster correlation.
        Cluster 1: A, B, C (corr ~0.9)
        Cluster 2: D, E, F (corr ~0.9)
        Cluster 3: G, H (corr ~0.9)
        """
        np.random.seed(self.seed + 2)
        n = 1000
        
        base1 = np.cumsum(np.random.randn(n)) + 100
        A = base1 + np.random.randn(n) * 0.5
        B = base1 + np.random.randn(n) * 0.5
        C = base1 + np.random.randn(n) * 0.5
        
        base2 = np.cumsum(np.random.randn(n)) + 200
        D = base2 + np.random.randn(n) * 0.5
        E = base2 + np.random.randn(n) * 0.5
        F = base2 + np.random.randn(n) * 0.5
        
        base3 = np.cumsum(np.random.randn(n)) + 150
        G = base3 + np.random.randn(n) * 0.5
        H = base3 + np.random.randn(n) * 0.5
        
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G, 'H': H
        })
    
    def create_periodic(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Different periodicities.
        
        A: 20-day cycle
        B: 50-day cycle
        C: 100-day cycle
        D: No cycle (random walk)
        E: Mixed (20 + 50 day)
        """
        np.random.seed(self.seed + 3)
        n = 1000
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        t = np.arange(n)
        
        A = 50 * np.sin(2 * np.pi * t / 20) + 0 + np.random.randn(n) * 1
        B = 50 * np.sin(2 * np.pi * t / 50) + 0 + np.random.randn(n) * 1
        C = 50 * np.sin(2 * np.pi * t / 100) + 0 + np.random.randn(n) * 1
        D = np.cumsum(np.random.randn(n) * 0.5) + 100
        E = 30 * np.sin(2 * np.pi * t / 20) + 30 * np.sin(2 * np.pi * t / 50) + np.random.randn(n) * 1
        
        df = pd.DataFrame({
            'date': dates,
            'A': A + 100,
            'B': B + 100,
            'C': C + 100,
            'D': D,
            'E': E + 100
        })
        
        return df
    
    def create_anomalies(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Injected anomalies.
        
        A: Clean
        B: Point anomalies at indices 100, 300, 500, 700, 900
        C: Collective anomaly from 400-420
        D: Clean
        E: Random 10 point anomalies
        """
        np.random.seed(self.seed + 4)
        n = 1000
        
        A = np.cumsum(np.random.randn(n) * 0.5) + 100
        D = np.cumsum(np.random.randn(n) * 0.5) + 100
        
        B = np.cumsum(np.random.randn(n) * 0.5) + 100
        for idx in [100, 300, 500, 700, 900]:
            B[idx] += np.random.choice([-1, 1]) * 10
        
        C = np.cumsum(np.random.randn(n) * 0.5) + 100
        C[400:420] += 15
        
        E = np.cumsum(np.random.randn(n) * 0.5) + 100
        anomaly_indices = np.random.choice(n, 10, replace=False)
        E[anomaly_indices] += np.random.randn(10) * 8
        
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'A': A, 'B': B, 'C': C, 'D': D, 'E': E
        })
    
    def create_pure_noise(self) -> pd.DataFrame:
        """
        GROUND TRUTH: No patterns, just random walks.
        """
        np.random.seed(self.seed + 5)
        n = 1000
        
        A = np.cumsum(np.random.randn(n)) + 100
        B = np.cumsum(np.random.randn(n)) + 100
        C = np.cumsum(np.random.randn(n)) + 100
        D = np.cumsum(np.random.randn(n)) + 100
        E = np.cumsum(np.random.randn(n)) + 100
        F = np.cumsum(np.random.randn(n)) + 100
        
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F
        })
    
    def generate_all(self):
        """Generate all benchmarks and save to CSV."""
        generators = {
            'clear_leader': self.create_clear_leader,
            'two_regimes': self.create_two_regimes,
            'clusters': self.create_clusters,
            'periodic': self.create_periodic,
            'anomalies': self.create_anomalies,
            'pure_noise': self.create_pure_noise,
        }
        
        print("=" * 60)
        print("PRISM BENCHMARK GENERATOR")
        print("=" * 60)
        
        for key, generator in generators.items():
            df = generator()
            filename = f"benchmark_{key}.csv"
            df.to_csv(self.output_dir / filename, index=False)
            
            meta = self.benchmarks[key]
            print(f"\n✓ Generated: {meta['name']}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {[c for c in df.columns if c != 'date']}")
            print(f"  CSV: {filename}")
        
        print("\n" + "=" * 60)
        print("✓ All 6 benchmarks generated!")
        print("\nVALIDATION CHECKLIST:")
        for key, meta in self.benchmarks.items():
            print(f"  □ {key}:  {list(meta['expected_results'].values())[0]}")
        print("=" * 60)


if __name__ == '__main__':
    generator = BenchmarkGenerator()
    generator.generate_all()
