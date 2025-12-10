"""
PRISM Benchmark Generator v2
============================
Adds: PCA, Network, Transfer Entropy, Mutual Info benchmarks

Run: python benchmark_generator_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


class BenchmarkGeneratorV2:
    """Generate synthetic data with known patterns for lens validation."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.output_dir = Path('.')
    
    def create_pca_benchmark(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Known factor structure
        
        3 latent factors drive 9 indicators:
        - Factor 1 (market): drives A, B, C (loadings ~0.9)
        - Factor 2 (sector): drives D, E, F (loadings ~0.8)
        - Factor 3 (idio):   drives G, H, I (loadings ~0.7)
        
        Expected: 
        - 3 components explain >80% variance
        - A,B,C cluster on PC1; D,E,F on PC2; G,H,I on PC3
        """
        np.random.seed(self.seed + 10)
        n = 1000
        
        # Latent factors
        factor1 = np.cumsum(np.random.randn(n) * 0.5)  # Market
        factor2 = np.cumsum(np.random.randn(n) * 0.4)  # Sector
        factor3 = np.cumsum(np.random.randn(n) * 0.3)  # Idiosyncratic
        
        # Indicators with known loadings
        noise = 0.1
        A = 0.9 * factor1 + np.random.randn(n) * noise
        B = 0.85 * factor1 + np.random.randn(n) * noise
        C = 0.88 * factor1 + np.random.randn(n) * noise
        
        D = 0.8 * factor2 + np.random.randn(n) * noise
        E = 0.75 * factor2 + np.random.randn(n) * noise
        F = 0.82 * factor2 + np.random.randn(n) * noise
        
        G = 0.7 * factor3 + np.random.randn(n) * noise
        H = 0.65 * factor3 + np.random.randn(n) * noise
        I = 0.72 * factor3 + np.random.randn(n) * noise
        
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'A': A, 'B': B, 'C': C,
            'D': D, 'E': E, 'F': F,
            'G': G, 'H': H, 'I': I
        })
    
    def create_network_benchmark(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Hub-spoke network topology
        
        - A is the HUB: connects to B, C, D, E (high centrality)
        - F, G are PERIPHERAL: weakly connected
        - H is ISOLATED: no real connections
        
        Expected:
        - A has highest centrality/degree
        - H has lowest centrality
        - Network density moderate (~0.4)
        """
        np.random.seed(self.seed + 11)
        n = 1000
        
        # Hub signal
        hub = np.cumsum(np.random.randn(n) * 1.0)
        
        # A is the hub
        A = hub + np.random.randn(n) * 0.1
        
        # B, C, D, E follow hub with lag
        B = 0.8 * np.roll(hub, 1) + np.random.randn(n) * 0.2
        C = 0.75 * np.roll(hub, 2) + np.random.randn(n) * 0.2
        D = 0.7 * np.roll(hub, 1) + np.random.randn(n) * 0.25
        E = 0.65 * np.roll(hub, 3) + np.random.randn(n) * 0.3
        
        # F, G peripheral - weak connection
        peripheral = np.cumsum(np.random.randn(n) * 0.5)
        F = 0.3 * hub + 0.7 * peripheral + np.random.randn(n) * 0.3
        G = 0.2 * hub + 0.8 * peripheral + np.random.randn(n) * 0.3
        
        # H isolated - independent
        H = np.cumsum(np.random.randn(n) * 0.8)
        
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'A': A, 'B': B, 'C': C, 'D': D,
            'E': E, 'F': F, 'G': G, 'H': H
        })
    
    def create_transfer_entropy_benchmark(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Known information flow directions
        
        - A → B (strong, lag 1)
        - A → C (medium, lag 2)
        - D → E (strong, lag 1)
        - F is independent (no flow in/out)
        
        Expected:
        - A ranks high as information SOURCE
        - B, C, E rank high as information SINKS
        - F has low transfer entropy both ways
        """
        np.random.seed(self.seed + 12)
        n = 1000
        
        # Independent drivers
        A = np.cumsum(np.random.randn(n) * 1.0)
        D = np.cumsum(np.random.randn(n) * 0.8)
        F = np.cumsum(np.random.randn(n) * 0.6)  # Independent
        
        # B follows A with lag 1
        B = np.zeros(n)
        B[1:] = 0.8 * A[:-1] + np.random.randn(n-1) * 0.2
        
        # C follows A with lag 2
        C = np.zeros(n)
        C[2:] = 0.6 * A[:-2] + np.random.randn(n-2) * 0.3
        
        # E follows D with lag 1
        E = np.zeros(n)
        E[1:] = 0.75 * D[:-1] + np.random.randn(n-1) * 0.25
        
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'A': A, 'B': B, 'C': C,
            'D': D, 'E': E, 'F': F
        })
    
    def create_mutual_info_benchmark(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Known dependency structure (linear + nonlinear)
        
        - A, B: HIGH linear correlation (~0.9)
        - C, D: HIGH nonlinear dependency (quadratic)
        - E, F: MEDIUM linear correlation (~0.5)
        - G: INDEPENDENT of all
        
        Expected:
        - Mutual info: A-B high, C-D high, E-F medium, G-* low
        - Should detect nonlinear C-D that correlation misses
        """
        np.random.seed(self.seed + 13)
        n = 1000
        
        # A, B: linear relationship
        A = np.random.randn(n) * 2
        B = 0.9 * A + np.random.randn(n) * 0.3
        
        # C, D: nonlinear (quadratic)
        C = np.random.randn(n) * 2
        D = 0.5 * C**2 - 2 + np.random.randn(n) * 0.5  # Quadratic
        
        # E, F: medium linear
        E = np.random.randn(n) * 2
        F = 0.5 * E + np.random.randn(n) * 1.0
        
        # G: independent
        G = np.random.randn(n) * 2
        
        # Add cumsum to make time-series like
        A = np.cumsum(A) / 10 + 100
        B = np.cumsum(B) / 10 + 100
        C = np.cumsum(C) / 10 + 100
        D = D + 100  # Keep D stationary to preserve nonlinearity
        E = np.cumsum(E) / 10 + 100
        F = np.cumsum(F) / 10 + 100
        G = np.cumsum(G) / 10 + 100
        
        dates = pd.date_range('2000-01-01', periods=n, freq='D')
        
        return pd.DataFrame({
            'date': dates,
            'A': A, 'B': B, 'C': C, 'D': D,
            'E': E, 'F': F, 'G': G
        })
    
    def generate_all(self):
        """Generate all v2 benchmarks."""
        benchmarks = {
            'pca': (self.create_pca_benchmark, {
                'name': 'PCA Factor Structure',
                'expected': '3 factors, A-B-C / D-E-F / G-H-I clusters'
            }),
            'network': (self.create_network_benchmark, {
                'name': 'Network Hub-Spoke',
                'expected': 'A=hub (high centrality), H=isolated (low centrality)'
            }),
            'transfer_entropy': (self.create_transfer_entropy_benchmark, {
                'name': 'Transfer Entropy Flow',
                'expected': 'A→B,C and D→E detected, F independent'
            }),
            'mutual_info': (self.create_mutual_info_benchmark, {
                'name': 'Mutual Information Dependencies',
                'expected': 'A-B high, C-D high (nonlinear), G independent'
            }),
        }
        
        print("=" * 60)
        print("PRISM BENCHMARK GENERATOR v2")
        print("=" * 60)
        
        for key, (generator, meta) in benchmarks.items():
            df = generator()
            filename = f"benchmark_{key}.csv"
            df.to_csv(self.output_dir / filename, index=False)
            
            print(f"\n✓ Generated: {meta['name']}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {[c for c in df.columns if c != 'date']}")
            print(f"  CSV: {filename}")
            print(f"  Expected: {meta['expected']}")
        
        print("\n" + "=" * 60)
        print("✓ All 4 new benchmarks generated!")
        print("=" * 60)


if __name__ == '__main__':
    generator = BenchmarkGeneratorV2()
    generator.generate_all()
