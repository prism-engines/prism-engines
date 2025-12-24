"""
PRISM Benchmark Data Generator
==============================

Creates synthetic datasets with KNOWN structure for validating PRISM.
Integrates directly with the PRISM database schema.

Usage:
    # Generate CSVs only
    python benchmark_generator.py --csv
    
    # Generate and load into database
    python benchmark_generator.py --db path/to/prism.db
    
    # Generate specific benchmark
    python benchmark_generator.py --benchmark clear_leader --db prism.db

    # Python API
    from benchmark_generator import BenchmarkGenerator
    gen = BenchmarkGenerator(db_path='prism.db')
    gen.generate_all()
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# DATABASE SCHEMA (for reference / initialization)
# =============================================================================

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS systems (
    system TEXT PRIMARY KEY,
    description TEXT
);

CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    system TEXT NOT NULL,
    frequency TEXT NOT NULL DEFAULT 'daily',
    source TEXT NOT NULL DEFAULT 'UNKNOWN',
    description TEXT,
    metadata TEXT,
    FOREIGN KEY(system) REFERENCES systems(system),
    UNIQUE(name, system)
);

CREATE TABLE IF NOT EXISTS indicator_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    value REAL,
    value_2 REAL,
    adjusted_value REAL,
    FOREIGN KEY(indicator_id) REFERENCES indicators(id),
    UNIQUE(indicator_id, date)
);

INSERT OR IGNORE INTO systems(system, description) VALUES
    ('market', 'Market price time series'),
    ('economy', 'Economic indicators'),
    ('benchmark', 'Synthetic benchmark datasets'),
    ('custom', 'User-defined or research datasets');

CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_name TEXT,
    fred_code TEXT,
    timestamp TEXT,
    status TEXT,
    message TEXT
);
"""


# =============================================================================
# BENCHMARK DEFINITIONS
# =============================================================================

BENCHMARKS = {
    'clear_leader': {
        'name': 'Clear Leader',
        'description': 'Column A drives everything else with known lags',
        'ground_truth': {
            'leader': 'A',
            'followers': {'B': 3, 'C': 5, 'D': 7},  # lag in days
            'noise': ['E', 'F'],
        },
        'expected_results': {
            'granger': 'A should Granger-cause B, C, D',
            'transfer_entropy': 'A should have highest outflow',
            'influence': 'A should rank #1',
            'ranking': ['A', 'B', 'C', 'D', 'E', 'F'],
        }
    },
    'two_regimes': {
        'name': 'Two Regimes',
        'description': 'Clear regime change at day 500',
        'ground_truth': {
            'regime_change_day': 500,
            'regime_1': {'drift': 'positive', 'volatility': 'low'},
            'regime_2': {'drift': 'negative', 'volatility': 'high'},
            'control_column': 'F',  # Does not change
        },
        'expected_results': {
            'regime': 'Should detect split at ~day 500',
            'volatility': 'Should increase after day 500',
        }
    },
    'clusters': {
        'name': 'Three Clusters',
        'description': 'Three distinct groups of correlated assets',
        'ground_truth': {
            'clusters': {
                1: ['A', 'B', 'C'],
                2: ['D', 'E', 'F'],
                3: ['G', 'H'],
            },
            'within_correlation': 0.9,
            'between_correlation': 0.1,
        },
        'expected_results': {
            'clustering': 'Should find 3 clusters',
            'network': 'Should show 3 communities',
        }
    },
    'periodic': {
        'name': 'Hidden Periodicity',
        'description': 'Different cycle lengths in different columns',
        'ground_truth': {
            'cycles': {
                'A': 20,   # days
                'B': 50,
                'C': 100,
                'D': None,  # No cycle
                'E': [20, 50],  # Mixed
            }
        },
        'expected_results': {
            'wavelet': 'Should detect 20/50/100 day frequencies',
            'decomposition': 'A, B, C should show strong seasonal',
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
        'description': 'Independent random walks - NO structure',
        'ground_truth': {
            'structure': None,
            'correlation': 'near zero',
        },
        'expected_results': {
            'all_lenses': 'Should find NO strong patterns',
            'consensus': 'Low agreement between lenses',
        }
    },
}


# =============================================================================
# BENCHMARK GENERATOR CLASS
# =============================================================================

class BenchmarkGenerator:
    """
    Generates synthetic benchmark datasets with known ground truth.
    Can output to CSV files and/or load directly into PRISM database.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        output_dir: str = '.',
        seed: int = 42
    ):
        """
        Initialize the benchmark generator.
        
        Args:
            db_path: Path to SQLite database (optional)
            output_dir: Directory for CSV output
            seed: Random seed for reproducibility
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Initialize database if path provided
        if db_path:
            self._init_database()
    
    def _init_database(self):
        """Initialize database with schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        print(f"Database initialized: {self.db_path}")
    
    def _register_indicator(
        self,
        conn: sqlite3.Connection,
        name: str,
        benchmark_name: str,
        description: str,
        metadata: dict
    ) -> int:
        """Register an indicator in the database and return its ID."""
        cursor = conn.cursor()
        
        # Full indicator name includes benchmark prefix
        full_name = f"benchmark_{benchmark_name}_{name}"
        
        cursor.execute("""
            INSERT OR REPLACE INTO indicators (name, system, frequency, source, description, metadata)
            VALUES (?, 'benchmark', 'daily', 'synthetic', ?, ?)
        """, (full_name, description, json.dumps(metadata)))
        
        # Get the indicator ID
        cursor.execute("SELECT id FROM indicators WHERE name = ?", (full_name,))
        result = cursor.fetchone()
        return result[0]
    
    def _store_values(
        self,
        conn: sqlite3.Connection,
        indicator_id: int,
        df: pd.DataFrame,
        column: str
    ):
        """Store time series values in the database."""
        cursor = conn.cursor()
        
        # Prepare data for bulk insert
        data = [
            (indicator_id, date.strftime('%Y-%m-%d'), float(value))
            for date, value in zip(df.index, df[column])
            if pd.notna(value)
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO indicator_values (indicator_id, date, value)
            VALUES (?, ?, ?)
        """, data)
    
    def _log_generation(self, conn: sqlite3.Connection, benchmark_name: str, status: str, message: str):
        """Log the generation event."""
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO fetch_log (indicator_name, timestamp, status, message)
            VALUES (?, ?, ?, ?)
        """, (f"benchmark_{benchmark_name}", datetime.now().isoformat(), status, message))
    
    # =========================================================================
    # BENCHMARK GENERATORS
    # =========================================================================
    
    def create_clear_leader(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Column 'A' drives everything else.
        
        - A is the leader (random walk with momentum)
        - B follows A with 3-day lag
        - C follows A with 5-day lag
        - D follows A with 7-day lag
        - E, F are independent noise
        """
        np.random.seed(self.seed)
        n = 1000
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        
        # The leader - random walk with momentum
        leader = np.cumsum(np.random.randn(n) * 0.02) + 100
        
        # Followers - leader + lag + noise
        follower1 = np.roll(leader, 3) + np.random.randn(n) * 0.5
        follower2 = np.roll(leader, 5) + np.random.randn(n) * 0.8
        follower3 = np.roll(leader, 7) + np.random.randn(n) * 1.0
        
        # Fix roll artifacts
        for f in [follower1, follower2, follower3]:
            f[:10] = leader[:10] + np.random.randn(10) * 0.5
        
        # Independent noise
        noise1 = np.cumsum(np.random.randn(n) * 0.01) + 50
        noise2 = np.cumsum(np.random.randn(n) * 0.01) + 50
        
        df = pd.DataFrame({
            'A': leader,
            'B': follower1,
            'C': follower2,
            'D': follower3,
            'E': noise1,
            'F': noise2,
        }, index=dates)
        df.index.name = 'date'
        
        return df
    
    def create_two_regimes(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Clear regime change at day 500.
        
        Regime 1 (days 1-500): Low volatility, positive drift
        Regime 2 (days 501-1000): High volatility, negative drift
        Column F does NOT change (control)
        """
        np.random.seed(self.seed + 1)
        n = 1000
        regime_change = 500
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        
        data = {}
        for col in ['A', 'B', 'C', 'D', 'E']:
            series = np.zeros(n)
            
            # Regime 1: calm, trending up
            drift1 = np.random.uniform(0.001, 0.003)
            vol1 = np.random.uniform(0.005, 0.01)
            series[:regime_change] = np.cumsum(
                np.random.randn(regime_change) * vol1 + drift1
            ) + 100
            
            # Regime 2: volatile, trending down
            drift2 = np.random.uniform(-0.003, -0.001)
            vol2 = np.random.uniform(0.02, 0.04)
            series[regime_change:] = series[regime_change-1] + np.cumsum(
                np.random.randn(n - regime_change) * vol2 + drift2
            )
            
            data[col] = series
        
        # Control column - constant behavior throughout
        data['F'] = np.cumsum(np.random.randn(n) * 0.01) + 100
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'date'
        
        return df
    
    def create_clusters(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Three distinct clusters.
        
        Cluster 1: A, B, C (ρ ≈ 0.9 within)
        Cluster 2: D, E, F (ρ ≈ 0.9 within)
        Cluster 3: G, H (ρ ≈ 0.9 within)
        Between clusters: ρ ≈ 0.1
        """
        np.random.seed(self.seed + 2)
        n = 1000
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        
        # Cluster 1 base signal
        base1 = np.cumsum(np.random.randn(n) * 0.02) + 100
        A = base1 + np.random.randn(n) * 0.3
        B = base1 + np.random.randn(n) * 0.3
        C = base1 + np.random.randn(n) * 0.3
        
        # Cluster 2 base signal
        base2 = np.cumsum(np.random.randn(n) * 0.02) + 50
        D = base2 + np.random.randn(n) * 0.3
        E = base2 + np.random.randn(n) * 0.3
        F = base2 + np.random.randn(n) * 0.3
        
        # Cluster 3 base signal
        base3 = np.cumsum(np.random.randn(n) * 0.02) + 75
        G = base3 + np.random.randn(n) * 0.3
        H = base3 + np.random.randn(n) * 0.3
        
        df = pd.DataFrame({
            'A': A, 'B': B, 'C': C,
            'D': D, 'E': E, 'F': F,
            'G': G, 'H': H,
        }, index=dates)
        df.index.name = 'date'
        
        return df
    
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
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        t = np.arange(n)
        
        A = 10 * np.sin(2 * np.pi * t / 20) + t * 0.01 + np.random.randn(n) * 1
        B = 15 * np.sin(2 * np.pi * t / 50) + t * 0.01 + np.random.randn(n) * 1
        C = 20 * np.sin(2 * np.pi * t / 100) + t * 0.01 + np.random.randn(n) * 1
        D = np.cumsum(np.random.randn(n) * 0.5) + 100
        E = 8 * np.sin(2 * np.pi * t / 20) + 8 * np.sin(2 * np.pi * t / 50) + np.random.randn(n) * 1
        
        df = pd.DataFrame({
            'A': A + 100,
            'B': B + 100,
            'C': C + 100,
            'D': D,
            'E': E + 100,
        }, index=dates)
        df.index.name = 'date'
        
        return df
    
    def create_anomalies(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Injected anomalies.
        
        A: Clean
        B: 5 point anomalies (spikes at known locations)
        C: Collective anomaly (days 400-420)
        D: Clean
        E: 10 random point anomalies
        """
        np.random.seed(self.seed + 4)
        n = 1000
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        
        # Base random walks
        A = np.cumsum(np.random.randn(n) * 0.01) + 100
        B = np.cumsum(np.random.randn(n) * 0.01) + 100
        C = np.cumsum(np.random.randn(n) * 0.01) + 100
        D = np.cumsum(np.random.randn(n) * 0.01) + 100
        E = np.cumsum(np.random.randn(n) * 0.01) + 100
        
        # Inject point anomalies in B
        for idx in [100, 300, 500, 700, 900]:
            B[idx] += np.random.choice([-1, 1]) * 5
        
        # Inject collective anomaly in C
        C[400:420] += np.cumsum(np.random.randn(20) * 0.5)
        
        # Inject point anomalies in E
        spike_idx = np.random.choice(range(50, 950), 10, replace=False)
        for idx in spike_idx:
            E[idx] += np.random.choice([-1, 1]) * 4
        
        df = pd.DataFrame({
            'A': A, 'B': B, 'C': C, 'D': D, 'E': E,
        }, index=dates)
        df.index.name = 'date'
        
        return df
    
    def create_pure_noise(self) -> pd.DataFrame:
        """
        GROUND TRUTH: Pure noise - NO structure.
        
        All columns are independent random walks.
        If PRISM finds patterns, it's overfitting.
        """
        np.random.seed(self.seed + 5)
        n = 1000
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        
        df = pd.DataFrame({
            col: np.cumsum(np.random.randn(n) * 0.01) + 100
            for col in ['A', 'B', 'C', 'D', 'E', 'F']
        }, index=dates)
        df.index.name = 'date'
        
        return df
    
    # =========================================================================
    # GENERATION METHODS
    # =========================================================================
    
    def generate(
        self,
        benchmark: str,
        save_csv: bool = True,
        load_db: bool = True
    ) -> pd.DataFrame:
        """
        Generate a specific benchmark dataset.
        
        Args:
            benchmark: Benchmark name (clear_leader, two_regimes, etc.)
            save_csv: Save to CSV file
            load_db: Load into database (if db_path configured)
        
        Returns:
            Generated DataFrame
        """
        # Get generator method
        generators = {
            'clear_leader': self.create_clear_leader,
            'two_regimes': self.create_two_regimes,
            'clusters': self.create_clusters,
            'periodic': self.create_periodic,
            'anomalies': self.create_anomalies,
            'pure_noise': self.create_pure_noise,
        }
        
        if benchmark not in generators:
            raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(generators.keys())}")
        
        # Generate data
        df = generators[benchmark]()
        bench_info = BENCHMARKS[benchmark]
        
        print(f"✓ Generated: {bench_info['name']}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Save CSV
        if save_csv:
            csv_path = self.output_dir / f"benchmark_{benchmark}.csv"
            df.to_csv(csv_path)
            print(f"  CSV: {csv_path}")
        
        # Load to database
        if load_db and self.db_path:
            self._load_to_database(benchmark, df, bench_info)
            print(f"  DB: loaded {len(df.columns)} indicators")
        
        return df
    
    def _load_to_database(
        self,
        benchmark_name: str,
        df: pd.DataFrame,
        bench_info: dict
    ):
        """Load benchmark data into the database."""
        with sqlite3.connect(self.db_path) as conn:
            for col in df.columns:
                # Determine column role from ground truth
                ground_truth = bench_info.get('ground_truth', {})
                role = self._determine_role(col, ground_truth)
                
                # Register indicator
                indicator_id = self._register_indicator(
                    conn=conn,
                    name=col,
                    benchmark_name=benchmark_name,
                    description=f"{bench_info['name']} - {col} ({role})",
                    metadata={
                        'benchmark': benchmark_name,
                        'role': role,
                        'ground_truth': str(ground_truth),
                    }
                )
                
                # Store values
                self._store_values(conn, indicator_id, df, col)
            
            # Log generation
            self._log_generation(
                conn, benchmark_name, 'success',
                f"Generated {len(df.columns)} indicators, {len(df)} rows"
            )
            
            conn.commit()
    
    def _determine_role(self, col: str, ground_truth: dict) -> str:
        """Determine the role of a column based on ground truth."""
        if ground_truth.get('leader') == col:
            return 'leader'
        if col in ground_truth.get('followers', {}):
            return f"follower_lag{ground_truth['followers'][col]}"
        if col in ground_truth.get('noise', []):
            return 'noise'
        if col in ground_truth.get('clean', []):
            return 'clean'
        if col == ground_truth.get('control_column'):
            return 'control'
        
        # Check clusters
        for cluster_id, members in ground_truth.get('clusters', {}).items():
            if col in members:
                return f'cluster_{cluster_id}'
        
        # Check cycles
        cycles = ground_truth.get('cycles', {})
        if col in cycles:
            cycle = cycles[col]
            if cycle is None:
                return 'no_cycle'
            elif isinstance(cycle, list):
                return f"cycle_mixed_{'+'.join(map(str, cycle))}"
            else:
                return f'cycle_{cycle}d'
        
        return 'unknown'
    
    def generate_all(self, save_csv: bool = True, load_db: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate all benchmark datasets.
        
        Returns:
            Dictionary mapping benchmark name to DataFrame
        """
        print("=" * 60)
        print("PRISM BENCHMARK GENERATOR")
        print("=" * 60)
        print()
        
        results = {}
        for benchmark in BENCHMARKS.keys():
            results[benchmark] = self.generate(benchmark, save_csv, load_db)
            print()
        
        print("=" * 60)
        print("✓ All 6 benchmarks generated!")
        print()
        self.print_validation_checklist()
        print("=" * 60)
        
        return results
    
    def print_validation_checklist(self):
        """Print the validation checklist."""
        print("VALIDATION CHECKLIST:")
        print("  □ clear_leader:  A ranks #1, Granger A→B,C,D")
        print("  □ two_regimes:   Regime split detected ~day 500")
        print("  □ clusters:      3 clusters: {A,B,C}, {D,E,F}, {G,H}")
        print("  □ periodic:      Wavelet finds 20/50/100 day cycles")
        print("  □ anomalies:     B,C,E rank high on anomaly lens")
        print("  □ pure_noise:    NO strong patterns (control)")
    
    def print_answer_key(self):
        """Print detailed answer key for all benchmarks."""
        print("\n" + "=" * 60)
        print("BENCHMARK ANSWER KEY")
        print("=" * 60)
        
        for name, info in BENCHMARKS.items():
            print(f"\n{name}")
            print("-" * len(name))
            print(f"Description: {info['description']}")
            print(f"Ground Truth: {json.dumps(info['ground_truth'], indent=2)}")
            print(f"Expected Results:")
            for lens, expectation in info['expected_results'].items():
                print(f"  - {lens}: {expectation}")
    
    def get_ground_truth(self, benchmark: str) -> dict:
        """Get the ground truth for a specific benchmark."""
        if benchmark not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        return BENCHMARKS[benchmark]['ground_truth']
    
    def validate_results(self, benchmark: str, results: dict) -> dict:
        """
        Validate PRISM results against ground truth.
        
        Args:
            benchmark: Benchmark name
            results: PRISM analysis results
        
        Returns:
            Validation report
        """
        ground_truth = self.get_ground_truth(benchmark)
        validation = {'benchmark': benchmark, 'checks': []}
        
        # Add specific validation logic per benchmark type
        if benchmark == 'clear_leader':
            # Check if A is ranked #1
            if 'rankings' in results:
                top_indicator = results['rankings'].get('consensus', {}).get(0)
                validation['checks'].append({
                    'test': 'Leader is A',
                    'expected': 'A',
                    'actual': top_indicator,
                    'passed': top_indicator == 'A'
                })
        
        elif benchmark == 'clusters':
            # Check if 3 clusters found
            if 'clustering' in results:
                n_clusters = results['clustering'].get('n_clusters', 0)
                validation['checks'].append({
                    'test': 'Number of clusters',
                    'expected': 3,
                    'actual': n_clusters,
                    'passed': n_clusters == 3
                })
        
        # Add more validation logic as needed...
        
        return validation


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Benchmark Data Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmarks:
  clear_leader  - Column A drives B,C,D with known lags
  two_regimes   - Regime change at day 500
  clusters      - Three distinct correlation clusters
  periodic      - Different cycle lengths (20/50/100 days)
  anomalies     - Injected point and collective anomalies
  pure_noise    - Control (no structure)

Examples:
  python benchmark_generator.py --all
  python benchmark_generator.py --db prism.db --all
  python benchmark_generator.py --benchmark clear_leader --csv
  python benchmark_generator.py --answer-key
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Generate all benchmarks')
    parser.add_argument('--benchmark', '-b', choices=list(BENCHMARKS.keys()),
                        help='Generate specific benchmark')
    parser.add_argument('--db', type=str, help='Database path (loads data into DB)')
    parser.add_argument('--output', '-o', type=str, default='.', help='Output directory for CSVs')
    parser.add_argument('--csv', action='store_true', default=True, help='Save CSV files (default: True)')
    parser.add_argument('--no-csv', action='store_true', help='Skip CSV output')
    parser.add_argument('--answer-key', action='store_true', help='Print answer key and exit')
    parser.add_argument('--checklist', action='store_true', help='Print validation checklist and exit')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create generator
    gen = BenchmarkGenerator(
        db_path=args.db,
        output_dir=args.output,
        seed=args.seed
    )
    
    # Handle info requests
    if args.answer_key:
        gen.print_answer_key()
        return
    
    if args.checklist:
        gen.print_validation_checklist()
        return
    
    # Generate benchmarks
    save_csv = args.csv and not args.no_csv
    load_db = args.db is not None
    
    if args.all:
        gen.generate_all(save_csv=save_csv, load_db=load_db)
    elif args.benchmark:
        gen.generate(args.benchmark, save_csv=save_csv, load_db=load_db)
    else:
        # Default: generate all
        gen.generate_all(save_csv=save_csv, load_db=load_db)


if __name__ == '__main__':
    main()
