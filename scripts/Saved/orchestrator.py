from typing import List, Dict, Optional
"""
PRISM Agent Orchestrator

Coordinates all domain agents for unified data acquisition.
Runs overnight fetches, validates data, stores to unified schema.

Usage:
    from prism.agents import Orchestrator
    
    orch = Orchestrator()
    orch.run_all_agents()
    orch.print_summary()
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

import pandas as pd

from prism.db.connection import get_db_path, get_connection
from .base import DomainAgent, IndicatorMeta, FetchResult
from .climate import ClimateAgent
from .epidemiology import EpidemiologyAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Coordinates all domain agents.
    
    Responsibilities:
    - Discover indicators across all domains
    - Run fetches with rate limiting
    - Validate data quality
    - Store to unified schema
    - Generate reports
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path if db_path else get_db_path()
        self.agents: Dict[str, DomainAgent] = {}
        self.all_results: List[FetchResult] = []
        
        # Register agents
        self._register_agents()
    
    def _register_agents(self):
        """Register all available domain agents."""
        self.agents["climate"] = ClimateAgent()
        self.agents["epidemiology"] = EpidemiologyAgent()
        # Finance agent would go here (refactor existing fetch/)
        
        logger.info(f"Registered {len(self.agents)} domain agents")
    
    def discover_all(self) -> Dict[str, List[IndicatorMeta]]:
        """Discover indicators from all agents."""
        all_indicators = {}
        
        for domain, agent in self.agents.items():
            indicators = agent.discover()
            all_indicators[domain] = indicators
            logger.info(f"{domain}: {len(indicators)} indicators")
        
        return all_indicators
    
    def _ensure_schema(self):
        """Ensure multi-domain schema exists."""
        conn = get_connection(self.db_path)
        
        # Create domain data table
        conn.execute('''
            CREATE SCHEMA IF NOT EXISTS domains
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS domains.indicators (
                date DATE,
                domain VARCHAR,
                indicator_id VARCHAR,
                value DOUBLE,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (date, domain, indicator_id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS domains.metadata (
                indicator_id VARCHAR PRIMARY KEY,
                domain VARCHAR,
                name VARCHAR,
                source VARCHAR,
                category VARCHAR,
                frequency VARCHAR,
                description VARCHAR,
                first_date DATE,
                last_date DATE,
                row_count INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.close()
        logger.info("Domain schema ready")
    
    def _store_data(self, domain: str, indicator_id: str, df: pd.DataFrame):
        """Store fetched data to database."""
        if df is None or len(df) == 0:
            return

        conn = get_connection(self.db_path)
        
        # Insert data
        for _, row in df.iterrows():
            conn.execute('''
                INSERT OR REPLACE INTO domains.indicators 
                (date, domain, indicator_id, value, fetched_at)
                VALUES (?, ?, ?, ?, ?)
            ''', [row['date'], domain, indicator_id, row['value'], datetime.now()])
        
        conn.close()
    
    def _update_metadata(self, domain: str, indicator: IndicatorMeta, df: pd.DataFrame):
        """Update indicator metadata."""
        if df is None or len(df) == 0:
            return

        conn = get_connection(self.db_path)
        
        conn.execute('''
            INSERT OR REPLACE INTO domains.metadata
            (indicator_id, domain, name, source, category, frequency, 
             description, first_date, last_date, row_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [
            indicator.id,
            domain,
            indicator.name,
            indicator.source,
            indicator.category,
            indicator.frequency,
            indicator.description,
            df['date'].min(),
            df['date'].max(),
            len(df),
            datetime.now()
        ])
        
        conn.close()
    
    def run_agent(
        self, 
        domain: str, 
        delay: float = 1.0,
        indicators: Optional[List[str]] = None,
        store: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Run a single domain agent.
        
        Args:
            domain: Domain name (climate, epidemiology, etc.)
            delay: Seconds between requests
            indicators: Specific indicators (None = all)
            store: Whether to store to database
        
        Returns:
            Dict mapping indicator_id -> DataFrame
        """
        if domain not in self.agents:
            raise ValueError(f"Unknown domain: {domain}")
        
        agent = self.agents[domain]
        
        # Discover if needed
        if not agent.indicators:
            agent.discover()
        
        # Ensure schema
        if store:
            self._ensure_schema()
        
        # Fetch
        results = agent.fetch_all(delay=delay, indicators=indicators)
        
        # Store and update metadata
        if store:
            for ind_id, df in results.items():
                self._store_data(domain, ind_id, df)
                
                # Find indicator metadata
                ind_meta = next((i for i in agent.indicators if i.id == ind_id), None)
                if ind_meta:
                    self._update_metadata(domain, ind_meta, df)
        
        # Collect results
        self.all_results.extend(agent.fetch_results)
        
        return results
    
    def run_all_agents(
        self, 
        delay: float = 1.0,
        store: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Run all domain agents.
        
        Args:
            delay: Seconds between requests
            store: Whether to store to database
        
        Returns:
            Dict mapping domain -> indicator_id -> DataFrame
        """
        all_results = {}
        
        for domain in self.agents:
            print(f"\n{'='*50}")
            print(f"Running {domain.upper()} agent...")
            print(f"{'='*50}")
            
            results = self.run_agent(domain, delay=delay, store=store)
            all_results[domain] = results
        
        return all_results
    
    def get_domain_summary(self) -> pd.DataFrame:
        """Get summary of all domain data."""
        conn = get_connection(self.db_path)
        
        try:
            df = conn.execute('''
                SELECT 
                    domain,
                    COUNT(DISTINCT indicator_id) as n_indicators,
                    COUNT(*) as n_rows,
                    MIN(date) as first_date,
                    MAX(date) as last_date
                FROM domains.indicators
                GROUP BY domain
                ORDER BY domain
            ''').fetchdf()
            return df
        except:
            return pd.DataFrame()
        finally:
            conn.close()
    
    def print_summary(self):
        """Print summary of all fetch operations."""
        succeeded = sum(1 for r in self.all_results if r.success)
        failed = sum(1 for r in self.all_results if not r.success)
        total_rows = sum(r.rows for r in self.all_results)
        
        print(f"\n{'='*60}")
        print("ORCHESTRATOR SUMMARY")
        print(f"{'='*60}")
        print(f"Domains:     {len(self.agents)}")
        print(f"Indicators:  {succeeded + failed} attempted")
        print(f"Succeeded:   {succeeded}")
        print(f"Failed:      {failed}")
        print(f"Total rows:  {total_rows:,}")
        
        # Per-domain breakdown
        print(f"\nPer-domain:")
        for domain, agent in self.agents.items():
            agent_succeeded = sum(1 for r in agent.fetch_results if r.success)
            agent_failed = sum(1 for r in agent.fetch_results if not r.success)
            agent_rows = sum(r.rows for r in agent.fetch_results)
            print(f"  {domain}: {agent_succeeded} OK, {agent_failed} failed, {agent_rows:,} rows")
        
        # Database summary
        db_summary = self.get_domain_summary()
        if not db_summary.empty:
            print(f"\nDatabase state:")
            print(db_summary.to_string(index=False))
        
        # Failed indicators
        failed_results = [r for r in self.all_results if not r.success]
        if failed_results:
            print(f"\nFailed indicators:")
            for r in failed_results[:10]:
                print(f"  - {r.indicator_id}: {r.message}")
            if len(failed_results) > 10:
                print(f"  ... and {len(failed_results) - 10} more")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PRISM Agent Orchestrator")
    parser.add_argument("--domain", choices=["climate", "epidemiology", "all"], default="all")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--no-store", action="store_true", help="Don't store to database")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    orch = Orchestrator()
    
    if args.domain == "all":
        orch.run_all_agents(delay=args.delay, store=not args.no_store)
    else:
        orch.run_agent(args.domain, delay=args.delay, store=not args.no_store)
    
    orch.print_summary()
