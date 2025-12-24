"""
PRISM Domain Agent Base Class
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IndicatorMeta:
    """Metadata for an indicator."""
    id: str
    name: str
    domain: str
    source: str
    category: str
    frequency: str
    start_date: Optional[str] = None
    description: Optional[str] = None


@dataclass
class FetchResult:
    """Result of a fetch operation."""
    indicator_id: str
    success: bool
    rows: int = 0
    message: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DomainAgent(ABC):
    """Base class for domain data agents."""
    
    domain: str = "base"
    sources: List[str] = []
    
    def __init__(self):
        self.indicators: List[IndicatorMeta] = []
        self.fetch_results: List[FetchResult] = []
    
    @abstractmethod
    def discover(self) -> List[IndicatorMeta]:
        """Find available indicators."""
        pass
    
    @abstractmethod
    def fetch_one(self, indicator_id: str) -> Optional[pd.DataFrame]:
        """Fetch single indicator."""
        pass
    
    def fetch_all(self, delay: float = 1.0, indicators: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Fetch all indicators with rate limiting."""
        import time
        
        if not self.indicators:
            self.discover()
        
        to_fetch = indicators or [i.id for i in self.indicators]
        results = {}
        
        for ind_id in to_fetch:
            try:
                logger.info(f"Fetching {ind_id} from {self.domain}")
                df = self.fetch_one(ind_id)
                
                if df is not None and len(df) > 0:
                    results[ind_id] = df
                    self.fetch_results.append(FetchResult(
                        indicator_id=ind_id, success=True, rows=len(df), message="OK"
                    ))
                else:
                    self.fetch_results.append(FetchResult(
                        indicator_id=ind_id, success=False, message="No data"
                    ))
            except Exception as e:
                logger.error(f"Failed to fetch {ind_id}: {e}")
                self.fetch_results.append(FetchResult(
                    indicator_id=ind_id, success=False, message=str(e)
                ))
            
            time.sleep(delay)
        
        return results
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Check data quality."""
        return df is not None and len(df) > 0
    
    def to_unified_schema(self, df: pd.DataFrame, indicator_id: str) -> pd.DataFrame:
        """Convert to standard schema."""
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame({
            "date": pd.to_datetime(df["date"]),
            "domain": self.domain,
            "indicator_id": indicator_id,
            "value": df["value"].astype(float)
        }).sort_values("date").reset_index(drop=True)
    
    def print_summary(self):
        """Print fetch summary."""
        succeeded = sum(1 for r in self.fetch_results if r.success)
        failed = sum(1 for r in self.fetch_results if not r.success)
        total_rows = sum(r.rows for r in self.fetch_results)
        
        print(f"\n{'='*50}")
        print(f"{self.domain.upper()} AGENT SUMMARY")
        print(f"{'='*50}")
        print(f"Succeeded: {succeeded}, Failed: {failed}, Rows: {total_rows:,}")
