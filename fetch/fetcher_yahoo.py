import pandas as pd
import yfinance as yf
import logging
from fetch.fetcher_base import BaseFetcher

logger = logging.getLogger(__name__)


class YahooFetcher(BaseFetcher):
    """
    Production-ready Yahoo Fetcher.

    Handles:
    - MultiIndex columns
    - Missing 'Close' columns (VIX, DXY, some indices)
    - Always returns: date, value, adjusted_value
    """

    def validate_response(self, df):
        return df is not None and not df.empty

    def _flatten(self, cols):
        out = []
        for c in cols:
            if isinstance(c, tuple):
                flat = "_".join([str(x) for x in c if x])
            else:
                flat = str(c)
            out.append(flat.lower())
        return out

    def _detect_close_column(self, df):
        """
        Try to find the best price column.
        Yahoo can return:
          - 'close'
          - 'close_*'
          - 'adjclose'
          - 'adj_close'
          - 'vixclose' (weird VIX cases)
        """
        candidates = [
            "close",
            "adjclose",
            "adj_close",
            "close_price",
            "last",
        ]

        # direct match
        for c in candidates:
            if c in df.columns:
                return c

        # anything containing 'close'
        for col in df.columns:
            if "close" in col:
                return col

        # last resort: first numeric column
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"[YahooFetcher] Using fallback column: {col}")
                return col

        raise ValueError("[YahooFetcher] Could not detect close column")

    def fetch_single(self, ticker, start_date=None, end_date=None, **kwargs):
        logger.info(f"[YahooFetcher] Requesting {ticker} ...")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )

        if not self.validate_response(df):
            logger.error(f"[YahooFetcher] Invalid response for {ticker}")
            return None

        df = df.reset_index()
        df.columns = self._flatten(df.columns)

        if "date" not in df.columns:
            raise ValueError(f"[YahooFetcher] No 'date' column for {ticker}")

        # pick the best available 'close-like' column
        close_col = self._detect_close_column(df)

        # pick adjusted close if present
        adj_col = "adjclose" if "adjclose" in df.columns else close_col

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df["date"])
        out["value"] = df[close_col]
        out["value_2"] = None
        out["adjusted_value"] = df[adj_col]

        logger.info(f"[YahooFetcher] {ticker} → {len(out)} rows")
        return out
