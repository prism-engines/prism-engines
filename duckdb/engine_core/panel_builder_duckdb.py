import duckdb, pandas as pd
from pathlib import Path

DUCKDB_PATH = Path(__file__).resolve().parents[2] / "duckdb" / "prism.duckdb"

class DuckDBPanelBuilder:
    def __init__(self):
        self.con = duckdb.connect(str(DUCKDB_PATH))

    def load_raw(self, path):
        return self.con.execute(f"SELECT * FROM read_csv_auto('{path}')").df()

    def build_panel(self, folder, pattern='*.csv'):
        folder = Path(folder)
        files = list(folder.glob(pattern))
        if not files:
            raise ValueError(f'No files found in {folder}')
        dfs = []
        for fp in files:
            df = self.load_raw(fp)
            df['source'] = fp.stem
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
