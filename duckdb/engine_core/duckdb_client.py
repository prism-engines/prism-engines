import duckdb
from pathlib import Path

class DuckDBClient:
    def __init__(self, path='duckdb/prism.duckdb'):
        self.path = Path(path)
        self.con = duckdb.connect(str(self.path))

    def query(self, sql):
        return self.con.execute(sql).df()
