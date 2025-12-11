import yaml
from pathlib import Path

class ConfigRegistry:
    def __init__(self, config_root='config'):
        self.root = Path(config_root)

    def load(self, name):
        path = self.root / f'{name}.yaml'
        with open(path) as f:
            return yaml.safe_load(f)
