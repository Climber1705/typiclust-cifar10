import yaml
from pathlib import Path
from typing import Any, Dict

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data