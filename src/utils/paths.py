from pathlib import Path
from typing import Tuple

from src.config import Config

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def get_directories(config: Config) -> Tuple[Path, Path]:
    project_root = get_project_root()
    data_directory = (project_root / config.paths.data_directory).resolve()
    model_directory = (project_root / config.paths.model_directory).resolve()
    return data_directory, model_directory
