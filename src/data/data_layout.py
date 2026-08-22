"""Central path helpers for data, model, and result artifacts.

Keep experiment code using these helpers instead of building paths manually, so
real and synthetic outputs stay in the same folder structure.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.task.base_task import BaseTask


def get_project_root() -> Path:
    """Returns the root of this project.
    This is a helper function that is very useful for running notebooks (if they do not live in the
    main project folder, but in a subfolder like 'notebooks') 
    """
    # Start from the current location
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        # Based on the assumption that the project root is the only ancestor folder that contains "src", "scripts" and "configs" folders
        if (parent / "src").exists() and (parent / "scripts").exists() and (parent / "configs").exists():
            return parent
    print("Warning: Could not find project root. Does your project contain 'src', 'scripts', and 'configs' folders? Defaulting to current working directory.")
    return Path.cwd()

PROJECT_ROOT = get_project_root()
CONFIG_FOLDER = PROJECT_ROOT / "configs"
SOURCE_DATA_FOLDER = PROJECT_ROOT / "data/source"
RAW_DATA_FOLDER = PROJECT_ROOT / "data/raw"
SPLIT_DATA_FOLDER = PROJECT_ROOT / "data/split"

RESULT_FOLDER = PROJECT_ROOT / "results"
ADDITIONAL_RESULT_FOLDER = PROJECT_ROOT / "results_additional"
MODEL_FOLDER = PROJECT_ROOT / "models"
PLOT_FOLDER = PROJECT_ROOT / "plots"

class RealDataLayout:
    """Path setup for real-data experiments."""

    @staticmethod
    def get_raw_df_path(experiment_name: str, response_var: str):
        """Return the raw real-data parquet path for one response variable."""
        return f"{RAW_DATA_FOLDER}/real/{experiment_name}/{response_var}.parquet"

    @staticmethod
    def get_split_df_path(task: BaseTask, experiment_name: str, response_var=None, seed: int=None):
        """Return the split real-data parquet path for one task and seed."""
        return f"{SPLIT_DATA_FOLDER}/real/{experiment_name}/{response_var}/{task.name}/seed{seed}.parquet"
    
    @staticmethod
    def get_model_path(experiment_name: str, model_name: str, 
                       task: BaseTask, split_data_seed: int):
        """Return ``(folder, filename_stem)`` for a saved real-data model."""
        return f"{MODEL_FOLDER}/{experiment_name}/{model_name}", f"{model_name}-{task.name}-seed{split_data_seed}"

class SyntheticDataLayout:
    """Path setup for synthetic experiments grouped by fixed effect."""

    @staticmethod
    def get_raw_df_path(experiment_name: str, fixed_effect: str, seed: int):
        """Return the raw synthetic parquet path for one fixed effect and seed."""
        return f"{RAW_DATA_FOLDER}/synthetic/{experiment_name}/{fixed_effect}/seed{seed}.parquet"

    @staticmethod
    def get_split_df_path(task: BaseTask, experiment_name: str, fixed_effect: str, seed: int=None):
        """Return the split synthetic parquet path for one task and seed."""
        return f"{SPLIT_DATA_FOLDER}/synthetic/{experiment_name}/{fixed_effect}/{task.name}/seed{seed}.parquet"

    @staticmethod
    def get_model_path(experiment_name: str, model_name: str, fixed_effect_name: str,
                       task: BaseTask, split_data_seed: int):
        """Return ``(folder, filename_stem)`` for a saved synthetic model."""
        return f"{MODEL_FOLDER}/{experiment_name}/{fixed_effect_name}/{model_name}", f"{model_name}-{task.name}-seed{split_data_seed}"
