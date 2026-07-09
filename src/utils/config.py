import os
from dataclasses import dataclass
from pathlib import Path
import yaml
from dotenv import load_dotenv

# load env vars from .env file if it exists
load_dotenv()

@dataclass(frozen=True)
class PathConfig:
    data_dir: Path
    raw_x_path: Path
    raw_y_path: Path
    reports_dir: Path

@dataclass(frozen=True)
class SchemaConfig:
    id_col: str
    target_col: str
    time_col: str
    turbine_col: str

@dataclass(frozen=True)
class FeatureSelectionConfig:
    mi_sample_size: int
    mi_threshold: float

@dataclass(frozen=True)
class PipelineConfig:
    paths: PathConfig
    schema: SchemaConfig
    feature_selection: FeatureSelectionConfig


def load_config(config_path: str = "config/config.yaml") -> PipelineConfig:
    """
    Parses config.yaml and env variables to construct a PipelineConfig object.
    """
    path_ref = Path(config_path)
    if not path_ref.exists():
        raise FileNotFoundError(f"Configuration file not found at: '{config_path}'")
        
    with open(path_ref, "r") as f:
        raw_config = yaml.safe_load(f)
        
    # resolve root data dir, favoring env overrides (Kaggle/Colab paths)
    env_data_dir = os.getenv("DATA_DIR")
    data_dir_str = env_data_dir if env_data_dir else raw_config["paths"]["data_dir"]
    data_dir = Path(data_dir_str)
    
    # construct PathConfig
    paths_dict = raw_config["paths"]
    paths_config = PathConfig(
        data_dir=data_dir,
        raw_x_path=data_dir / paths_dict["raw_x_file"],
        raw_y_path=data_dir / paths_dict["raw_y_file"],
        reports_dir=Path(paths_dict["reports_dir"])
    )
    
    # construct SchemaConfig
    schema_dict = raw_config["schema"]
    schema_config = SchemaConfig(
        id_col=schema_dict["id_col"],
        target_col=schema_dict["target_col"],
        time_col=schema_dict["time_col"],
        turbine_col=schema_dict["turbine_col"]
    )
    
    # construct FeatureSelectionConfig
    fs_dict = raw_config["feature_selection"]
    fs_config = FeatureSelectionConfig(
        mi_sample_size=int(fs_dict["mi_sample_size"]),
        mi_threshold=float(fs_dict["mi_threshold"])
    )
    
    return PipelineConfig(paths=paths_config, schema=schema_config, feature_selection=fs_config)