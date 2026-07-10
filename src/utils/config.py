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
    spearman_threshold: float

@dataclass(frozen=True)
class SequenceConfig:
    lookback_steps: int
    step_size: float

@dataclass(frozen=True)
class ModelParametersConfig:
    standby_min_kw: float
    rf_n_estimators: int
    rf_max_depth: int | None
    xgb_n_estimators: int
    xgb_learning_rate: float
    dl_epochs: int
    dl_batch_size: int
    dl_early_stopping_patience: int
    dl_learning_rate: float

@dataclass(frozen=True)
class PipelineConfig:
    paths: PathConfig
    schema: SchemaConfig
    feature_selection: FeatureSelectionConfig
    sequence_parameters: SequenceConfig
    model_parameters: ModelParametersConfig


def load_config(config_path: str = "config/config.yaml") -> PipelineConfig:
    """
    parses config.yaml and env variables to construct PipelineConfig object.
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
        mi_threshold=float(fs_dict["mi_threshold"]),
        spearman_threshold=float(fs_dict["spearman_threshold"])
    )
    
    # construct SequenceConfig
    seq_dict = raw_config["sequence_parameters"]
    seq_config = SequenceConfig(
        lookback_steps=int(seq_dict["lookback_steps"]),
        step_size=float(seq_dict["step_size"])
    )
    
    # construct ModelParametersConfig
    mp_dict = raw_config["model_parameters"]
    max_depth_val = mp_dict["rf_max_depth"]
    rf_max_depth = int(max_depth_val) if max_depth_val is not None else None
    
    mp_config = ModelParametersConfig(
        standby_min_kw=float(mp_dict["standby_min_kw"]),
        rf_n_estimators=int(mp_dict["rf_n_estimators"]),
        rf_max_depth=rf_max_depth,
        xgb_n_estimators=int(mp_dict["xgb_n_estimators"]),
        xgb_learning_rate=float(mp_dict["xgb_learning_rate"]),
        dl_epochs=int(mp_dict["dl_epochs"]),
        dl_batch_size=int(mp_dict["dl_batch_size"]),
        dl_early_stopping_patience=int(mp_dict["dl_early_stopping_patience"]),
        dl_learning_rate=float(mp_dict["dl_learning_rate"])
    )
    
    return PipelineConfig(
        paths=paths_config,
        schema=schema_config,
        feature_selection=fs_config,
        sequence_parameters=seq_config,
        model_parameters=mp_config
    )