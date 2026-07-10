# main.py
import argparse
import sys
import yaml
import joblib
from pathlib import Path
import xgboost as xgb
from sklearn.pipeline import Pipeline

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.ingestion import load_and_merge_raw_data
from src.features.selection import run_feature_selection_pipeline
from src.data.splits import split_chronologically
from src.features.preprocessing import ColumnDropper, build_column_transformer
from src.models.metrics import calculate_evaluation_metrics
from src.models.tuning import tune_xgboost_hyperparameters

logger = setup_logger()

def execute_tuning() -> None:
    """executes the hyperparameter search and serializes results."""
    logger.info("starting automated hyperparameter optimization phase...")
    config = load_config()
    df = load_and_merge_raw_data(config)
    
    # isolate targets and features
    target_col = config.schema.target_col
    df_pruned = df.drop(columns=run_feature_selection_pipeline(df, config))
    
    df_train, _ = split_chronologically(df_pruned, config, train_ratio=0.8)
    y_train_raw = df_train[target_col].values
    
    # preprocess inputs to fit optimizer
    id_col = config.schema.id_col
    time_col = config.schema.time_col
    turbine_col = config.schema.turbine_col
    
    categorical_cols = [turbine_col]
    excluded = [id_col, target_col, time_col] + categorical_cols
    numerical_cols = [col for col in df_pruned.columns if col not in excluded]
    
    preprocessor = build_column_transformer(numerical_cols, categorical_cols)
    X_train_scaled = preprocess_data_frame_internal(df_train, preprocessor, numerical_cols, categorical_cols)
    
    tune_xgboost_hyperparameters(X_train_scaled.values, y_train_raw, config, n_iter=10)


def execute_training() -> None:
    """trains unified pipeline on training data and serializes the pipeline."""
    logger.info("starting automated production pipeline training phase...")
    config = load_config()
    df = load_and_merge_raw_data(config)
    
    # feature selection
    columns_to_drop = run_feature_selection_pipeline(df, config)
    
    # chronological value-based splitting
    df_train, df_test = split_chronologically(df, config, train_ratio=0.8)
    
    target_col = config.schema.target_col
    y_train_raw = df_train[target_col].values
    y_test_raw = df_test[target_col].values
    
    X_train_raw = df_train.drop(columns=[target_col])
    X_test_raw = df_test.drop(columns=[target_col])
    
    # prepare column groupings for preprocessor
    id_col = config.schema.id_col
    time_col = config.schema.time_col
    turbine_col = config.schema.turbine_col
    
    categorical_cols = [turbine_col]
    excluded = [id_col, time_col] + categorical_cols + columns_to_drop
    numerical_cols = [col for col in X_train_raw.columns if col not in excluded]
    
    # load tuned parameters with fallback to defaults
    tuned_params_path = Path("models/best_xgb_params.yaml")
    if tuned_params_path.exists():
        with open(tuned_params_path, "r") as f:
            xgb_kwargs = yaml.safe_load(f)
        logger.info(f"tuned hyperparameters successfully loaded from: '{tuned_params_path}'")
    else:
        logger.info("no tuned parameters found on disk. Initializing with configuration defaults.")
        xgb_kwargs = {
            "n_estimators": config.model_parameters.xgb_n_estimators,
            "learning_rate": config.model_parameters.xgb_learning_rate
        }
        
    xgb_kwargs["random_state"] = 42
    xgb_kwargs["n_jobs"] = -1
    
    # assemble unified pipeline
    xgb_model = xgb.XGBRegressor(**xgb_kwargs)
    production_pipeline = Pipeline(steps=[
        ("dropper", ColumnDropper(columns_to_drop=columns_to_drop)),
        ("preprocessor", build_column_transformer(numerical_cols, categorical_cols)),
        ("estimator", xgb_model)
    ])
    
    # fit and serialize
    logger.info("fitting unified pipeline on raw features...")
    production_pipeline.fit(X_train_raw, y_train_raw)
    
    pipeline_path = Path("models/turbine_xgb_pipeline.joblib")
    joblib.dump(production_pipeline, pipeline_path)
    logger.info(f"pipeline serialized successfully to: '{pipeline_path}'")
    
    # verification predict on raw test set
    y_pred = production_pipeline.predict(X_test_raw)
    metrics = calculate_evaluation_metrics(y_test_raw, y_pred, config.model_parameters.standby_min_kw)
    
    logger.info("=== retraining execution metrics ===")
    logger.info(f"  -> MAE (kW)                       : {metrics['MAE (kW)']:.4f} kW")
    logger.info(f"  -> RMSE (kW)                      : {metrics['RMSE (kW)']:.4f} kW")
    logger.info(f"  -> R2 Score                       : {metrics['R2 Score']:.4f}")


def preprocess_data_frame_internal(df, preprocessor, numerical_cols, categorical_cols):
    """helper method to isolate fitting from serialization during tuning runs."""
    feature_cols = numerical_cols + categorical_cols
    X = df[feature_cols]
    X_trans = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    cleaned_names = [name.split("__")[-1] for name in feature_names]
    return pd.DataFrame(X_trans, columns=cleaned_names, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wind Turbine Active Power Retraining and Optimization CLI."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tune", 
        action="store_true", 
        help="run TimeSeriesSplit cross-validation search for optimal hyperparameters."
    )
    group.add_argument(
        "--train", 
        action="store_true", 
        help="train unified production Pipeline and serialize to models/ directory."
    )
    
    args = parser.parse_args()
    
    if args.tune:
        execute_tuning()
    elif args.train:
        execute_training()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()