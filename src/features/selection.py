import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from src.utils.config import PipelineConfig
from src.utils.logger import setup_logger

logger = setup_logger()

def identify_low_relevance_features(df: pd.DataFrame, config: PipelineConfig) -> list[str]:
    """
    computes MI regression scores on a representative randomized sample of the dataset.
	returns a list of columns falling below the statistical information threshold.
    """
    id_col = config.schema.id_col
    target_col = config.schema.target_col
    time_col = config.schema.time_col
    turbine_col = config.schema.turbine_col
    
    # isolate candidates for feature selection (exclude target and non-predictive metadata)
    excluded_cols = [id_col, target_col, time_col, turbine_col]
    feature_candidates = [col for col in df.columns if col not in excluded_cols]
    
    # identify and log any non-numerical features before running MI (it only accepts numerical inputs)
    numerical_candidates = [col for col in feature_candidates if pd.api.types.is_numeric_dtype(df[col])]
    
    # extract a randomized sample to accelerate the k-NN entropy estimation
    sample_size = min(config.feature_selection.mi_sample_size, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    logger.info(f"computing MI scores using a sample size of {sample_size} records...")
    
    X_sample = df_sample[numerical_candidates].copy()
    y_sample = df_sample[target_col]
    
    # impute NaNs using median to prevent k-NN crashes (temporary)
    for col in X_sample.columns:
        if X_sample[col].isnull().any():
            median_val = X_sample[col].median()
            X_sample[col] = X_sample[col].fillna(median_val)
            
    # compute MI regression
    mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
    
    # map scores back to column names
    mi_series = pd.Series(mi_scores, index=numerical_candidates).sort_values(ascending=False)
    
    # filter features based on threshold config
    threshold = config.feature_selection.mi_threshold
    low_relevance_cols = list(mi_series[mi_series < threshold].index)
    
    logger.info("=== top 5 most informative features ===")
    for col, score in mi_series.head(5).items():
        logger.info(f"  -> {col:<40} : {score:.4f} bits")
        
    logger.info("=== bottom 5 least informative features ===")
    for col, score in mi_series.tail(5).items():
        logger.info(f"  -> {col:<40} : {score:.4f} bits")
        
    logger.info(f"identified {len(low_relevance_cols)} columns with MI < {threshold} bits.")
    return low_relevance_cols