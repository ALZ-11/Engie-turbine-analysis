import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from src.utils.config import PipelineConfig
from src.utils.logger import setup_logger

logger = setup_logger()

class TurbineTargetScaler:
    """
    robust target scaling wrapper designed to support gradient-based 
    deep learning optimization by transforming active power targets to stable ranges.
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self._is_fitted = False

    def fit(self, y: pd.Series | np.ndarray) -> "TurbineTargetScaler":
        y_arr = np.array(y).reshape(-1, 1)
        self.scaler.fit(y_arr)
        self._is_fitted = True
        logger.info("TurbineTargetScaler successfully fitted on targets.")
        return self

    def transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("scaler must be fitted before transforming targets.")
        y_arr = np.array(y).reshape(-1, 1)
        return self.scaler.transform(y_arr).flatten()

    def fit_transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y_scaled: pd.Series | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("scaler must be fitted before inverse-transforming targets.")
        y_arr = np.array(y_scaled).reshape(-1, 1)
        return self.scaler.inverse_transform(y_arr).flatten()


def build_column_transformer(numerical_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    constructs an isolated, serializable ColumnTransformer for clean feature scaling
    and category processing.
    """
    logger.info("constructing ColumnTransformer pipelines...")
    
    # numerical pipeline: median imputation followed by robust scaling
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    # categorical pipeline: one-hot encoding
    cat_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])
    
    # combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ],
        remainder="drop" # drop any unselected/metadata columns explicitly
    )
    
    return preprocessor


def preprocess_data_frame(
    df: pd.DataFrame, 
    preprocessor: ColumnTransformer, 
    numerical_cols: list[str], 
    categorical_cols: list[str],
    fit: bool = False
) -> pd.DataFrame:
    """
    applies ColumnTransformer to a Pandas DataFrame, reconstructing a 
    Pandas DataFrame with readable column names to prevent alignment errors.
    """
    feature_cols = numerical_cols + categorical_cols
    X = df[feature_cols]
    
    if fit:
        logger.info("fitting ColumnTransformer on features...")
        X_trans = preprocessor.fit_transform(X)
    else:
        X_trans = preprocessor.transform(X)
        
    # extract structural column names dynamically to preserve alignment
    feature_names = preprocessor.get_feature_names_out()
    
    # clean generated column prefixes (e.g., 'num__Rotor_speed' -> 'Rotor_speed')
    cleaned_names = [name.split("__")[-1] for name in feature_names]
    
    # reconstruct DataFrame with preserved indexes
    df_transformed = pd.DataFrame(X_trans, columns=cleaned_names, index=df.index)
    return df_transformed