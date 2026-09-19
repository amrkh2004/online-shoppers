"""
Feature processing and scaling utilities for prodml.
"""

from typing import List, Optional, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

from prodml.config import CATEGORICAL_FEATURES
from prodml.logging_conf import logger


class FeaturePipeline:
    """
    Feature transformation pipeline handling One-Hot Encoding alignment and StandardScaler.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.feature_columns: Optional[List[str]] = None
        self.is_fitted: bool = False

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies One-Hot Encoding to categorical features.
        """
        df_encoded = df.copy()

        if "Weekend" in df_encoded.columns and df_encoded["Weekend"].dtype == "bool":
            df_encoded["Weekend"] = df_encoded["Weekend"].astype(int)

        cols_to_encode = [
            col for col in CATEGORICAL_FEATURES if col in df_encoded.columns
        ]

        if cols_to_encode:
            df_encoded = pd.get_dummies(
                df_encoded, columns=cols_to_encode, drop_first=True, dtype=int
            )

        return df_encoded

    def fit_transform(self, X_train: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fits encoder feature columns and scaler on training data.
        """
        logger.info("Fitting feature pipeline on training data")
        X_encoded = self.preprocess(X_train)
        self.feature_columns = list(X_encoded.columns)

        X_scaled_arr = self.scaler.fit_transform(X_encoded)
        X_scaled_df = pd.DataFrame(
            X_scaled_arr, columns=self.feature_columns, index=X_train.index
        )

        self.is_fitted = True
        logger.info(
            "Feature pipeline fitted successfully",
            extra={"num_features": len(self.feature_columns)},
        )
        return X_scaled_df, self.feature_columns

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input data using stored feature columns and fitted scaler.
        """
        if not self.is_fitted or self.feature_columns is None:
            raise RuntimeError("FeaturePipeline must be fitted before calling transform.")

        X_encoded = self.preprocess(X)

        # Align columns with training features (fill missing encoded cols with 0)
        X_aligned = X_encoded.reindex(columns=self.feature_columns, fill_value=0)

        X_scaled_arr = self.scaler.transform(X_aligned)
        return pd.DataFrame(
            X_scaled_arr, columns=self.feature_columns, index=X.index
        )
