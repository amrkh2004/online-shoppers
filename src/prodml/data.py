"""
Data loading and preprocessing utilities for prodml.
"""

from pathlib import Path
from typing import Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

from prodml.config import RANDOM_STATE, TARGET_COL
from prodml.logging_conf import logger


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Loads dataset from CSV file path.
    """
    logger.info("Loading raw dataset", extra={"filepath": str(filepath)})
    df = pd.read_csv(filepath)
    logger.info("Dataset loaded successfully", extra={"shape": df.shape})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans duplicates and resets index.
    """
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    removed_rows = initial_rows - len(df_cleaned)
    logger.info(
        "Data cleaning completed",
        extra={"initial_rows": initial_rows, "cleaned_rows": len(df_cleaned), "removed_duplicates": removed_rows},
    )
    return df_cleaned


def split_data(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into stratified train/test sets.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from DataFrame")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        "Data train/test split completed",
        extra={"train_shape": X_train.shape, "test_shape": X_test.shape},
    )

    return X_train, X_test, y_train, y_test
