"""
DVC Stage 1: Data Preparation script.
"""

import pandas as pd
import yaml

from prodml.config import BASE_DIR
from prodml.data import clean_data, load_data, split_data
from prodml.logging_conf import logger


def main():
    logger.info("=== DVC Stage 1: Prepare Data ===")

    params_file = BASE_DIR / "params.yaml"
    with open(params_file, "r") as f:
        params = yaml.safe_load(f).get("prepare", {})

    test_size = params.get("test_size", 0.20)
    random_state = params.get("random_state", 42)

    raw_path = BASE_DIR / "notebooks" / "online_shoppers_intention.csv"
    if not raw_path.exists():
        possible_paths = list(BASE_DIR.glob("**/*.csv"))
        if possible_paths:
            raw_path = possible_paths[0]

    df = load_data(raw_path)
    df_cleaned = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(
        df_cleaned, test_size=test_size, random_state=random_state
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info("Saved processed train and test datasets", extra={"output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
