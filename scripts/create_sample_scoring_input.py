from pathlib import Path

import pandas as pd


def main():
    base_dir = Path(__file__).resolve().parent.parent
    input_dir = base_dir / "data" / "scoring" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    sample_df = pd.DataFrame(
        {
            "distance_km": [3.5, 10.2, 1.8, 7.4, 15.0],
            "passengers": [1, 2, 1, 4, 3],
            "hour_of_day": [9, 17, 12, 21, 8],
        }
    )

    file_path = input_dir / "scoring_input.parquet"
    sample_df.to_parquet(file_path, index=False)
    print(f"Created sample batch scoring input at: {file_path}")


if __name__ == "__main__":
    main()
