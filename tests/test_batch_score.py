import tempfile
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import MagicMock

from prodml.batch_score import batch_score


def test_batch_score_mock_model():
    df_input = pd.DataFrame({
        "distance_km": [5.2, 12.0, 3.1],
        "passengers": [1, 2, 1],
        "hour_of_day": [8, 14, 22],
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_file = tmp_path / "input.parquet"
        output_file = tmp_path / "output.parquet"

        df_input.to_parquet(input_file, index=False)

        mock_model = MagicMock()
        mock_model.predict.return_value = [15.5, 32.1, 10.0]

        df_result = batch_score(
            input_path=input_file,
            output_path=output_file,
            model_obj=mock_model,
        )

        assert output_file.exists()
        assert "prediction" in df_result.columns
        assert "run_date" in df_result.columns
        assert len(df_result) == 3
        assert list(df_result["prediction"]) == [15.5, 32.1, 10.0]
