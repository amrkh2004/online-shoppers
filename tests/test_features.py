import pandas as pd
import pytest
from prodml.features import FeaturePipeline


def test_feature_pipeline_fit_transform():
    df = pd.DataFrame(
        {
            "Administrative": [1, 2, 3],
            "Month": ["May", "Nov", "May"],
            "Weekend": [True, False, True],
        }
    )
    pipeline = FeaturePipeline()
    X_scaled, features = pipeline.fit_transform(df)

    assert pipeline.is_fitted
    assert len(features) > 0
    assert X_scaled.shape[0] == 3


def test_feature_pipeline_transform_unfitted():
    pipeline = FeaturePipeline()
    df = pd.DataFrame({"Administrative": [1]})
    with pytest.raises(RuntimeError, match="FeaturePipeline must be fitted"):
        pipeline.transform(df)
