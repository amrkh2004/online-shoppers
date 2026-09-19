import pandas as pd
import pytest
from prodml.data import clean_data, split_data


def test_clean_data():
    raw_df = pd.DataFrame(
        {
            "Administrative": [1, 1, 2],
            "Revenue": [0, 0, 1],
        }
    )
    cleaned = clean_data(raw_df)
    assert len(cleaned) == 2


def test_split_data():
    df = pd.DataFrame(
        {
            "Administrative": list(range(20)),
            "Revenue": [0] * 10 + [1] * 10,
        }
    )
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20)
    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_split_data_missing_target():
    df = pd.DataFrame({"Administrative": [1, 2]})
    with pytest.raises(ValueError, match="Target column 'Revenue' missing"):
        split_data(df)
