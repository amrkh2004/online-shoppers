import pandas as pd

from prodml.train import evaluate_model, train_baseline_model, train_tuned_model


def test_train_and_evaluate():
    X = pd.DataFrame(
        {
            "feat_1": [0.1, 0.5, 0.9, 1.2, 0.3, 0.8, 1.5, 0.2],
            "feat_2": [1.1, 0.4, 0.7, 0.1, 0.9, 0.4, 0.3, 1.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    _ = train_baseline_model(X, y)
    model_tuned = train_tuned_model(X, y)

    metrics = evaluate_model(model_tuned, X, y)
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
