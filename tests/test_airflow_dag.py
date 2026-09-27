import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure dags directory is in path
dags_dir = Path(__file__).resolve().parent.parent / "dags"
sys.path.insert(0, str(dags_dir))


def test_airflow_dag_import():
    try:
        from dags.retrain_pipeline import DEFAULT_ARGS, dag

        assert DEFAULT_ARGS["owner"] == "mlops_team"
        assert DEFAULT_ARGS["retries"] == 2
        assert dag is not None
        assert dag.dag_id == "ride_duration_retrain_pipeline"
    except ImportError:
        # If airflow is not present in local python environment, mock DAG
        with patch.dict(
            "sys.modules", {"airflow": MagicMock(), "airflow.operators.python": MagicMock()}
        ):
            from dags.retrain_pipeline import DEFAULT_ARGS

            assert DEFAULT_ARGS["owner"] == "mlops_team"
            assert DEFAULT_ARGS["retries"] == 2
