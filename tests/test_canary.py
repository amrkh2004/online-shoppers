from pathlib import Path
from canary.rollout_manager import update_nginx_weights, STAGES

def test_canary_stages_config():
    assert len(STAGES) == 4
    assert STAGES[0]["blue_weight"] == 95
    assert STAGES[0]["green_weight"] == 5
    assert STAGES[3]["blue_weight"] == 0
    assert STAGES[3]["green_weight"] == 100

def test_update_nginx_weights(tmp_path):
    conf_file = tmp_path / "nginx.conf"
    conf_file.write_text("""
    upstream model_backend {
        server blue:8000 weight=95;
        server green:8000 weight=5;
    }
    """)

    update_nginx_weights(blue_weight=80, green_weight=20, conf_file=conf_file)
    content = conf_file.read_text()
    assert "weight=80;" in content
    assert "weight=20;" in content
