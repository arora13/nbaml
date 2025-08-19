from pathlib import Path
import os
import pytest
from predict import predict_from_text

def test_predict_smoke():
    if not Path("data/games.csv").exists():
        pytest.skip("data/games.csv missing (run data_fetch.py locally)")
    os.environ["PREDICT_SKIP_PLAYERS"] = "1"  # avoid nba_api call in CI
    out = predict_from_text("Golden State Warriors vs Los Angeles Lakers")
    assert "prediction" in out
    p = out["prediction"]["home_win_prob"]
    assert 0.0 <= p <= 1.0
