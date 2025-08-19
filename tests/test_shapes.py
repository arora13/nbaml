from pathlib import Path
import pytest
import pandas as pd
from features import build_dataset

def test_build_dataset_runs():
    p = Path("data/games.csv")
    if not p.exists():
        pytest.skip("data/games.csv missing (run data_fetch.py locally)")
    raw = pd.read_csv(p)
    ds = build_dataset(raw)
    assert "HOME_WIN" in ds.columns
    assert "HOME_ELO_PRE" in ds.columns
    assert len(ds) > 1000
