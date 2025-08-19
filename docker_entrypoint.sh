#!/usr/bin/env bash
set -euo pipefail

mkdir -p data artifacts

if [ ! -f data/games.csv ]; then
  echo "↺ No data/games.csv found — fetching..."
  python data_fetch.py || true
fi

if [ ! -f artifacts/hgb_model.joblib ]; then
  echo "↺ No model found — training..."
  python train_hgb.py || true
fi

echo "▶ Starting API on :8000"
exec uvicorn app:app --host 0.0.0.0 --port 8000
