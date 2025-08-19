# tune_hgb.py
import json
import argparse
from pathlib import Path

import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

from train_hgb import load_team_rows_from_csv, build_Xy
from features import build_dataset

DATA = Path("data/games.csv")
ART = Path("artifacts"); ART.mkdir(exist_ok=True)

def split_by_season(df: pd.DataFrame, holdout_start: int):
    s = df["SEASON_ID"].astype(str)
    train_df = df[~s.str.contains(str(holdout_start))].copy()
    test_df  = df[ s.str.contains(str(holdout_start))].copy()
    return train_df, test_df

def time_val_split(df: pd.DataFrame, frac=0.12):
    n = len(df)
    k = max(1, int(n * (1.0 - frac)))
    return df.iloc[:k].copy(), df.iloc[k:].copy()

def objective_factory(Xtr, ytr, Xva, yva):
    def obj(trial: optuna.Trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_iter=trial.suggest_int("max_iter", 300, 1200),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            l2_regularization=trial.suggest_float("l2_regularization", 0.0, 2.0),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 200),
        )
        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            early_stopping=False,   # evaluate on our own validation split
            random_state=42,
            **params
        )
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xva)[:, 1]
        score = roc_auc_score(yva, p) - 0.5 * brier_score_loss(yva, p)
        return score
    return obj

def main(holdout: int, trials: int):
    raw = load_team_rows_from_csv(DATA)
    ds  = build_dataset(raw)

    tr_full, _ = split_by_season(ds, holdout_start=holdout)
    tr, va     = time_val_split(tr_full, frac=0.12)

    Xtr, ytr = build_Xy(tr)
    Xva, yva = build_Xy(va)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_factory(Xtr, ytr, Xva, yva), n_trials=trials, show_progress_bar=True)

    best = study.best_params
    best.setdefault("loss", "log_loss")

    out = ART / "hgb_optuna.json"
    out.write_text(json.dumps(best, indent=2))
    print("\nBest params saved to", out)
    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=2024, help="Season start year for holdout (e.g., 2024)")
    ap.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    args = ap.parse_args()
    main(args.holdout, args.trials)
