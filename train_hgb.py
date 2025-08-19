# train_hgb.py (robust to missing columns)
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from joblib import dump

from features import build_dataset  # your pipeline

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

DATA_CSV = Path("data/games.csv")

def load_team_rows_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["GAME_DATE", "TEAM_ID", "MATCHUP", "WL"]).copy()
    df["TEAM_ID"] = df["TEAM_ID"].astype(int)
    keep = [
        "SEASON_ID","GAME_ID","GAME_DATE","MATCHUP","WL","TEAM_ID",
        "PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT",
        "OREB","DREB","REB","AST","TOV","STL","BLK","PF","PLUS_MINUS"
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    return df[keep].sort_values("GAME_DATE").reset_index(drop=True)

def build_Xy(df: pd.DataFrame):
    y = df["HOME_WIN"].astype(int).values

    # Base features we try to use; if some are missing, we’ll create them as 0s.
    base_wanted = [
        "HOME_ELO_PRE","AWAY_ELO_PRE","HOME_ELO_EXP","ELO_DIFF",
        "DOW","MONTH",
        "REST_HOME","REST_AWAY","B2B_HOME","B2B_AWAY","REST_DIFF","B2B_DIFF",
        "REST_R5_HOME","REST_R5_AWAY","B2B_RATE_R5_HOME","B2B_RATE_R5_AWAY",
        "REST_R5_DIFF","B2B_RATE_R5_DIFF"
    ]
    # Add any missing base columns as zeros
    for col in base_wanted:
        if col not in df.columns:
            df[col] = 0.0

    # Rolling differentials: include whatever exists among R10/R30/SDT
    diff_cols = []
    for c in df.columns:
        if c.endswith("_DIFF_R10") or c.endswith("_DIFF_R30") or c.endswith("_DIFF_SDT"):
            diff_cols.append(c)

    X = df[base_wanted + diff_cols].copy()
    return X, y

def split_by_season(df: pd.DataFrame, holdout_start: int):
    s = df["SEASON_ID"].astype(str)
    train_df = df[~s.str.contains(str(holdout_start))].copy()
    test_df  = df[ s.str.contains(str(holdout_start))].copy()
    return train_df, test_df

def main(holdout=2024):
    # 1) Load team game logs
    team_rows = load_team_rows_from_csv(DATA_CSV)

    # 2) Build engineered dataset (game-level, HOME_WIN target)
    dataset = build_dataset(team_rows)

    # 3) Time-aware split
    tr, te = split_by_season(dataset, holdout_start=holdout)

    # 4) Features/labels
    Xtr, ytr = build_Xy(tr)
    Xte, yte = build_Xy(te)

    # 5) Train HGB
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_depth=6,
        max_iter=900,
        l2_regularization=0.0,
        validation_fraction=0.12,
        early_stopping=True,
        random_state=42,
    )
    clf.fit(Xtr, ytr)

    # 6) Evaluate
    proba = clf.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        "holdout_season": holdout,
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, proba)),
        "brier": float(brier_score_loss(yte, proba)),
        "log_loss": float(log_loss(yte, proba)),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }
    print(json.dumps(metrics, indent=2))
    (ART / "hgb_metrics.json").write_text(json.dumps(metrics, indent=2))

    # 7) Save model
    dump(clf, ART / "hgb_model.joblib")
    print("✅ Saved model to artifacts/hgb_model.joblib")

if __name__ == "__main__":
    main()
