# train_xgb.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from nba_api.stats.endpoints import leaguegamefinder
from features import build_dataset

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# ---------- data fetch (same as your train.py) ----------
def fetch_games(start_season: int, end_season: int) -> pd.DataFrame:
    frames = []
    for season in range(start_season, end_season + 1):
        season_str = f"{season}-{str(season+1)[-2:]}"
        df = leaguegamefinder.LeagueGameFinder(season_nullable=season_str).get_data_frames()[0]
        df = df[df["SEASON_ID"].astype(str).str.contains("2")].copy()
        frames.append(df)
    games = pd.concat(frames, ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    keep = [
        "SEASON_ID","GAME_ID","GAME_DATE","MATCHUP","WL","TEAM_ID",
        "PTS","AST","REB","TOV","FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS"
    ]
    return games[keep].sort_values("GAME_DATE").reset_index(drop=True)

# ---------- feature matrix ----------
def build_Xy(df: pd.DataFrame):
    y = df["HOME_WIN"].astype(int).values
    base = [
        "HOME_ELO_PRE","AWAY_ELO_PRE","HOME_ELO_EXP","ELO_DIFF",
        "DOW","MONTH",
        "REST_HOME","REST_AWAY","B2B_HOME","B2B_AWAY","REST_DIFF","B2B_DIFF"
    ]
    diffs = [c for c in df.columns if c.endswith("_DIFF_R10")]
    X = df[base + diffs].copy()
    return X, y, base + diffs

def split_by_season(df: pd.DataFrame, holdout_season_start: int):
    train_df = df[~df["SEASON_ID"].astype(str).str.contains(str(holdout_season_start))].copy()
    test_df  = df[ df["SEASON_ID"].astype(str).str.contains(str(holdout_season_start))].copy()
    return train_df, test_df

# ---------- time-aware validation split for early stopping ----------
def time_val_split(train_df: pd.DataFrame, val_frac=0.12):
    train_df = train_df.sort_values("GAME_DATE")
    cutoff = int(len(train_df) * (1 - val_frac))
    return train_df.iloc[:cutoff].copy(), train_df.iloc[cutoff:].copy()

def main(start_season=2018, end_season=2024, holdout=2024):
    raw = fetch_games(start_season, end_season)
    dataset = build_dataset(raw)
    tr_df, te_df = split_by_season(dataset, holdout)

    # time-aware validation from training seasons (last ~12%)
    tr_core, tr_val = time_val_split(tr_df, val_frac=0.12)

    Xtr, ytr, feat_names = build_Xy(tr_core)
    Xva, yva, _          = build_Xy(tr_val)
    Xte, yte, _          = build_Xy(te_df)

    # XGBoost handles scaling fine, but scaling never hurts for stability with mixed features
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    # Solid starter params; tune later
    model = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        tree_method="hist"  # fast, CPU-friendly
    )

    model.fit(
        Xtr_s, ytr,
        eval_set=[(Xva_s, yva)],
        verbose=False,
        early_stopping_rounds=75
    )

    # Evaluate on holdout season
    proba = model.predict_proba(Xte_s)[:,1]
    pred  = (proba >= 0.5).astype(int)
    metrics = {
        "holdout_season": holdout,
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else None,
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, proba)),
        "brier": float(brier_score_loss(yte, proba)),
        "log_loss": float(log_loss(yte, proba)),
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte))
    }
    print(json.dumps(metrics, indent=2))

    # Save artifacts
    ART.mkdir(exist_ok=True)
    (ART / "xgb_metrics.json").write_text(json.dumps(metrics, indent=2))
    # Save model + scaler; XGBoost has its own save method but pickle is fine here
    import joblib
    joblib.dump(model, ART / "xgb_model.joblib")
    joblib.dump(scaler, ART / "xgb_scaler.joblib")
    (ART / "xgb_feature_names.json").write_text(json.dumps(feat_names, indent=2))

    # Optional: dump feature importances for your README plot
    importances = model.feature_importances_.tolist()
    (ART / "xgb_feature_importances.json").write_text(
        json.dumps(dict(zip(feat_names, importances)), indent=2)
    )

if __name__ == "__main__":
    main()
