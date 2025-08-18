# train_xgb.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from nba_api.stats.endpoints import leaguegamefinder
from features import build_dataset

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

def fetch_games(start_season: int, end_season: int) -> pd.DataFrame:
    frames = []
    for season in range(start_season, end_season + 1):
        season_str = f"{season}-{str(season+1)[-2:]}"
        df = leaguegamefinder.LeagueGameFinder(season_nullable=season_str).get_data_frames()[0]
        df = df[df["SEASON_ID"].astype(str).str.contains("2")].copy()
        frames.append(df)
    games = pd.concat(frames, ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    keep = ["SEASON_ID","GAME_ID","GAME_DATE","MATCHUP","WL","TEAM_ID",
            "PTS","AST","REB","TOV","FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS"]
    return games[keep].sort_values("GAME_DATE").reset_index(drop=True)

def build_Xy(df: pd.DataFrame):
    y = df["HOME_WIN"].astype(int).values
    base = [
        "HOME_ELO_PRE","AWAY_ELO_PRE","HOME_ELO_EXP","ELO_DIFF",
        "DOW","MONTH",
        "REST_HOME","REST_AWAY","B2B_HOME","B2B_AWAY","REST_DIFF","B2B_DIFF"
    ]
    diffs = [c for c in df.columns if c.endswith("_DIFF_R10") or c.endswith("_DIFF_R30") or c.endswith("_DIFF_SDT")]
    X = df[base + diffs].copy()
    return X, y, base + diffs if "return 3" in build_Xy.__code__.co_consts else (X, y)


def split_by_season(df: pd.DataFrame, holdout: int):
    train_df = df[~df["SEASON_ID"].astype(str).str.contains(str(holdout))].copy()
    test_df  = df[ df["SEASON_ID"].astype(str).str.contains(str(holdout))].copy()
    return train_df, test_df

def time_val_split(train_df: pd.DataFrame, val_frac=0.10):
    train_df = train_df.sort_values("GAME_DATE")
    cutoff = int(len(train_df) * (1 - val_frac))
    return train_df.iloc[:cutoff].copy(), train_df.iloc[cutoff:].copy()

def main(start_season=2018, end_season=2024, holdout=2024):
    raw = fetch_games(start_season, end_season)
    dataset = build_dataset(raw)
    tr_df, te_df = split_by_season(dataset, holdout)

    tr_core, tr_val = time_val_split(tr_df, val_frac=0.10)

    Xtr, ytr, feat_names = build_Xy(tr_core)
    Xva, yva, _          = build_Xy(tr_val)
    Xte, yte, _          = build_Xy(te_df)

    model = XGBClassifier(
        # capacity
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.85,
        # regularization
        reg_alpha=1.0,
        reg_lambda=2.0,
        gamma=0.0,
        # objective/metrics
        objective="binary:logistic",
        eval_metric=["auc","logloss"],  # track AUC and logloss
        # perf
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )

    # Version-proof early stopping: try callbacks, then legacy arg, then plain.
    fitted = False
    try:
        from xgboost import callback as xcb
        es = xcb.EarlyStopping(rounds=100, save_best=True, maximize=True)  # maximize AUC
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[es], verbose=False)
        fitted = True
    except Exception:
        try:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=100, verbose=False)
            fitted = True
        except Exception:
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            fitted = True

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "holdout_season": holdout,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, proba)),
        "brier": float(brier_score_loss(yte, proba)),
        "log_loss": float(log_loss(yte, proba)),
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte))
    }
    print(json.dumps(metrics, indent=2))

    import joblib
    ART.mkdir(exist_ok=True)
    (ART / "xgb_metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, ART / "xgb_model.joblib")
    (ART / "xgb_feature_names.json").write_text(json.dumps(feat_names, indent=2))

    # Save importances
    importances = model.feature_importances_.tolist()
    (ART / "xgb_feature_importances.json").write_text(
        json.dumps(dict(zip(feat_names, importances)), indent=2)
    )

if __name__ == "__main__":
    main()
