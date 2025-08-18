# train_hgb.py
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss

from nba_api.stats.endpoints import leaguegamefinder
from features import build_dataset

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

# -------- data fetch (same as LR/XGB) --------
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

# -------- features -> X, y --------
def build_Xy(df: pd.DataFrame):
    y = df["HOME_WIN"].astype(int).values
    base = [
        "HOME_ELO_PRE","AWAY_ELO_PRE","HOME_ELO_EXP","ELO_DIFF",
        "DOW","MONTH",
        "REST_HOME","REST_AWAY","B2B_HOME","B2B_AWAY","REST_DIFF","B2B_DIFF",
    ]
    diffs = [c for c in df.columns if (
        c.endswith("_DIFF_R10") or c.endswith("_DIFF_R30") or c.endswith("_DIFF_SDT")
    )]
    X = df[base + diffs].copy()
    return X, y

def split_by_season(df: pd.DataFrame, holdout: int):
    train_df = df[~df["SEASON_ID"].astype(str).str.contains(str(holdout))].copy()
    test_df  = df[ df["SEASON_ID"].astype(str).str.contains(str(holdout))].copy()
    return train_df, test_df

def main(start_season=2018, end_season=2024, holdout=2024):
    raw = fetch_games(start_season, end_season)
    data = build_dataset(raw)
    tr, te = split_by_season(data, holdout)

    # time-aware (already sorted by GAME_DATE inside build_dataset merges, but re-sort to be safe)
    tr = tr.sort_values("GAME_DATE").reset_index(drop=True)
    te = te.sort_values("GAME_DATE").reset_index(drop=True)

    # small tail slice of train as validation via early_stopping built into HGB
    Xtr, ytr = build_Xy(tr)
    Xte, yte = build_Xy(te)

    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_depth=6,
        max_iter=900,
        l2_regularization=0.0,
        validation_fraction=0.12,   # last ~12% of training used as val
        early_stopping=True,
        random_state=42
    )
    clf.fit(Xtr, ytr)

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

if __name__ == "__main__":
    main()
