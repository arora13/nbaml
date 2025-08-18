# train.py
import json
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from joblib import dump

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
    keep = [
        "SEASON_ID","GAME_ID","GAME_DATE","MATCHUP","WL","TEAM_ID",
        "PTS","AST","REB","TOV","FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS"
    ]
    return games[keep].sort_values("GAME_DATE").reset_index(drop=True)

def build_Xy(df: pd.DataFrame):
    y = df["HOME_WIN"].astype(int).values
    base = [
        "HOME_ELO_PRE","AWAY_ELO_PRE","HOME_ELO_EXP","ELO_DIFF",
        "DOW","MONTH",
        "REST_HOME","REST_AWAY","B2B_HOME","B2B_AWAY","REST_DIFF","B2B_DIFF"
    ]
    diffs = [c for c in df.columns if c.endswith("_DIFF_R10")]
    X = df[base + diffs].copy()
    return X, y

def split_by_season(df: pd.DataFrame, holdout_season_start: int):
    train_df = df[~df["SEASON_ID"].astype(str).str.contains(str(holdout_season_start))]
    test_df  = df[ df["SEASON_ID"].astype(str).str.contains(str(holdout_season_start))]
    return train_df, test_df

def main(start_season=2018, end_season=2024, holdout=2024):
    raw = fetch_games(start_season, end_season)
    dataset = build_dataset(raw)
    tr, te = split_by_season(dataset, holdout)

    Xtr, ytr = build_Xy(tr)
    Xte, yte = build_Xy(te)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr_s, ytr)

    proba = clf.predict_proba(Xte_s)[:,1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "holdout_season": holdout,
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, proba)),
        "brier": float(brier_score_loss(yte, proba)),
        "log_loss": float(log_loss(yte, proba)),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte))
    }
    print(json.dumps(metrics, indent=2))

    dump(clf, ART / "model.joblib")
    dump(scaler, ART / "scaler.joblib")
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ART / "feature_cols.json").write_text(json.dumps(list(Xtr.columns), indent=2))

if __name__ == "__main__":
    main()
