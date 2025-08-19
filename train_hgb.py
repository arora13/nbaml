# train_hgb.py — trains HGB and fits an isotonic calibrator
import json
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss

from features import build_dataset

ART = Path("artifacts"); ART.mkdir(exist_ok=True)
DATA = Path("data/games.csv")

def load_team_rows_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["GAME_DATE","TEAM_ID","MATCHUP","WL"]).copy()
    df["TEAM_ID"] = df["TEAM_ID"].astype(int)
    keep = ["SEASON_ID","GAME_ID","GAME_DATE","MATCHUP","WL","TEAM_ID",
            "PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT",
            "OREB","DREB","REB","AST","TOV","STL","BLK","PF","PLUS_MINUS"]
    for c in keep:
        if c not in df.columns: df[c] = pd.NA
    return df[keep].sort_values("GAME_DATE").reset_index(drop=True)

def build_Xy(df: pd.DataFrame):
    """Return (X, y) with robust column set. Missing engineered cols are zero-filled."""
    y = df["HOME_WIN"].astype(int).values
    base_wanted = [
        "HOME_ELO_PRE","AWAY_ELO_PRE","HOME_ELO_EXP","ELO_DIFF",
        "DOW","MONTH",
        "REST_HOME","REST_AWAY","B2B_HOME","B2B_AWAY","REST_DIFF","B2B_DIFF",
        "REST_R5_HOME","REST_R5_AWAY","B2B_RATE_R5_HOME","B2B_RATE_R5_AWAY",
        "REST_R5_DIFF","B2B_RATE_R5_DIFF",
        # If you later add Four Factors, these will be present; else they stay as zeros.
        "EFG_PCT_HOME","EFG_PCT_AWAY","TOV_PCT_HOME","TOV_PCT_AWAY",
        "ORB_PCT_HOME","ORB_PCT_AWAY","FTR_HOME","FTR_AWAY","PACE_HOME","PACE_AWAY",
        "EFG_PCT_DIFF_R10","TOV_PCT_DIFF_R10","ORB_PCT_DIFF_R10","FTR_DIFF_R10","PACE_DIFF_R10",
    ]
    for col in base_wanted:
        if col not in df.columns:
            df[col] = 0.0
    diffs = [c for c in df.columns if c.endswith("_DIFF_R10") or c.endswith("_DIFF_R30") or c.endswith("_DIFF_SDT")]
    X = df[base_wanted + diffs].copy()
    return X, y

def split_by_season(df: pd.DataFrame, holdout_start: int):
    s = df["SEASON_ID"].astype(str)
    train_df = df[~s.str.contains(str(holdout_start))].copy()
    test_df  = df[ s.str.contains(str(holdout_start))].copy()
    return train_df, test_df

def time_val_split(df: pd.DataFrame, frac=0.12):
    """Chronological split for validation from the tail of training."""
    n = len(df)
    k = max(1, int(n * (1.0 - frac)))
    return df.iloc[:k].copy(), df.iloc[k:].copy()

def main(holdout=2024):
    raw = load_team_rows_from_csv(DATA)
    ds  = build_dataset(raw)

    tr_full, te = split_by_season(ds, holdout_start=holdout)
    tr, va = time_val_split(tr_full, frac=0.12)

    Xtr, ytr = build_Xy(tr)
    Xva, yva = build_Xy(va)
    Xte, yte = build_Xy(te)

    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_depth=6,
        max_iter=900,
        validation_fraction=0.12,
        early_stopping=True,
        random_state=42,
    )
    clf.fit(Xtr, ytr)

    # Isotonic calibration on validation probs
    p_va = clf.predict_proba(Xva)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, yva)

    # Evaluate on holdout with calibrated probs
    p_te_raw = clf.predict_proba(Xte)[:, 1]
    p_te = iso.predict(p_te_raw)
    pred = (p_te >= 0.5).astype(int)

    metrics = {
        "holdout_season": holdout,
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, p_te)),
        "brier": float(brier_score_loss(yte, p_te)),
        "log_loss": float(log_loss(yte, p_te)),
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
    }
    print(json.dumps(metrics, indent=2))
    ART.mkdir(exist_ok=True)
    (ART / "hgb_metrics.json").write_text(json.dumps(metrics, indent=2))

    dump(clf, ART / "hgb_model.joblib")
    dump(iso, ART / "isotonic.joblib")
    (ART / "model_info.json").write_text(json.dumps({
        "model": "HistGradientBoostingClassifier",
        "feature_count": int(Xtr.shape[1]),
        "holdout_season": holdout,
        "features_sample": Xtr.columns[:12].tolist(),
    }, indent=2))
    print("✅ Saved model + isotonic to artifacts/")

if __name__ == "__main__":
    main()
