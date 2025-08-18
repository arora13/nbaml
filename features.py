# features.py
import pandas as pd
import numpy as np

def team_to_game_rows(team_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-level rows (one row per team per game) into game-level rows
    (one row per game with home/away split and result).
    """
    tr = team_rows.copy()

    # Ensure MATCHUP is a string and handle missing values safely
    tr["MATCHUP"] = tr["MATCHUP"].astype(str)
    tr["IS_HOME"] = tr["MATCHUP"].str.contains(" vs. ", na=False)

    feats = ["PTS","AST","REB","TOV","FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS"]

    home = tr[tr["IS_HOME"]].copy()
    away = tr[~tr["IS_HOME"]].copy()

    game = home[["GAME_ID","TEAM_ID","GAME_DATE","SEASON_ID","WL"] + feats].merge(
        away[["GAME_ID","TEAM_ID"] + feats],
        on="GAME_ID",
        suffixes=("_HOME","_AWAY")
    )
    game = game.rename(columns={"TEAM_ID_HOME":"HOME_ID","TEAM_ID_AWAY":"AWAY_ID"})
    game["HOME_WIN"] = (game["WL"] == "W").astype(int)

    # Keep a stable sort
    return game.sort_values("GAME_DATE").reset_index(drop=True)

def add_rolling(team_rows: pd.DataFrame, window=10) -> pd.DataFrame:
    """
    Compute shifted rolling means so we use only information prior to each game (no leakage).
    """
    tr = team_rows.sort_values(["TEAM_ID","GAME_DATE"]).copy()
    metrics = ["PTS","AST","REB","TOV","FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS"]

    for m in metrics:
        tr[f"{m}_R{window}"] = (
            tr.groupby("TEAM_ID")[m]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
        )

    tr["GAMES_PLAYED"] = tr.groupby("TEAM_ID").cumcount()
    return tr

def add_elo(game_rows: pd.DataFrame, k=20, home_adv=65) -> pd.DataFrame:
    """
    Simple Elo updated sequentially over time.
    Returns pre-game Elo and expected home win probability for each game.
    """
    df = game_rows.copy()
    teams = pd.unique(pd.concat([df["HOME_ID"], df["AWAY_ID"]]))
    elo = {int(t): 1500.0 for t in teams}
    pre = []

    for _, r in df.sort_values("GAME_DATE").iterrows():
        h, a = int(r["HOME_ID"]), int(r["AWAY_ID"])
        Rh, Ra = elo[h], elo[a]
        Eh = 1.0 / (1.0 + 10 ** (-(Rh + home_adv - Ra) / 400))
        pre.append({
            "GAME_ID": r["GAME_ID"],
            "HOME_ELO_PRE": Rh,
            "AWAY_ELO_PRE": Ra,
            "HOME_ELO_EXP": Eh
        })
        out = int(r["HOME_WIN"])
        elo[h] = Rh + k * (out - Eh)
        elo[a] = Ra + k * ((1 - out) - (1 - Eh))

    pre = pd.DataFrame(pre)
    return df.merge(pre, on="GAME_ID")

def build_dataset(team_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Master dataset builder:
    - Rolling team form (R10) joined onto game rows for home/away
    - Elo features
    - Calendar features
    - Differentials (home - away) for rolling stats and Elo
    """
    # Rolling stats per team, shifted
    tr = add_rolling(team_rows, window=10)

    # Game rows (home/away split + result)
    game = team_to_game_rows(team_rows)

    # Join rolling features for home and away
    base = ["TEAM_ID","GAME_ID"]
    roll_cols = [c for c in tr.columns if c.endswith("_R10")]

    home = tr[base + roll_cols].rename(columns={"TEAM_ID":"HOME_ID"})
    home = home.rename(columns={c: f"{c}_HOME" for c in roll_cols})

    away = tr[base + roll_cols].rename(columns={"TEAM_ID":"AWAY_ID"})
    away = away.rename(columns={c: f"{c}_AWAY" for c in roll_cols})

    df = game.merge(home, on=["GAME_ID","HOME_ID"], how="left")
    df = df.merge(away, on=["GAME_ID","AWAY_ID"], how="left")

    # Elo features
    df = add_elo(df)

    # Calendar context
    df["DOW"] = df["GAME_DATE"].dt.dayofweek
    df["MONTH"] = df["GAME_DATE"].dt.month

    # Drop rows with NaNs from early-season rolling gaps
    df = df.dropna().reset_index(drop=True)

    # Rolling differentials (home - away)
    for m in ["PTS","AST","REB","TOV","FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS"]:
        df[f"{m}_DIFF_R10"] = df[f"{m}_R10_HOME"] - df[f"{m}_R10_AWAY"]

    # Elo differential
    df["ELO_DIFF"] = df["HOME_ELO_PRE"] - df["AWAY_ELO_PRE"]

    return df
