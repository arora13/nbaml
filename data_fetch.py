# data_fetch.py (robust)
import os
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "games.csv")

def fetch_games(start_season: int, end_season: int) -> pd.DataFrame:
    frames = []
    for season in range(start_season, end_season + 1):
        season_str = f"{season}-{str(season+1)[-2:]}"
        df = leaguegamefinder.LeagueGameFinder(season_nullable=season_str).get_data_frames()[0]

        # Parse types
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

        # Coerce TEAM_ID to numeric, drop rows without a valid id or date
        df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["TEAM_ID", "GAME_DATE"]).copy()
        df["TEAM_ID"] = df["TEAM_ID"].astype(int)

        # Keep regular/post-season team rows
        df = df[df["SEASON_ID"].astype(str).str.contains("2")].copy()

        frames.append(df)

    games = pd.concat(frames, ignore_index=True)

    keep = [
        "SEASON_ID","GAME_ID","GAME_DATE","MATCHUP","WL","TEAM_ID",
        "PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT",
        "OREB","DREB","REB","AST","TOV","STL","BLK","PF","PLUS_MINUS"
    ]
    # Some seasons may miss a couple fields; add them if absent
    for col in keep:
        if col not in games.columns:
            games[col] = pd.NA

    games = games[keep].sort_values("GAME_DATE").reset_index(drop=True)
    return games

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = fetch_games(start_season=2015, end_season=2024)
    # Final clean: drop rows missing critical columns we need
    df = df.dropna(subset=["GAME_ID","TEAM_ID","GAME_DATE","MATCHUP","WL"]).copy()
    df.to_csv(OUT_FILE, index=False)
    print(f"✅ Saved {len(df):,} rows to {OUT_FILE}")

if __name__ == "__main__":
    main()
