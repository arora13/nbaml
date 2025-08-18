import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

df = leaguegamefinder.LeagueGameFinder(season_nullable="2022-23").get_data_frames()[0]
print(df.head())
