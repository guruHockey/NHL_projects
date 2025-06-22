import pandas as pd
from google.cloud import bigquery
import google.auth
import numpy as np
import openpyxl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

project = 'hockey-451914'

credentials, project = google.auth.default()

client = bigquery.Client()
"""
Summary: random forrest predctive model for declared value.

Version control:
    v1. used random forest model to predict average points per game for the next three seasons
    v2. added gridsearch tuning - tuning model to search for the optimal number of trees
"""

# query1 = """
# (
# with players as (
#   SELECT
#     ss.*, 
#     sh.usat_fenwick_pct,
#     sh.sat_corsiFor_pct,
#     sb.seasonAge2425, 
#     COUNT(ss.season) OVER (PARTITION BY ss.playerId) as season_count
#   FROM `hockey-451914.guruHockey11782.2020_2025skater_summary` ss
#     LEFT JOIN `hockey-451914.guruHockey11782.2020_2025skater_bios` sb ON ss.playerId=sb.playerId
#       LEFT JOIN `hockey-451914.guruHockey11782.2020_2025summaryshootingHistorical` sh ON ss.playerId=sh.playerId AND ss.season = sh.season
# )

# select * from players WHERE season_count = 5

# )
# """

# df_allstats= client.query(query1).to_dataframe()

"""
    query1 is the query I used to create this dataset based on a datamart that I created from
    NHL public facing API data.  This data can be found in the .xlsx named predictive_modeling_data.xlsx
"""
# place your file path here:
your_file_path = '/home/michael/Desktop/'

# reads xlsx file from local directory
df_allstats = pd.read_excel(f'{your_file_path}predictive_modeling_data.xlsx')

# available stats
df_allstats= df_allstats[[
    'skaterFullName',
    'season',
    'goals',
    'assists',
    'points',
    'gamesPlayed',
    'shots',
    'timeOnIcePerGame',
    'plusMinus',
    'penaltyMinutes', 
    'pointsPerGame',  
    'usat_fenwick_pct',
    'sat_corsiFor_pct',
    'seasonAge2425']]

# declare variable for anaylsis
statVar = 'pointsPerGame'

# feature engineering
# converting timeOnIcePerGame to minutes rather than seconds
df_allstats['timeOnIcePerGame'] = df_allstats['timeOnIcePerGame'] / 60  # convert seconds to minutes

# creating function to convert raw values to average per game
def convert_to_per_game(df, column):
    if column in df.columns and 'gamesPlayed' in df.columns:
        df[column + '_per_game'] = df[column] / df['gamesPlayed'].replace(0, pd.NA)
    return df

columns_to_convert = ['goals', 'assists', 'points', 'shots']  

convert_to_per_game(df_allstats, 'goals')
convert_to_per_game(df_allstats, 'assists')
convert_to_per_game(df_allstats, 'shots')
convert_to_per_game(df_allstats, 'penaltyMinutes')

# pivot dataframe to flatten so each player is one row, each season is a column
#     this will allow the use of training data as independent varables for the model
df_player = df_allstats.pivot_table(
    index='skaterFullName',
    columns='season',
    values=[
    'goals_per_game',
    'assists_per_game',
    'gamesPlayed',
    'pointsPerGame',  
    'shots_per_game', 
    'timeOnIcePerGame',
    'plusMinus',
    'penaltyMinutes_per_game', 
    'usat_fenwick_pct',
    'sat_corsiFor_pct',
    'seasonAge2425'],
    aggfunc='sum'
).fillna(0)

# append the statVar to the column names
df_player.columns = [f"{stat}_{season}" for stat, season in df_player.columns]

# including derived percentage metrics and excluding skaterFullName, season, and statVar from model training
metrics = [col for col in df_allstats.columns if col not in ['skaterFullName', 'season', statVar]]

# aggregate all other player metrics except skaterFullName season and statVar
numeric_metrics = df_allstats[metrics].select_dtypes(include='number').columns.tolist()
df_metrics = df_allstats.groupby('skaterFullName')[numeric_metrics].mean()
df_player = df_player.merge(df_metrics, left_index=True, right_index=True, how='left')

df_player = df_player.reset_index()

# define seasons for training and predictions
seasons = [f"{statVar}_{season}" for season in ['20202021', '20212022', '20222023', '20232024', '20242025']]
future_seasons = [f"predicted_{statVar}_{season}" for season in ['202526', '202627', '202728']]

# prepare data for training using time series
    # X is the first four seasons and y is the target season
X = df_player[seasons[:-1]]
y = df_player[seasons[-1]]

# ***v2 gridsearch tuning - tuning model to search for the optimal number of trees ***
    # setting paramaters for trees and depth
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20]
}

    # grid search to find the best parameters
    # model = RandomForestRegressor
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
grid.fit(X, y)
    # predicting 20242025 based on the best model selected from grid search
best_model = grid.best_estimator_
df_player['predicted_20242025'] = best_model.predict(X)

# predicting the next three seasons using autoregression
future_X = df_player[seasons[1:]].copy() 
predictions = pd.DataFrame(index=df_player.index)

for season in future_seasons:
    future_X.columns = X.columns

    # predict next season
    pred = best_model.predict(future_X)
    predictions[season] = pred

    # shift window for predictions
    future_X = future_X.shift(-1, axis=1)
    future_X.iloc[:, -1] = pred

# combining predictions and player data
df_player = pd.concat([df_player, predictions], axis=1)

# cleaning up final dataframe
df_finalPredictions = df_player[[
    'skaterFullName',
    'gamesPlayed_20202021',
    'gamesPlayed_20212022',
    'gamesPlayed_20222023',
    'gamesPlayed_20232024',
    'gamesPlayed_20242025',
    'pointsPerGame_20202021',
    'pointsPerGame_20212022',
    'pointsPerGame_20222023',	
    'pointsPerGame_20232024',
    'pointsPerGame_20242025',
    'predicted_20242025',
    'predicted_pointsPerGame_202526',
    'predicted_pointsPerGame_202627',
    'predicted_pointsPerGame_202728'
    ]]

print(df_finalPredictions)

# evaluating model performance
mse = mean_squared_error(y, df_player['predicted_20242025'])
r2 = r2_score(y, df_player['predicted_20242025'])

print("Mean Squared Error:", mse)
print("R² Score:", r2)
print("Best parameters:", grid.best_params_)

# for output, references your above filepath. output will create a new tab in the same .xlsx
Template_Path = (f'{your_file_path}predictive_modeling_data.xlsx')

# # Define context manager using ExcelWriter function and store it as the variable "writer"
with pd.ExcelWriter(Template_Path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
     df_finalPredictions.to_excel(writer, sheet_name='rForest_predictive_modeling', startcol=0, startrow=0, index=False, header=True)