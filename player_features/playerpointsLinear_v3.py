import pandas as pd
from google.cloud import bigquery
import google.auth
import numpy as np
import openpyxl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

project = 'hockey-451914'

credentials, project = google.auth.default()

client = bigquery.Client()
"""
Summary: linear regression predctive model for declared value.

Version control:

v1. linear regression model to predict points for players based on their stats from the previous 4 seasons.
v2. Added autoregressive predictions for the next 3 seasons based on the previous 4 seasons.
    -- modified from using average player stats to treating each season as a separate feature by flattening dataframe
v3. Hyperparameter Tuning
    -- using Lasso regression with GridSearchCV to find the best alpha parameter
    -- then used elastic net with evaluation critera to determine most appropriate l1 ratio 
        -- l1 came in at 1.0 which is 100% lasso which is what I used originally, but I kept it in case I want to try ridge regression later

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

# converts to dataframe
df_allstats = pd.DataFrame(df_allstats)

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

# declare variable for anaylsis - allows for user to change target variable for other predictions
statVar = 'pointsPerGame'

# feature engineering
    # converting timeOnIcePerGame to minutes rather than seconds
df_allstats['timeOnIcePerGame'] = df_allstats['timeOnIcePerGame'] / 60  # convert seconds to minutes

    # creating function to convert raw values to average per game for normalization
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
    # this will allow the use of training data as independent varables for the model rather than averages
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

# ***v3 elastic net tuning***
    # evaluation for finding the best parameters for elastic net
        # alpha tests benchmarks for regulaization
        # l1_ratio will see what proprotion of lasso turning vs ridge tuning is apporpriate for this model
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]
}

    # set iterations for regression
elastic = ElasticNet(max_iter=10000)
grid = GridSearchCV(elastic, param_grid, cv=5)
    # trains for best parameters
grid.fit(X, y)

# fit tuning model
    # model will use gird output as model tuning inputs 
model = ElasticNet(alpha=grid.best_params_['alpha'], l1_ratio=grid.best_score_) 
model.fit(X, y)
    # current season prediction for later validation
df_player['predicted_20242025'] = model.predict(X)

# autoregressive to predict next three seasons 
future_X = df_player[seasons[1:]].copy() 
predictions = pd.DataFrame(index=df_player.index)

for season in future_seasons:
    future_X.columns = X.columns

    # predict the  next season
    pred = model.predict(future_X)
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
    'predicted_pointsPerGame_202728',
    ]]

print(df_finalPredictions)

# 7. Evaluate model performance
mse = mean_squared_error(y, df_player['predicted_20242025'])
r2 = r2_score(y, df_player['predicted_20242025'])

print("Mean Squared Error:", mse)
print("R² Score:", r2)
print("Best alpha:", grid.best_params_['alpha'])
print("Best l1_ratio:", grid.best_params_['l1_ratio'])
print("Best CV score:", grid.best_score_)

# for output, references your above filepath 
Template_Path = (f'{your_file_path}predictive_modeling_data.xlsx')

# # Define context manager using ExcelWriter function and store it as the variable "writer"
with pd.ExcelWriter(Template_Path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
     df_finalPredictions.to_excel(writer, sheet_name='linear_predictive_modeling', startcol=0, startrow=0, index=False, header=True)