import requests
import pandas as pd
from pandas import json_normalize
import xlsxwriter
from datetime import datetime, timedelta
import numpy as np
from google.cloud import bigquery
import google.auth

project = 'hockey-451914'

credentials, project = google.auth.default()

client = bigquery.Client()

"""
    API call to get skater bio data
"""

enhancedURL= "https://api.nhle.com/stats/rest/en/skater/bios?isAggregate=true&isGame=false&sort=[{%22property%22:%22playerId%22,%22direction%22:%22ASC%22}]&limit=0&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20242025%20and%20seasonId%3E=20202021&start="

# JSON format
response = requests.get(enhancedURL, params={"Content-Type": "application/json"})
data = response.json()

# normalize the JSON data, convert to dataframe
df = json_normalize(data)

# test df
# print(df)

expanded_entries = []
# Iterate through the data
for index, skaters in enumerate(df['data']):
    for entry in skaters:
        # firstName = entry.get('firstName', {}).get('default', None)
        # lastName = entry.get('lastName', {}).get('default', None)
        expanded_entries.append({
            'skaterFullName': entry['skaterFullName'],
            'lastName': entry['lastName'],
            'playerId': entry['playerId'],
            'shootsCatches': entry['shootsCatches'],     
            'birthCountryCode': entry['birthCountryCode'],
            'birthCity': entry['birthCity'],
            'birthStateProvinceCode': entry['birthStateProvinceCode'],            
            'birthDate': entry['birthDate'],
            'positionCode': entry['positionCode']
            })

skaterBio2425 = pd.DataFrame(expanded_entries)

# # covert bday to date time
skaterBio2425['birthDate'] = pd.to_datetime(skaterBio2425['birthDate'])

# # set date for calculation as O10/1 of season start year to get rough age per player during the most recent season
startDate = pd.to_datetime('2024-10-01')

# function to calculate season age
def calculate_season_age(row):
    return startDate.year - row['birthDate'].year - ((startDate.month, startDate.day) < (row['birthDate'].month, row['birthDate'].day))

# add season age calculations
skaterBio2425['seasonAge2425'] = skaterBio2425.apply(calculate_season_age, axis=1)

print(skaterBio2425)

table_id = 'hockey-451914.guruHockey11782.2020_2025skater_bios'

# append dataset 
def append_to_bigquery_table(dataf, table_id):
    # Assuming the table already exists, we directly load the new data
    job = client.load_table_from_dataframe(dataf, table_id, job_config=bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    ))
    job.result()  # Waits for the job to complete
    print(f"Appended {len(dataf)} rows to {table_id}")

# Call the function to append data
append_to_bigquery_table(skaterBio2425, table_id)

"""
    API call to get skater summary data,
    it was necessary to run for separate seasons or cumulative stats will be returned (will need to turn into a function rather than calling each time)
"""

enhancedURL= "https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=false&sort=[{%22property%22:%22playerId%22,%22direction%22:%22ASC%22}]&limit=0&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20242025%20and%20seasonId%3E=20242025&start="

# JSON format
response = requests.get(enhancedURL, params={"Content-Type": "application/json"})
data = response.json()

# Assuming 'data' is your original JSON data
df = json_normalize(data)

# test = pd.DataFrame(df)
# print(df)

expanded_entries = []
# Iterate through the data
for index, skaters in enumerate(df['data']):
    for entry in skaters:
        # firstName = entry.get('firstName', {}).get('default', None)
        # lastName = entry.get('lastName', {}).get('default', None)
        expanded_entries.append({
            'skaterFullName': entry['skaterFullName'],
            'lastName': entry['lastName'],
            'playerId': entry['playerId'],
            'shootsCatches': entry['shootsCatches'],     
            'goals': entry['goals'],
            'assists': entry['assists'],
            'points': entry['points'],
            'plusMinus': entry['plusMinus'],
            'evPoints': entry['evPoints'],
            'faceoffWinPct': entry['faceoffWinPct'],
            'gameWinningGoals': entry['gameWinningGoals'],
            'gamesPlayed': entry['gamesPlayed'],
            'otGoals': entry['otGoals'],
            'penaltyMinutes': entry['penaltyMinutes'],
            'pointsPerGame': entry['pointsPerGame'],
            'positionCode': entry['positionCode'],
            'ppGoals': entry['ppGoals'],
            'ppPoints': entry['ppPoints'],
            'shGoals': entry['shGoals'],
            'shots': entry['shots'],
            'timeOnIcePerGame': entry['timeOnIcePerGame'],
            })

skater_summary2425 = pd.DataFrame(expanded_entries)

skater_summary2425['season'] = '20242025'

# print(skater_summary2425)

enhancedURL= "https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=false&sort=[{%22property%22:%22playerId%22,%22direction%22:%22ASC%22}]&limit=0&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20232024%20and%20seasonId%3E=20232024&start="

# JSON format
response = requests.get(enhancedURL, params={"Content-Type": "application/json"})
data = response.json()

# Assuming 'data' is your original JSON data
df = json_normalize(data)

# test = pd.DataFrame(df)
# print(df)

expanded_entries = []
# Iterate through the data
for index, skaters in enumerate(df['data']):
    for entry in skaters:
        # firstName = entry.get('firstName', {}).get('default', None)
        # lastName = entry.get('lastName', {}).get('default', None)
        expanded_entries.append({
            'skaterFullName': entry['skaterFullName'],
            'lastName': entry['lastName'],
            'playerId': entry['playerId'],
            'shootsCatches': entry['shootsCatches'],     
            'goals': entry['goals'],
            'assists': entry['assists'],
            'points': entry['points'],
            'plusMinus': entry['plusMinus'],
            'evPoints': entry['evPoints'],
            'faceoffWinPct': entry['faceoffWinPct'],
            'gameWinningGoals': entry['gameWinningGoals'],
            'gamesPlayed': entry['gamesPlayed'],
            'otGoals': entry['otGoals'],
            'penaltyMinutes': entry['penaltyMinutes'],
            'pointsPerGame': entry['pointsPerGame'],
            'positionCode': entry['positionCode'],
            'ppGoals': entry['ppGoals'],
            'ppPoints': entry['ppPoints'],
            'shGoals': entry['shGoals'],
            'shots': entry['shots'],
            'timeOnIcePerGame': entry['timeOnIcePerGame'],
            })

skater_summary20232024 = pd.DataFrame(expanded_entries)

skater_summary20232024['season'] = '20232024'

# print(skater_summary20232024)

enhancedURL= "https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=false&sort=[{%22property%22:%22playerId%22,%22direction%22:%22ASC%22}]&limit=0&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20222023%20and%20seasonId%3E=20222023&start="

# JSON format
response = requests.get(enhancedURL, params={"Content-Type": "application/json"})
data = response.json()

# Assuming 'data' is your original JSON data
df = json_normalize(data)

# test = pd.DataFrame(df)
# print(df)

expanded_entries = []
# Iterate through the data
for index, skaters in enumerate(df['data']):
    for entry in skaters:
        # firstName = entry.get('firstName', {}).get('default', None)
        # lastName = entry.get('lastName', {}).get('default', None)
        expanded_entries.append({
            'skaterFullName': entry['skaterFullName'],
            'lastName': entry['lastName'],
            'playerId': entry['playerId'],
            'shootsCatches': entry['shootsCatches'],     
            'goals': entry['goals'],
            'assists': entry['assists'],
            'points': entry['points'],
            'plusMinus': entry['plusMinus'],
            'evPoints': entry['evPoints'],
            'faceoffWinPct': entry['faceoffWinPct'],
            'gameWinningGoals': entry['gameWinningGoals'],
            'gamesPlayed': entry['gamesPlayed'],
            'otGoals': entry['otGoals'],
            'penaltyMinutes': entry['penaltyMinutes'],
            'pointsPerGame': entry['pointsPerGame'],
            'positionCode': entry['positionCode'],
            'ppGoals': entry['ppGoals'],
            'ppPoints': entry['ppPoints'],
            'shGoals': entry['shGoals'],
            'shots': entry['shots'],
            'timeOnIcePerGame': entry['timeOnIcePerGame'],
            })

skater_summary20222023= pd.DataFrame(expanded_entries)

skater_summary20222023['season'] = '20222023'

# print(skater_summary20232024)

enhancedURL= "https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=false&sort=[{%22property%22:%22playerId%22,%22direction%22:%22ASC%22}]&limit=0&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20212022%20and%20seasonId%3E=20212022&start="

# JSON format
response = requests.get(enhancedURL, params={"Content-Type": "application/json"})
data = response.json()

# Assuming 'data' is your original JSON data
df = json_normalize(data)

# test = pd.DataFrame(df)
# print(df)

expanded_entries = []
# Iterate through the data
for index, skaters in enumerate(df['data']):
    for entry in skaters:
        # firstName = entry.get('firstName', {}).get('default', None)
        # lastName = entry.get('lastName', {}).get('default', None)
        expanded_entries.append({
            'skaterFullName': entry['skaterFullName'],
            'lastName': entry['lastName'],
            'playerId': entry['playerId'],
            'shootsCatches': entry['shootsCatches'],     
            'goals': entry['goals'],
            'assists': entry['assists'],
            'points': entry['points'],
            'plusMinus': entry['plusMinus'],
            'evPoints': entry['evPoints'],
            'faceoffWinPct': entry['faceoffWinPct'],
            'gameWinningGoals': entry['gameWinningGoals'],
            'gamesPlayed': entry['gamesPlayed'],
            'otGoals': entry['otGoals'],
            'penaltyMinutes': entry['penaltyMinutes'],
            'pointsPerGame': entry['pointsPerGame'],
            'positionCode': entry['positionCode'],
            'ppGoals': entry['ppGoals'],
            'ppPoints': entry['ppPoints'],
            'shGoals': entry['shGoals'],
            'shots': entry['shots'],
            'timeOnIcePerGame': entry['timeOnIcePerGame'],
            })

skater_summary20212022 = pd.DataFrame(expanded_entries)

skater_summary20212022['season'] = '20212022'

# print(skater_summary20212022)

enhancedURL= "https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=true&isGame=false&sort=[{%22property%22:%22playerId%22,%22direction%22:%22ASC%22}]&limit=0&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameTypeId=2%20and%20seasonId%3C=20202021%20and%20seasonId%3E=20202021&start="

# JSON format
response = requests.get(enhancedURL, params={"Content-Type": "application/json"})
data = response.json()

# Assuming 'data' is your original JSON data
df = json_normalize(data)

# test = pd.DataFrame(df)
# print(df)

expanded_entries = []
# Iterate through the data
for index, skaters in enumerate(df['data']):
    for entry in skaters:
        # firstName = entry.get('firstName', {}).get('default', None)
        # lastName = entry.get('lastName', {}).get('default', None)
        expanded_entries.append({
            'skaterFullName': entry['skaterFullName'],
            'lastName': entry['lastName'],
            'playerId': entry['playerId'],
            'shootsCatches': entry['shootsCatches'],     
            'goals': entry['goals'],
            'assists': entry['assists'],
            'points': entry['points'],
            'plusMinus': entry['plusMinus'],
            'evPoints': entry['evPoints'],
            'faceoffWinPct': entry['faceoffWinPct'],
            'gameWinningGoals': entry['gameWinningGoals'],
            'gamesPlayed': entry['gamesPlayed'],
            'otGoals': entry['otGoals'],
            'penaltyMinutes': entry['penaltyMinutes'],
            'pointsPerGame': entry['pointsPerGame'],
            'positionCode': entry['positionCode'],
            'ppGoals': entry['ppGoals'],
            'ppPoints': entry['ppPoints'],
            'shGoals': entry['shGoals'],
            'shots': entry['shots'],
            'timeOnIcePerGame': entry['timeOnIcePerGame'],
            })

skater_summary20202021 = pd.DataFrame(expanded_entries)

skater_summary20202021['season'] = '20202021'

# print(skater_summary20202021)

skater_summaryCOMB = pd.concat([
    skater_summary2425, 
    skater_summary20232024, 
    skater_summary20222023,
    skater_summary20212022,
    skater_summary20202021
    ], ignore_index=True)

skater_summaryCOMB = skater_summaryCOMB.sort_values(by='skaterFullName', ascending=True)

# print(skater_summaryCOMB)

# # table_id = 'hockey-451914.guruHockey11782.evensTEST'
table_id = 'hockey-451914.guruHockey11782.2020_2025skater_summary'

# append dataset with new evens info
def append_to_bigquery_table(dataf, table_id):
    # Assuming the table already exists, we directly load the new data
    job = client.load_table_from_dataframe(dataf, table_id, job_config=bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    ))
    job.result()  # Waits for the job to complete
    print(f"Appended {len(dataf)} rows to {table_id}")

# Call the function to append data
append_to_bigquery_table(skater_summaryCOMB, table_id)