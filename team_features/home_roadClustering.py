import matplotlib
matplotlib.use('TkAgg')  # Change the backend to TkAgg or another interactive backend
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from google.cloud import bigquery
import google.auth

# project = 'hockey-451914'

# credentials, project = google.auth.default()

# client = bigquery.Client()

# query1 = """
# (
# SELECT
#     team,
#     `home_win%`,
#     homeGoalDiff,
#     home_avgGoalsAgainst, 
#     home_avgGoalsFor,
#     `road_win%`,
#     roadGoalDiff,
#     road_avgGoalsAgainst, 
#     road_avgGoalsFor,

# FROM `hockey-451914.guruHockey11782.enhanced_anaytics` 
# )
# """

# df = client.query(query1).to_dataframe()
# # print(df)

"""
query1 is the query I used to create this dataset based on a datamart that I created from
NHL public facing API data.  This data can be found in the .xlsx named predictive_modeling_data.xlsx
"""
# # place your file path here:
your_file_path = '/home/michael/Desktop/'

# # reads xlsx file from local directory
df = pd.read_excel(f'{your_file_path}clustering_scoring_data.xlsx')

# converts to dataframe
df = pd.DataFrame(df)

# preprocessing data,
features = df[['home_win%', 'homeGoalDiff', 'home_avgGoalsAgainst', 'home_avgGoalsFor', 'road_win%', 'roadGoalDiff','road_avgGoalsAgainst','road_avgGoalsFor']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# optimal number of clusters using the elbow method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# evaluate clustering quality with sil score
labels = kmeans.labels_
sil_score = silhouette_score(scaled_features, labels)
print("silhouette score:", sil_score)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid()
plt.show()

# # elbow = 2 clusters
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# this portion allows you to choose which features to map to the graph
inputx = 'home_avgGoalsFor'
inputy = 'road_avgGoalsFor'

""""
  possible inputs for this model:
    `home_win%`,
    homeGoalDiff,
    home_avgGoalsAgainst, 
    home_avgGoalsFor,
    `road_win%`,
    roadGoalDiff,
    road_avgGoalsAgainst, 
    road_avgGoalsFor,
"""

# map clusters for graphing
plt.figure(figsize=(10, 6))
plt.scatter(df[inputx], df[inputy], c=df['Cluster'], cmap='viridis', marker='o', s=100)
plt.title('K-Means Clustering of Hockey Teams')
plt.xlabel(inputx)
plt.ylabel(inputy)
plt.colorbar(label='Cluster')
for i, player in enumerate(df['team']):
    plt.annotate(player, (df[inputx][i], df[inputy][i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.grid()
plt.show()


# # for output, references your above filepath 
# Template_Path = (f'{your_file_path}clustering_scoring_data.xlsx')

# # # # Define context manager using ExcelWriter function and store it as the variable "writer"
# with pd.ExcelWriter(Template_Path, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
#      df.to_excel(writer, sheet_name='clustering_scoring_data', startcol=0, startrow=0, index=False, header=True)
