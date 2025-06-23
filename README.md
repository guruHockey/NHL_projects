# NHL_projects
public facing NHL projects

This repository contains predictive modeling that pertains to the NHL.

Files include:

1. Player performance data modeling: predicting player performance with at least 5 years NHL experience as of the 2024-2025 season.

Files that pertain to player features:
  - playerpointsLinear_3.py is a linear regression model that predicts player performance of a chosen variable and uses ElasticNet methodology
  - playerpoints_RF.py is a ensemble tree-based model using RandomForest modeling to predict player performance of a chosen variable
  - NHL_API_calls.py is a file that calls the data from the NHL's public facing api
  - predictive_modeling_data.xlsx contains the data that the models are run on as well as the output of each predictive models.

2. Team based machine learning: analyzing team performance from the 2024-2025 season.

Files that pertain to team features:
  - possessionCluster.py clusters teams by team-level possession performance
      - poss_clustering_data.xlsx contains the data that the models are run on.
  - home_roadClustering.py clusters teams by team-level scoring performance
    - clustering_scoring_data.xlsx contains the data that the models are run on.
  

