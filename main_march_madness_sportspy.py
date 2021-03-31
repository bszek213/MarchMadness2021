import pandas as pd
import numpy as np
import helper_march_madness_sportspy
# Get input of teams from cin #
team1,team2 = helper_march_madness_sportspy.get_input_args()
# Get all teams in a format that sportspy understands #
save_teams = helper_march_madness_sportspy.get_all_teams()
#print(save_teams[1])
# get all features #
df_all_teams = helper_march_madness_sportspy.create_df_all_teams(save_teams)
# Training Data #
X_train, X_test, y_train, y_test, X, y = helper_march_madness_sportspy.train_the_data(df_all_teams)
# Run ML models #
model, model2= helper_march_madness_sportspy.ml_analysis(X_train,y_train)
# Predict the game outcomes #
print('Gradient Boosting Regressor Predcition: ')
helper_march_madness_sportspy.compare_two_teams(team1,team2,df_all_teams,model)
print(' ')
print('Random Forest Regressor Predcition: ')
helper_march_madness_sportspy.compare_two_teams(team1,team2,df_all_teams,model2)
# Graph feature importances #
helper_march_madness_sportspy.plot_feature_importances(model,X) #Gradient Boosted Regressor
helper_march_madness_sportspy.plot_feature_importances(model2,X) #Random Forest Regressor

