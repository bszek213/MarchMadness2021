import pandas as pd
import numpy as np
import sys
from helper_march_madness import change_conferences,train_the_data,ml_analysis,compare_two_teams,plot_feature_importances
# Get input of teams from cin #
# Count the arguments
arguments = len(sys.argv) - 1
# Output argument-wise
position = 1
while (arguments >= position):
    print ("Team %i: %s" % (position, sys.argv[position]))
    position = position + 1
team_1 = sys.argv[1]
team_2 = sys.argv[2]

# Read in the team stats data from Kaggle #
team_stats = pd.read_csv('/home/bszekely/Desktop/ProjectsResearch/March_Madness/cbb21.csv')
team_stats.drop(['BARTHAG','ADJ_T','WAB','SEED'],axis=1,inplace=True)

# Change conferences #
team_stats_updated = change_conferences(team_stats)
# Train the data #
X_train, X_test, y_train, y_test, X, y = train_the_data(team_stats_updated)
# Run ML model on data #
print('Gradient Boosting Regressor Prediction: ')
model,model2,model3  = ml_analysis(X_train,y_train)
# compare two teams
compare_two_teams(team_1,team_2,team_stats_updated,model)
print(' ' )
print('Random Forest Regressor Prediction: ')
compare_two_teams(team_1,team_2,team_stats_updated,model2)
print(' ' )
print('Decision Tree Classifier Prediction: ')
compare_two_teams(team_1,team_2, team_stats_updated,model3)
# Plot the feature importances for Gradient Boosting Regressor #
plot_feature_importances(model,X)
# Plot the features importances for Random Forest Regressor #
plot_feature_importances(model2,X)
