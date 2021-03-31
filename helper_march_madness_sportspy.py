import pandas as pd
import numpy as np
from sportsipy.ncaab.teams import Teams
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def get_all_teams():
    #teams = Teams()
    counter = 0
    save_teams = {}
    team_names = pd.read_csv('/home/bszekely/Desktop/ProjectsResearch/March_Madness/teamnames.txt',header=None)
    for teamind in team_names[0]:
        substring = teamind[teamind.find("(")+1:teamind.find(")")]
        save_teams[counter] = substring
        counter = counter +1
    return save_teams

def get_input_args():
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
    return team_1, team_2

def create_df_all_teams(save_teams):
   all_teams = Teams()
   save_all_team_data = {}
   counter = 0
   for ind in save_teams:
       ind_team = save_teams[ind]
       get_all_teams = {}
       team = all_teams(ind_team)
       if team.games_played > 0:
           get_all_teams['TEAM'] = ind_team
           get_all_teams['wins'] = team.wins
           get_all_teams['PPG'] = team.points/team.games_played
           get_all_teams['OPG'] = team.opp_points/team.games_played
           get_all_teams['3Game'] = team.three_point_field_goals/team.games_played
           get_all_teams['Turnovers'] = team.turnovers/team.games_played
           get_all_teams['Off Rating'] = team.offensive_rating
           get_all_teams['SOS'] = team.strength_of_schedule
           get_all_teams['SRS'] = team.simple_rating_system
           get_all_teams['assistgame'] = team.assists/team.games_played
           get_all_teams['blocks'] = team.blocks/team.games_played
           get_all_teams['field_goals'] = team.field_goals/team.games_played
           get_all_teams['free throws'] = team.free_throws/team.games_played
           get_all_teams['pace'] = team.pace
           get_all_teams['opp rebound'] = team.opp_total_rebounds/team.games_played
           get_all_teams['total rebounds'] = team.total_rebounds/team.games_played
           save_all_team_data[counter] = get_all_teams
           counter = counter +1
   data_transformed_df = pd.DataFrame.from_dict(save_all_team_data).T
   return data_transformed_df

def train_the_data(team_stats):
    X = team_stats[['PPG','OPG','3Game','Turnovers',
                  'Off Rating','SOS','SRS','assistgame','blocks',
                    'field_goals','free throws','pace','opp rebound','total rebounds']]
    y = team_stats['wins']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, X, y

def ml_analysis(X_train,y_train):
    #This model worked best for me but you can add in more here if you want
    model = GradientBoostingRegressor(n_estimators=100,max_depth=5)
    model.fit(X_train,y_train)
    model2 = RandomForestRegressor()
    model2.fit(X_train,y_train)
    return model, model2

def compare_two_teams(team1,team2,team_stats,model):
    findteam1 = team_stats['TEAM']== team1
    team_vector1 = team_stats[['PPG','OPG','3Game','Turnovers',
                                'Off Rating','SOS','SRS','assistgame','blocks',
                                'field_goals','free throws','pace','opp rebound','total rebounds']][findteam1]
    findteam2 = team_stats['TEAM']== team2
    team_vector2 = team_stats[['PPG','OPG','3Game','Turnovers',
                                'Off Rating','SOS','SRS','assistgame','blocks',
                                'field_goals','free throws','pace','opp rebound','total rebounds']][findteam2]
    team1_np =team_vector1.to_numpy()
    team2_np =team_vector2.to_numpy()

    diff = [a - b for a, b in zip(team1_np, team2_np)]
    arr = np.array(diff)
    nx, ny = arr.shape
    final_vector = arr.reshape((1,nx*ny))

    diff = [b - a for a, b in zip(team1_np, team2_np)]
    arr = np.array(diff)
    nx, ny = arr.shape
    final_vector2 = arr.reshape((1,nx*ny))

    print('Probability that ' + team1 + ' wins:', model.predict(final_vector))
    print('Probability that ' + team2 + ' wins:', model.predict(final_vector2))

def plot_feature_importances(model,X):
    feature_imp = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
    plot1 = plt.figure(1)
    sns.barplot(x=feature_imp,y=feature_imp.index)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('NCAA MENS BASKETBALL')
    plt.show()
