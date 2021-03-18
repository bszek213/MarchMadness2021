import pandas as pd
import numpy as np
from sportsreference.ncaab.teams import Teams
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys

# Count the arguments
arguments = len(sys.argv) - 1
# Output argument-wise
position = 1
while (arguments >= position):
    print ("Team %i: %s" % (position, sys.argv[position]))
    position = position + 1
team_1 = sys.argv[1]
team_2 = sys.argv[2]

# set frames for input
dataset = pd.DataFrame()
teams = Teams()
team1 = teams(team_1)
team2 = teams(team_2)

#Get data to train
#wins
team1_wins = team1.wins
team2_wins = team2.wins
#PPG
team1_PPG = team1.points/team1.games_played
team2_PPG = team2.points/team2.games_played
#PPG - Opposing
team1_OPP_PPG = team1.opp_points/team1.games_played
team2_OPP_PPG = team2.opp_points/team2.games_played
#Threes a game
team1_3 = team1.three_point_field_goals/team1.games_played
team2_3 = team2.three_point_field_goals/team2.games_played
#Threes a game
team1_turn = team1.turnovers/team1.games_played
team2_turn = team2.turnovers/team2.games_played
#Assists
team1_assists = team1.assists/team1.games_played
team2_assists = team2.assists/team2.games_played
#Rebounds
team1_rebound = (team1.offensive_rebounds+team1.defensive_rebounds)/team1.games_played
team2_rebound = (team2.offensive_rebounds+team2.defensive_rebounds)/team2.games_played
#Steals
team1_steals = team1.steals/team1.games_played
team2_steals = team2.steals/team2.games_played
#Conference
if team1.conference == 'sec' or team1.conference == 'acc' or team1.conference == 'big-12' or team1.conference == 'big-10' or team1.conference == 'pac-12' or team1.conference == 'big-east':
    team1_conf = 1
else:
    team1_conf = 0
if team2.conference == 'sec' or team2.conference == 'acc' or team2.conference == 'big-12' or team2.conference == 'big-10' or team2.conference == 'pac-12' or team2.conference == 'big-east':
    team2_conf = 1
else:
    team2_conf = 0
#simple rating system
team1_rs = team1.simple_rating_system
team2_rs = team2.simple_rating_system
#Strength of schedule
team1_Str_Sched = team1.strength_of_schedule
team2_Str_Sched = team2.strength_of_schedule
#make dataframe
team1_class = np.zeros(1)
team2_class = np.ones(1)
team1_DF = {'Wins': team1_wins,'PPG':team1_PPG,'PPG_OPP':team1_OPP_PPG,
            '3AGAME':team1_3,'Turn':team1_turn,'CONF':team1_conf,
            'Rating':team1_rs,'Team Sched':team1_Str_Sched, 'Class': team1_class}
team2_DF = {'Wins': team2_wins,'PPG':team2_PPG,'PPG_OPP':team2_OPP_PPG,
            '3AGAME':team2_3,'Turn':team2_turn,'CONF':team2_conf,
            'Rating':team2_rs,'Team Sched':team2_Str_Sched,'Class': team2_class}

df_Team1 = pd.DataFrame(data=team1_DF)
df_Team2 = pd.DataFrame(data=team2_DF)
df_final = df_Team1.append([df_Team2])
#train data
X = df_final[['Wins','PPG','PPG_OPP','3AGAME','Turn','CONF','Rating','Team Sched']]
y = df_final['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y)
#random forest regressor
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}
model = RandomForestRegressor(**parameters)
model.fit(X_train, y_train)
if model.predict(X_test) == 0:
    print(team1.abbreviation + " wins")
else:
     print(team2.abbreviation + " wins")    
