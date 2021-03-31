import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
#from numba import njiti
from sportsipy.ncaab.teams import Teams
def change_conferences(team_stats):
    dataframe_temp = np.zeros([len(team_stats.index)])
    for ind in range(len(team_stats.index)):
        if team_stats['CONF'][ind] == 'B10':
            dataframe_temp[ind] = 1
        elif team_stats['CONF'][ind] == 'B12':
            dataframe_temp[ind] = 1
        elif team_stats['CONF'][ind] == 'ACC':
            dataframe_temp[ind] = 1
        elif team_stats['CONF'][ind] == 'BE':
            dataframe_temp[ind] = 1
        elif team_stats['CONF'][ind] == 'P12':
            dataframe_temp[ind] = 1
        elif team_stats['CONF'][ind] == 'SEC':
            dataframe_temp[ind] = 1
        else:
            dataframe_temp[ind] = 0;
    index = team_stats.index
    a_list = list(index)
    team_stats['Conference_Bool']=pd.Series(data=dataframe_temp,index=a_list)
    team_stats.drop(['CONF'],axis=1,inplace=True)
    return team_stats

def train_the_data(team_stats):
    X = team_stats[['ADJOE','ADJDE','EFG_O','EFG_D',
                  'TOR','TORD','ORB','DRB','Conference_Bool',
                    'FTR','FTRD','2P_O','2P_D','3P_O','3P_D']]
    y = team_stats['W']

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
    model3 = tree.DecisionTreeClassifier()
    model3.fit(X_train,y_train)
    return model, model2, model3

def compare_two_teams(team1,team2,team_stats,model):
    findteam1 = team_stats['TEAM']== team1
    team_vector1 = team_stats[['ADJOE','ADJDE','EFG_O','EFG_D',
                            'TOR','TORD','ORB','DRB','Conference_Bool',
                            'FTR','FTRD','2P_O','2P_D','3P_O','3P_D']][findteam1]
    findteam2 = team_stats['TEAM']== team2
    team_vector2 = team_stats[['ADJOE','ADJDE','EFG_O','EFG_D',
                            'TOR','TORD','ORB','DRB','Conference_Bool',
                            'FTR','FTRD','2P_O','2P_D','3P_O','3P_D']][findteam2]
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
