# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:14:42 2022

@author: Bharath
"""
import time
from sportsipy.ncaab.teams import Teams
from html_parse import html_to_df_web_scrape
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split#,RandomizedSearchCV
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from sklearn.metrics import classification_report
import seaborn as sns

class marchMad:
    def __init__(self):
        team_data = pd.Series(dtype = float)
    
    def input_arg(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-t1", "--team1", help = "team 1 input")
        parser.add_argument("-t2", "--team2", help = "team 2 input")
        parser.add_argument("-g", "--games", help = "number of games for test data")
        self.args = parser.parse_args()
        
    def get_teams(self, year):
        all_teams = Teams(year)
        team_names = all_teams.dataframes.abbreviation
        print(team_names)
        final_list = []
        self.year_store = year
        for abv in team_names:
            str_combine = 'https://www.sports-reference.com/cbb/schools/' + abv.lower() + '/' + str(self.year_store) + '-gamelogs.html'
            print(f'current team: {abv}')
            df_inst = html_to_df_web_scrape(str_combine)
            final_list.append(df_inst)
        output = pd.concat(final_list)
        output['game_result'].loc[output['game_result'].str.contains('W')] = 'W'
        output['game_result'].loc[output['game_result'].str.contains('L')] = 'L'
        output['game_result'] = output['game_result'].replace({'W': 1, 'L': 0})
        
        final_data = output.replace(r'^\s*$', np.NaN, regex=True) #replace empty string with NAN
        self.all_data = final_data.dropna()
        print('len data: ', len(self.all_data))
        self.all_data.to_csv('all_data.csv')
        
    def split(self):
        #self.drop_cols = ['game_result']#, 'fta', 'ft_pct', 'fga', 'opp_orb', 'orb', 'opp_fta', 'blk', 'opp_fg_pct']
        self.y = self.all_data['game_result']
        self.x = self.all_data.drop(columns=['game_result'])
        self.correlate_analysis()
        # x_data = self.all_data.drop(columns=self.drop_cols)
        # scaler = MinMaxScaler()
        # scaled_data = scaler.fit_transform(x_data)
        # cols = x_data.columns
        # x = pd.DataFrame(scaled_data, columns = cols)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_no_corr,self.y, train_size=0.8)

    def machine(self):
        Gradclass = GradientBoostingClassifier()
        Gradclass.fit(self.x_train,self.y_train)
        
        RandForclass = RandomForestClassifier()
        RandForclass.fit(self.x_train,self.y_train)
        
        DecTreeclass = DecisionTreeClassifier()
        DecTreeclass.fit(self.x_train,self.y_train)
        
        SVCclass = SVC()
        SVCclass.fit(self.x_train,self.y_train)
        
        LogReg = LogisticRegression(max_iter=500)
        LogReg.fit(self.x_train,self.y_train)
        
        MLPClass = MLPClassifier()
        MLPClass.fit(self.x_train,self.y_train)
        
        KClass = KNeighborsClassifier()
        KClass.fit(self.x_train,self.y_train)
        
        Gradclass_err = accuracy_score(self.y_test, Gradclass.predict(self.x_test))
        RandForclass_err = accuracy_score(self.y_test, RandForclass.predict(self.x_test))
        DecTreeclass_err = accuracy_score(self.y_test, DecTreeclass.predict(self.x_test))
        SVCclass_err = accuracy_score(self.y_test, SVCclass.predict(self.x_test))
        LogReg_err = accuracy_score(self.y_test, LogReg.predict(self.x_test))
        MLPClass_err = accuracy_score(self.y_test, MLPClass.predict(self.x_test))
        KClass_err = accuracy_score(self.y_test, KClass.predict(self.x_test))
    
        print('Gradclass',Gradclass_err)
        print('RandForclass',RandForclass_err)
        print('DecTreeclass',DecTreeclass_err)
        print('SVCclass',SVCclass_err)
        print('LogReg',LogReg_err)
        print('MLPClass',MLPClass_err)
        print('KClass',KClass_err)
        err_list = [Gradclass_err, RandForclass_err, DecTreeclass_err, SVCclass_err, LogReg_err,
                    MLPClass_err, KClass_err]
        highest_acc = err_list.index(max(err_list))
        print(highest_acc)
        if highest_acc == 0:
            self.model_save = Gradclass
            self.name = 'Gradclass'
        if highest_acc == 1:
            self.model_save = RandForclass
            self.name = 'RandForclass'
        if highest_acc == 2:
            self.model_save = DecTreeclass
            self.name = 'DecTreeclass'
        if highest_acc == 3:
            self.model_save = SVCclass
            self.name = 'SVCclass'
        if highest_acc == 4:
            self.model_save = LogReg
            self.name = 'LogReg'
        if highest_acc == 5:
            self.model_save = MLPClass
            self.name = 'MLPClass'
        if highest_acc == 6:
            self.model_save = KClass
            self.name = 'KClass'
    def compare_two_teams(self):
        while True:
            try:
                team_1 = input('team 1 input: ')
                if team_1 == "exit":
                    break
                team_2 = input('team 2 input: ')
                if team_2 == "exit":
                    break
                # game_input = input('number of games to look over (all or int input): ')
                team1_str = 'https://www.sports-reference.com/cbb/schools/' + team_1 +  '/' + str(self.year_store) + '-gamelogs.html' #self.args.team1.lower()
                team2_str = 'https://www.sports-reference.com/cbb/schools/' + team_2 +  '/' + str(self.year_store) + '-gamelogs.html' #self.args.team2.lower()
                df_team1 = html_to_df_web_scrape(team1_str)
                df_team2 = html_to_df_web_scrape(team2_str)
                
                df_team1['game_result'].loc[df_team1['game_result'].str.contains('W')] = 'W'
                df_team1['game_result'].loc[df_team1['game_result'].str.contains('L')] = 'L'
                df_team1['game_result'] = df_team1['game_result'].replace({'W': 1, 'L': 0})
                df_team1_final = df_team1.replace(r'^\s*$', np.NaN, regex=True)
                df_team2['game_result'].loc[df_team2['game_result'].str.contains('W')] = 'W'
                df_team2['game_result'].loc[df_team2['game_result'].str.contains('L')] = 'L'
                df_team2['game_result'] = df_team2['game_result'].replace({'W': 1, 'L': 0})
                df_team2_final = df_team2.replace(r'^\s*$', np.NaN, regex=True)
                
                # scaler = MinMaxScaler()
                # scaled_data = scaler.fit_transform(df_team1_final)
                # cols = df_team1_final.columns
                # df_team1_final_scale = pd.DataFrame(scaled_data, columns = cols)
                
                # scaler2 = MinMaxScaler()
                # scaled_data2 = scaler2.fit_transform(df_team2_final)
                # cols = df_team2_final.columns
                # df_team2_final_scale = pd.DataFrame(scaled_data2, columns = cols)
                
                # df_team1_update = df_team1_final_scale.drop(columns=self.drop_cols).iloc[-10:].median(axis = 0, skipna = True).to_frame().T
                # df_team2_update = df_team2_final_scale.drop(columns=self.drop_cols).iloc[-10:].median(axis = 0, skipna = True).to_frame().T
                ####################10 games seems to be the best###########
                # if game_input == 'all':
                self.drop_cols.append('game_result')
                #7 game CONDITITION
                df_team1_update_7 = df_team1_final.drop(columns=self.drop_cols).iloc[-7:].median(axis = 0, skipna = True).to_frame().T
                df_team2_update_7 = df_team2_final.drop(columns=self.drop_cols).iloc[-7:].median(axis = 0, skipna = True).to_frame().T
                df_final_7 = df_team1_update_7.append(df_team2_update_7)
                # else:
                    # game_num = int(game_input) #int(self.args.games)
                #HEAT INDEX - 3 games
                df_team1_update_3 = df_team1_final.drop(columns=self.drop_cols).iloc[-3:].median(axis = 0, skipna = True).to_frame().T
                df_team2_update_3 = df_team2_final.drop(columns=self.drop_cols).iloc[-3:].median(axis = 0, skipna = True).to_frame().T
                df_final_3 = df_team1_update_3.append(df_team2_update_3)
                
                #10 games
                df_team1_update_10 = df_team1_final.drop(columns=self.drop_cols).iloc[-10:].median(axis = 0, skipna = True).to_frame().T
                df_team2_update_10 = df_team2_final.drop(columns=self.drop_cols).iloc[-10:].median(axis = 0, skipna = True).to_frame().T
                df_final_10 = df_team1_update_10.append(df_team2_update_10)
                
                #15 games
                df_team1_update_15 = df_team1_final.drop(columns=self.drop_cols).iloc[-15:].median(axis = 0, skipna = True).to_frame().T
                df_team2_update_15 = df_team2_final.drop(columns=self.drop_cols).iloc[-15:].median(axis = 0, skipna = True).to_frame().T
                df_final_15 = df_team1_update_15.append(df_team2_update_15)
                
                # proba_team_all = self.model_save.predict_proba(df_final_all)
                # proba_team_3 = self.model_save.predict_proba(df_final_3)
                # proba_team_10 = self.model_save.predict_proba(df_final_10)
                # proba_team_15 = self.model_save.predict_proba(df_final_15)
                # print('========================================================================================================')
                # print(f'probability of {team_1} winning is {float(proba_team[0][1])}, losing is {float(proba_team[0][0])}')
                # print(f'probability of {team_2} winning is {float(proba_team[1][1])}, losing is {float(proba_team[1][0])}')
                # print('========================================================================================================')
                # 7 game CONDITION #
                team1_np =df_team1_update_7.to_numpy()
                team2_np =df_team2_update_7.to_numpy()
        
                diff = [a - b for a, b in zip(team1_np, team2_np)]
                arr = np.array(diff)
                nx, ny = arr.shape
                final_vector = arr.reshape((1,nx*ny))
                
                diff = [b - a for a, b in zip(team1_np, team2_np)]
                arr = np.array(diff)
                nx, ny = arr.shape
                final_vector2 = arr.reshape((1,nx*ny))
                
                cols = df_team2_update_7.columns
                final_vect_df1 = pd.DataFrame(final_vector, columns = cols)
                final_vect_df2 = pd.DataFrame(final_vector2, columns = cols)
                proba_team1_7 = self.model_save.predict_proba(final_vect_df1)
                proba_team2_7 = self.model_save.predict_proba(final_vect_df2)
                
                # 3 CONDITION #
                team1_np =df_team1_update_3.to_numpy()
                team2_np =df_team2_update_3.to_numpy()
        
                diff = [a - b for a, b in zip(team1_np, team2_np)]
                arr = np.array(diff)
                nx, ny = arr.shape
                final_vector = arr.reshape((1,nx*ny))
                
                diff = [b - a for a, b in zip(team1_np, team2_np)]
                arr = np.array(diff)
                nx, ny = arr.shape
                final_vector2 = arr.reshape((1,nx*ny))
                
                cols = df_team1_update_3.columns
                final_vect_df1 = pd.DataFrame(final_vector, columns = cols)
                final_vect_df2 = pd.DataFrame(final_vector2, columns = cols)
                proba_team1_3 = self.model_save.predict_proba(final_vect_df1)
                proba_team2_3 = self.model_save.predict_proba(final_vect_df2)
                
                # 10 CONDITION #
                team1_np =df_team1_update_10.to_numpy()
                team2_np =df_team2_update_10.to_numpy()
        
                diff = [a - b for a, b in zip(team1_np, team2_np)]
                arr = np.array(diff)
                nx, ny = arr.shape
                final_vector = arr.reshape((1,nx*ny))
                
                diff = [b - a for a, b in zip(team1_np, team2_np)]
                arr = np.array(diff)
                nx, ny = arr.shape
                final_vector2 = arr.reshape((1,nx*ny))
                
                cols = df_team1_update_10.columns
                final_vect_df1 = pd.DataFrame(final_vector, columns = cols)
                final_vect_df2 = pd.DataFrame(final_vector2, columns = cols)
                proba_team1_10 = self.model_save.predict_proba(final_vect_df1)
                proba_team2_10 = self.model_save.predict_proba(final_vect_df2)
                
                # 15 CONDITION #
                team1_np =df_team1_update_15.to_numpy()
                team2_np =df_team2_update_15.to_numpy()
        
                diff = [a - b for a, b in zip(team1_np, team2_np)]
                arr = np.array(diff)
                nx, ny = arr.shape
                final_vector = arr.reshape((1,nx*ny))
                
                diff = [b - a for a, b in zip(team1_np, team2_np)]
                arr = np.array(diff)
                nx, ny = arr.shape
                final_vector2 = arr.reshape((1,nx*ny))
                
                cols = df_team1_update_15.columns
                final_vect_df1 = pd.DataFrame(final_vector, columns = cols)
                final_vect_df2 = pd.DataFrame(final_vector2, columns = cols)
                proba_team1_15 = self.model_save.predict_proba(final_vect_df1)
                proba_team2_15 = self.model_save.predict_proba(final_vect_df2)
                print('===================================================================================================================================================================')
                print(f'Probability that {team_1} wins over 3 games is {proba_team1_3[0][1]}, 7 games is {proba_team1_7[0][1]}, 10 is {proba_team1_10[0][1]}, 15 is {proba_team1_15[0][1]}')
                print(f'Probability that {team_2} wins over 3 games is {proba_team2_3[0][1]},  7 games is {proba_team2_7[0][1]}, 10 is {proba_team2_10[0][1]}, 15 is {proba_team2_15[0][1]}')
                print('===================================================================================================================================================================')
            except Exception as e:
                print(f'incorrect spelling of team names, reenter team neames: {e}')

    def plot_feature_importances(self):
        if self.name == 'LogReg':
            feature_imp = pd.Series(np.abs(self.model_save.coef_[0]),index=self.x_test.columns).sort_values(ascending=False) #feature_importances_
            plot1 = plt.figure(1)
            sns.barplot(x=feature_imp,y=feature_imp.index)
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.title('NCAA MENS BASKETBALL')
            plt.savefig('feature_importances.png')
            plt.close()
        # if self.name == 'MLPClass':
        #     feature_imp = pd.Series(np.abs(self.model_save.coefs_),index=self.x_test.columns).sort_values(ascending=False) #feature_importances_
    def correlate_analysis(self):
        corr_matrix = np.abs(self.x.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.90
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.80)]
        print('drop these:', to_drop)
        self.drop_cols = to_drop
        self.x_no_corr = self.x.drop(columns=to_drop)
        top_corr_features = corr_matrix.index
        plt.figure(figsize=(20,20))
        #plot heat map
        g=sns.heatmap(corr_matrix[top_corr_features],annot=True,cmap="RdYlGn")
        plt.savefig('correlations.png')
        plt.close()
        

if __name__ == '__main__':
    start_time = time.time()
    mad = marchMad()
    mad.input_arg()
    mad.get_teams(2022)
    mad.split()
    mad.machine()
    mad.compare_two_teams()
    mad.plot_feature_importances()
    print("--- %s seconds ---" % (time.time() - start_time))