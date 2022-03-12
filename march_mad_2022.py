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
from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split#,RandomizedSearchCV
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

class marchMad:
    def __init__(self):
        team_data = pd.Series(dtype = float)
    
    def input_arg(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-t1", "--team1", help = "team 1 input")
        parser.add_argument("-t2", "--team2", help = "team 2 input")
        self.args = parser.parse_args()
        
    def get_teams(self, year):
        all_teams = Teams(year)
        team_names = all_teams.dataframes.abbreviation
        final_list = []
        for abv in team_names:
            str_combine = 'https://www.sports-reference.com/cbb/schools/' + abv.lower() + '/2022-gamelogs.html'
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
        self.drop_cols = ['game_result', 'fta', 'ft_pct', 'fga', 'opp_orb', 'orb', 'opp_fta', 'blk', 'opp_fg_pct']
        y = self.all_data['game_result']
        x = self.all_data.drop(columns=self.drop_cols)
        # x_data = self.all_data.drop(columns=self.drop_cols)
        # scaler = MinMaxScaler()
        # scaled_data = scaler.fit_transform(x_data)
        # cols = x_data.columns
        # x = pd.DataFrame(scaled_data, columns = cols)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y, train_size=0.8)

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
        team1_str = 'https://www.sports-reference.com/cbb/schools/' + self.args.team1.lower() + '/2022-gamelogs.html'
        team2_str = 'https://www.sports-reference.com/cbb/schools/' + self.args.team2.lower() + '/2022-gamelogs.html'
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
        df_team1_update = df_team1_final.drop(columns=self.drop_cols).iloc[-10:].median(axis = 0, skipna = True).to_frame().T
        df_team2_update = df_team2_final.drop(columns=self.drop_cols).iloc[-10:].median(axis = 0, skipna = True).to_frame().T
        df_final = df_team1_update.append(df_team2_update)
        print(df_final)
        
        proba_team = self.model_save.predict_proba(df_final)
        predict_team1 = self.model_save.predict(df_team1_update)
        predict_team2 = self.model_save.predict(df_team2_update)
        print('========================================================================================================')
        print(f'probability of {self.args.team1} winning is {float(proba_team[0][1])}, losing is {float(proba_team[0][0])}')
        print(f'probability of {self.args.team2} winning is {float(proba_team[1][1])}, losing is {float(proba_team[1][0])}')
        print('========================================================================================================')
        print(f'prediction of {self.args.team1} winning is {predict_team1}')
        print(f'prediction of {self.args.team2} winning is {predict_team2}')
        print('========================================================================================================')
        
    def plot_feature_importances(self):
        if self.name == 'LogReg':
            feature_imp = pd.Series(np.abs(self.model_save.coef_[0]),index=self.x_test.columns).sort_values(ascending=False) #feature_importances_
            plot1 = plt.figure(1)
            sns.barplot(x=feature_imp,y=feature_imp.index)
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.title('NCAA MENS BASKETBALL')
            plt.savefig('feature_importances.png')
        # if self.name == 'MLPClass':
        #     feature_imp = pd.Series(np.abs(self.model_save.coefs_),index=self.x_test.columns).sort_values(ascending=False) #feature_importances_

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