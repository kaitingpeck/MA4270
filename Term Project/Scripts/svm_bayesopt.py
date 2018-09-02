from __future__ import print_function
from __future__ import division

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# for importing ML module
import ml_programs as ml

from sklearn import preprocessing

import os
os.getcwd()

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC, LinearSVC
# KFold
from sklearn.model_selection import KFold    

# one-hot encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Bayesian Optimization
from bayes_opt import BayesianOptimization
from skopt import gp_minimize
from skopt import Optimizer
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

############################# Main Program ################################
# set project_dir
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
train_df = pd.read_csv('./Data/train.csv')
test_df = pd.read_csv('./Data/test.csv')

# Data cleaning

combined_data = train_df.append(test_df)

combined_data.Age.fillna(value=combined_data.Age.mean(), inplace=True)
combined_data.Fare.fillna(value=combined_data.Fare.mean(), inplace=True)
combined_data.Embarked.fillna(value=(combined_data.Embarked.value_counts().idxmax()), inplace=True)
combined_data.Survived.fillna(value=-1, inplace=True) 

# drop columns that are not needed
combined_data.drop('Name', axis=1, inplace=True)
combined_data.drop('Cabin', axis=1, inplace=True)
combined_data.drop('Ticket', axis=1, inplace=True)

# Write cleaned data out

train = combined_data[combined_data['Survived']!=-1]
# train.to_csv("./Data/train-clean.csv")

test = combined_data[combined_data['Survived']==-1]
test.drop('Survived', axis=1, inplace=True)
# test.to_csv("./Data/test-clean.csv")

# One-hot encoding
train_encoded = pd.get_dummies(train, columns = ['Embarked','Sex'])
test_encoded = pd.get_dummies(test, columns = ['Embarked','Sex'])

# Rearrange columns
list_of_features = ['Age','Embarked_C','Embarked_Q','Embarked_S','Fare','Parch','Pclass','Sex_female','Sex_male','SibSp']
list_of_columns = list_of_features + ['Survived']
train_encoded = train_encoded[list_of_columns]
test_encoded = test_encoded[list_of_features]

# Transform training data into np arrays
X_train = train_encoded[list_of_features].values
X_test = test_encoded[list_of_features].values
y_train = train_encoded['Survived'].values
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)

################################## ML Stuff ###################################
def svccv(C):
    '''
    takes in input features X and labels y
    runs C-SVM with given C on the data matrix
    returns the score (accuracy in our case) calculated from k-fold validation

    '''
    clf = SVC(C=C, kernel='rbf', gamma=0.0001, max_iter=21470000)
    score = cross_val_score(clf, X_train, y_train, cv=10).mean()
    return score

def posterior(bo, x, xmin=0.001, xmax=200):
    xmin, xmax = 0.001, 500
    bo.gp.fit(bo.X, bo.Y)
    mu, sigma = bo.gp.predict(x, return_std=True)
    return mu, sigma

def plot_gp(bo, x, y):
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Gaussian Process and Utility Function After {} Steps'.format(len(bo.X)), fontdict={'size':30})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    
    mu, sigma = posterior(bo, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(bo.X.flatten(), bo.Y, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((0.001, 500))
    axis.set_ylim((None, None))
    axis.set_ylabel('SVM accuracy', fontdict={'size':20})
    axis.set_xlabel('X', fontdict={'size':20})
    
    '''utility = bo.util.utility(x, bo.gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((0.001, 500))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})'''
    
    lgd1 = axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    # lgd2 = acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    fig.savefig('bo_svm.jpg', bbox_extra_artists=(lgd1,), bbox_inches='tight')

if __name__ == "__main__":
    gp_params = {"alpha": 1e-5}
    
    x = np.linspace(0.001, 500, 100).reshape(-1,1)
    y = np.asarray([svccv(x[i][0]) for i in range(len(x))]).reshape(-1,1)
    plt.figure(figsize=(8, 10))
    plt.plot(x, y)
    # plt.show()

    
    svcBO = BayesianOptimization( svccv,
        {'C': (0.001, 500)})#, 'gamma': (0.0001, 0.1)} )
    # svcBO.explore({'C': [0.001, 0.01, 0.1]}
    svcBO.maximize(n_iter=10, **gp_params, acq='ei')
    plot_gp(svcBO, x, y)

    print('-' * 53)
    print('Final Results')
    print('SVC: %f' % svcBO.res['max']['max_val'])
    print('Params: ' + str(svcBO.res['max']))

