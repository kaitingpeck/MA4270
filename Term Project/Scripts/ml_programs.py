# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# MLP
from sklearn.neural_network import MLPClassifier

# KFold
from sklearn.model_selection import KFold    

# one-hot encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def run_nb(X, y, num_folds=5):
    gnb = GaussianNB()
    scores = cross_val_score(gnb, X, y, cv=num_folds)
    return scores
    
def run_svm(X, y, num_folds=10):
    clf = svm.SVC(C=125.265, gamma=0.001, max_iter=214748300)
    scores = cross_val_score(clf, X, y, cv=num_folds)
    # clf.fit(X, y)
    return scores

def get_k_fold(features, num_splits=5):
    kf = KFold(n_splits)
    return kf.split(features)
    
def create_mlp():
   clf = MLPClassifier(solver='sgd', alpha=1e-5, random_state=1,
                       early_stopping = True, learning_rate = 'adaptive',
                       learning_rate_init = 1e-4)
   return clf

def run_k_fold_mlp(features, labels, num_folds=5):
    clf = create_mlp()
    scores = cross_val_score(clf, features, labels, cv=num_folds)        
    return scores

def grid_search_mlp(features, labels, num_folds=5):
    parameters = {'solver':('sgd','adam','lbfgs'),
                  'activation':('identity', 'logistic', 'tanh', 'relu'),
                  'alpha':[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                  'learning_rate_init':[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
                  'learning_rate':('constant', 'adaptive', 'invscaling')}
    mlp= MLPClassifier(early_stopping = True)
    clf = GridSearchCV(mlp, parameters, cv=num_folds)
    cv_results = clf.fit(features, labels).cv_results_

    # Collate results
    mean_test_score = cv_results['mean_test_score'].tolist()
    params = cv_results['params']
    best_test_score_idx = mean_test_score.index(max(mean_test_score))
    
    return mean_test_score[best_test_score_idx], params[best_test_score_idx]

def grid_search_svm(features, labels, num_folds=5):
    '''
    takes in input features X and labels y
    runs C-SVM with given C on the data matrix
    returns the score (accuracy in our case) calculated from k-fold validation

    '''
    parameters = {'C':[0.1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90],
                 'kernel':('linear','poly','rbf')}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters, cv=num_folds)
    cv_results = clf.fit(features, labels).cv_results_

    # Collate results
    mean_test_score = cv_results['mean_test_score'].tolist()
    params = cv_results['params']
    best_test_score_idx = mean_test_score.index(max(mean_test_score))
    
    return mean_test_score[best_test_score_idx], params[best_test_score_idx]

def random_forest(features, labels, num_folds=5):
    clf = RandomForestClassifier(n_estimators = 87, random_state=0, oob_score = True, max_features=11)
    clf.fit(features, labels)
    #scores = cross_val_score(clf, features, labels, cv=num_folds)
    return clf

def boost(features, labels, num_folds=5):
    xgboost_model = xgb.XGBClassifier()
    scores = cross_val_score(xgboost_model, features, labels, cv=num_folds)
    return scores
