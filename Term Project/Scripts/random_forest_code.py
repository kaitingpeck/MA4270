# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import time

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel


# one-hot encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# xgboost
import xgboost as xgb

# for saving models
import pickle

# for plotting of confusion matrices
from sklearn.metrics import confusion_matrix
import itertools

import os
os.getcwd()

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

#Data cleaning

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

# preview the first 5 rows
train.head(5)

test = combined_data[combined_data['Survived']==-1]
test.drop('Survived', axis=1, inplace=True)
# test.to_csv("./Data/test-clean.csv")

# One-hot encoding
train_encoded = pd.get_dummies(train, columns = ['Embarked', 'Sex'])
test_encoded = pd.get_dummies(test, columns = ['Embarked', 'Sex'])

# Rearrange columns
list_of_features = ['Age','Embarked_C','Embarked_Q','Embarked_S','Fare','Parch','Pclass','Sex_female','Sex_male','SibSp'] #0478
list_of_columns = list_of_features + ['Survived']
train_encoded = train_encoded[list_of_columns]
test_encoded = test_encoded[list_of_features]

# Transform training and testing data into np arrays
train_x = train_encoded[list_of_features].values
test_x = test_encoded[['Age', 'Fare', 'Sex_female', 'Sex_male']].values
train_y = train_encoded['Survived'].values

list_classes = ['survive','don\'t survive']

"""
This cell defines functions to compute the performance of any given model.
"""

def compute_f1(model, X, y,k_folds):
    """
    Given a model and the evaluation data, returns the F1 score.
    """
    return np.mean(cross_val_score(model, X, y, cv=k_folds, scoring='f1_weighted'))

def accuracy(model, X, y,k_folds):
    """
    Given a model and the evaluation data, returns the accuracy
    score evaluated using cross validation.
    """
    return np.mean(cross_val_score(model, X, y, cv=k_folds, scoring='accuracy'))

def print_score_model(model,train_x,train_y, k_folds):
    print("F1 score is",compute_f1(model,train_x,train_y,k_folds))
    print("Accuracy is",accuracy(model,train_x,train_y,k_folds))

def random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators = 90, random_state=2, oob_score = True, criterion='gini')
    scores = cross_val_score(rf, X_train, y_train, cv=10, scoring='accuracy')
    print('Mean scores (before feature selection): ' + str(scores.mean()))
    rf = rf.fit(X_train, y_train)
    print(rf.feature_importances_)  

    model = SelectFromModel(rf, prefit=True)
    X_new = model.transform(X_train)
    # print(X_new[:5])
    rf = rf.fit(X_new, y_train)
    return rf

#start=time.clock()
rf = random_forest(train_x, train_y)
'''end=time.clock()
print('scores after feature selection: ' + str(scores.mean()))
print('Time taken: ' + str(end-start))'''
predictions = rf.predict(test_x)
predictions = [int(i) for i in predictions]

submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],'Survived': predictions })
submission.to_csv("submission.csv", index=False)
  
'''def random_forest(X, y, num_trees, num_folds=5):
    clf = RandomForestClassifier(n_estimators = int(num_trees), random_state = 2, oob_score = True)
    scores = cross_val_score(clf, X, y, cv=num_folds)
    return scores.mean()'''

scores = []
def run_rf_diff_trees(X, y, num_folds=5):
    x = np.linspace(30, 100, 71).reshape(-1,1)
    param_trees = x.reshape(1,71).tolist()[0]
    # param_trees = [30,40,50,60,70,80,90,100]
    for num_trees in param_trees:
        print('... Running random forest with ' + str(num_trees) + ' trees ...')
        score = random_forest(X, y, num_trees, 10)
        print(score)
        scores.append(score)
    highest = max(scores)
    idx = scores.index(highest)
    print('Highest accuracy: ' + str(highest))
    print('Achieved with index: ' + str(param_trees[idx]) + ' trees')
        
    fit = np.polyfit(param_trees,scores,2)
    fit_fn = np.poly1d(fit) 
    # fit_fn is now a function which takes in x and returns an estimate for y

    
    #plt.xlim(0, 5)
    #plt.ylim(0, 12)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Scores vs no. of trees in random forest', fontdict={'size':30})
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    axis.plot(param_trees,scores, 'bo', param_trees, fit_fn(param_trees), '--k')
    # axis.plot(param_trees, scores, 'b+')
    axis.set_ylabel('scores', fontdict={'size':20})
    axis.set_xlabel('num_trees', fontdict={'size':20})
    fig.show()

# run_rf_diff_trees(train_x, train_y, 5)
