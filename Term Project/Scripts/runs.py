# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import ml_programs as ml

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


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
import time

os.getcwd()

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    title = title + '.jpg'
    plt.show()
    plt.savefig(title, format='jpeg')

def generate_confusion_matrix(y_val, y_pred, plot_title, list_classes=['survive','don\'t survive']):
    '''
    takes in y_val (ground truth for validation set), y_pred (predicted values for validation set),
    plot_title (name of plot), list_classes (['survive', 'dont survive'])
    '''
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=list_classes, normalize=True,
                              title= plot_title)
    
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
features_train = train_encoded[list_of_features].values
features_test = test_encoded[list_of_features].values
labels_train = train_encoded['Survived'].values

# save arrays to matrix for MATLAB
#scipy.io.savemat('./MATLAB/features_train.mat', dict(features_train=features_train))
#scipy.io.savemat('./MATLAB/labels_train.mat', dict(labels_train=labels_train))

# normalized
#features_norm = preprocessing.scale(features_train)
#scores = ml.run_svm(features_norm, labels_train)
#print('Scores from SVM: ' + str(scores))
                               
# Run SVM on training data
'''start = time.clock()
scores = ml.run_svm(features_train, labels_train)
end = time.clock()
print('Scores from SVM: ' + str(scores))
print('Average score: ' + str(sum(scores)/len(scores)))
print('Time taken: ' + str(end-start))

# Plot confusion matrix
def run_svm(X, y):
    clf = svm.SVC(C=67.91843831809337, gamma=0.0001, max_iter=214748300)
    clf.fit(X,y)
    return clf

def k_fold(X, y, num_folds=5):
    kf = StratifiedKFold(n_splits=n_folds)
    i = 1
    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Obtained trained model using this set of training data
        model = run_svm(X_train, y_train)
        print('Model training completed')
        
        # Run model on validation data
        y_pred = model.predict(X_val)

        # Compute and plot normalized confusion matrix
        plot_title = 'Normalized confusion matrix'
        if i == 2:
            generate_confusion_matrix(y_val, y_pred, plot_title)
        
        # set index for next fold
        i += 1'''

# k_fold(features_train, labels_train, 10)

# Run Random Forest on training data
#rf_scores = ml.random_forest(features_train, labels_train, 10)
#print('Scores from Random Forest: ' + str(sum(rf_scores)/len(rf_scores)))

# Run XGBoost on training data
xg_scores = ml.boost(features_train, labels_train, 5)
print('Scores from XGBoost: ' + str(sum(xg_scores)/len(xg_scores)))

# rf = ml.random_forest(features_train, labels_train)

#predictions = svm.predict(features_test)
#predictions = [int(i) for i in predictions]

#submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],'Survived': predictions })
#submission.to_csv("submission.csv", index=False)

# Run grid search on SVM
#svm_grid_test_score, svm_grid_params = ml.grid_search_svm(features_train, labels_train, 5)
#print('Best test score: ' + str(svm_grid_test_score) + \
     # '\nParameter setting: ' + str(svm_grid_params))

