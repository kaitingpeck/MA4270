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
