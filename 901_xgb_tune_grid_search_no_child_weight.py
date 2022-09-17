import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
import time
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import mlflow

"""
summary:
- tested lambda range [0, 10] and 10 was selected the best, try to expand upper_b
- test col_sample [0.4, 1, 0.2], and 0.4 is selected, try to expand lowe_b
- with eta=0.02, and early stopping rounds = 36, n_estimators can grow more than 1000, consider 
  extending tree limits. 
- try grid search instead overnight as n_iter might not be enough
- final early stopping rounds need to be 50 with 1000 trees
"""
start_time = time.time()
sample_frac = 0.3
test_size = 0.2
data_folder = 'data'
early_stopping_rounds = 50

dry_run = True
experiment_name = 'Instacart Tune Hyperparameters'
run_name = 'Gridsearch, eta 0.3, 0.02, early_stopping = 50'
# eta=0.3 -> 0.02, early_stopping=50

X_train = pd.read_pickle('{}/X_train.pickle'.format(data_folder))
y_train = pd.read_pickle('{}/y_train.pickle'.format(data_folder))
X_val = pd.read_pickle('{}/X_val.pickle'.format(data_folder))
y_val = pd.read_pickle('{}/y_val.pickle'.format(data_folder))
if dry_run:
    # small sample to dry run
    X_train = X_train.head(1000)
    y_train = y_train.head(1000)
    X_val = X_val.head(1000)
    y_val = y_val.head(1000)


def xgb_quick_fit(model, X_train, y_train, X_val, y_val, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        # xgtest = xgb.DMatrix(X_val.values)
        # train model on train with early stopping
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='logloss', early_stopping_rounds=early_stopping_rounds)
        # set rounds as early stopped rounds
        best_n_estimators = cvresult.shape[0]
        print('best_n_estimators', best_n_estimators)
        model.set_params(n_estimators=best_n_estimators)
    # refit data
    model.fit(X_train, y_train)
    # Predict val set:
    train_pred = model.predict(X_train)
    train_pred_prob = model.predict_proba(X_train)[:, 1]
    # Print model report:
    print("\n Model Report")
    print("Logloss : %.4g" % metrics.log_loss(y_train, train_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, train_pred_prob))

    # Predict on testing data:
    val_pred = model.predict(X_val)
    val_pred_prob = model.predict_proba(X_val)[:, 1]
    print('AUC Score (Test): %f' % metrics.roc_auc_score(y_val, val_pred_prob))
    return best_n_estimators, model
"""
Step 1: Find the number of estimators for a high learning rate
- xgboost
find optimal num trees
"""
# print('\n Tune n_estimators')
# xgb1 = XGBClassifier(
#         learning_rate=0.3,
#         n_estimators=1000,
#         objective='binary:logistic',
#         # eval_metric='logloss',
#         max_depth=6,
#         gamma=0,
#         subsample=0.75,
#         colsample_bytree=0.8,
#         tree_method='hist',
#         random_state=0,
#         )
# 
# best_n_estimators, _ = xgb_quick_fit(xgb1, X_train, y_train, X_val, y_val, useTrainCV=True, cv_folds=5, early_stopping_rounds=early_stopping_rounds)
# print('best_n_estimators', best_n_estimators)
# 
# print('\n Tune Max Depth')
# param_test1 = {
#     'max_depth':[4, 6, 8, 10]
# }
# xgb2 = XGBClassifier(
#         learning_rate=0.3,
#         n_estimators=best_n_estimators,
#         objective='binary:logistic',
#         max_depth=6,
#         gamma=0,
#         subsample=0.75,
#         colsample_bytree=0.8,
#         tree_method='hist',
#         random_state=0
# )
# 
# random_search1 = GridSearchCV(xgb2, param_grid=param_test1, scoring='neg_log_loss', cv=5)
# random_search1.fit(X_train, y_train)
# print(random_search1.cv_results_['mean_test_score'])
# print(random_search1.best_score_)
# print(random_search1.best_params_)
# best_max_depth = random_search1.best_params_['max_depth']
# print('best_max_depth:', best_max_depth)


# print('\n Tune Max Gamma')
# best_n_estimators = 68
# best_max_depth = 6
# param_test2 = {
#     # 'gamma':np.arange(0, 1, 0.2) best -> 0.8
#     # 'gamma':np.arange(1, 6, 1)   best -> 5
#     # 'gamma':[5, 8, 10, 12, 16, 20] best -> 5
#     'gamma':np.arange(7, 8, 0.2)   # best -> 7.8
# }
#
# xgb3 = XGBClassifier(
#         learning_rate=0.3,
#         n_estimators=best_n_estimators,
#         objective='binary:logistic',
#         max_depth=best_max_depth,
#         gamma=0,
#         subsample=0.75,
#         colsample_bytree=0.8,
#         tree_method='hist',
#         random_state=0
# )
#
# random_search2 = GridSearchCV(xgb3, param_grid=param_test2, scoring='neg_log_loss', cv=5)
# random_search2.fit(X_train, y_train)
# print(random_search2.cv_results_['mean_test_score'])
# print(random_search2.best_score_)
# best_gamma = random_search2.best_params_['gamma']
# print('best_gamma', best_gamma)


# # print('\n Tune subsample and colsample_bytree')
# best_n_estimators = 68
# best_max_depth = 6
# best_gamma = 7.8
# param_test3 = {
#  'subsample':[0.5, 0.75, 1],
#  'colsample_bytree':[0.4, 0.6, 0.8, 1]
# }
#
# xgb4 = XGBClassifier(
#         learning_rate=0.3,
#         n_estimators=best_n_estimators,
#         objective='binary:logistic',
#         max_depth=best_max_depth,
#         gamma=best_gamma,
#         subsample=0.75,
#         colsample_bytree=0.8,
#         tree_method='hist',
#         random_state=0
# )
#
# random_search3 = GridSearchCV(xgb4, param_grid=param_test3, scoring='neg_log_loss', cv=5)
# random_search3.fit(X_train, y_train)
# print(pd.DataFrame({'param_subsample': random_search3.cv_results_['param_subsample'],
#                     'param_colsample_bytree':random_search3.cv_results_['param_colsample_bytree'],
#                     'mean_test_score_logloss': random_search3.cv_results_['mean_test_score']}))
# print(random_search3.best_score_)
# best_params = random_search3.best_params_
# print(best_params)
# best_subsample, best_colsample_bytree = best_params['subsample'], best_params['colsample_bytree']



# print('\n Tune alpha and lambda')
# best_n_estimators = 68
# best_max_depth = 6
# best_gamma = 7.8
# best_subsample = 1
# best_colsample_bytree = 0.4
# param_test4 = {
#  'reg_alpha':[0.005],  # best_reg_alpha = 0.005
#  'reg_lambda':[11, 15, 18, 19, 20]      # best_reg_lambda = 20
# }
# xgb5 = XGBClassifier(
#         learning_rate=0.3,
#         n_estimators=best_n_estimators,
#         objective='binary:logistic',
#         max_depth=best_max_depth,
#         gamma=best_gamma,
#         subsample=best_subsample,
#         colsample_bytree=best_colsample_bytree,
#         tree_method='hist',
#         random_state=0
# )
# random_search4 = GridSearchCV(xgb5, param_grid=param_test4, scoring='neg_log_loss', cv=5)
# random_search4.fit(X_train, y_train)
# print(pd.DataFrame({'param_reg_alpha': random_search4.cv_results_['param_reg_alpha'],
#                     'param_reg_lambda':random_search4.cv_results_['param_reg_lambda'],
#                     'mean_test_score_logloss': random_search4.cv_results_['mean_test_score']}))
# print(random_search4.best_score_)
# best_params = random_search4.best_params_
# print(best_params)
# best_reg_alpha, best_reg_lambda = best_params['reg_alpha'], best_params['reg_lambda']
# 
# 
# print('\n Check if the tuned one has better performance?')
best_n_estimators = 68
best_max_depth = 6
best_gamma = 7.8
best_subsample = 1
best_colsample_bytree = 0.4
best_reg_alpha, best_reg_lambda = 0.005, 20
# xgb6 = XGBClassifier(
#         learning_rate=0.3,
#         n_estimators=best_n_estimators,
#         objective='binary:logistic',
#         max_depth=best_max_depth,
#         gamma=best_gamma,
#         subsample=best_subsample,
#         colsample_bytree=best_colsample_bytree,
#         tree_method='hist',
#         reg_alpha=best_reg_alpha,
#         reg_lambda=best_reg_lambda,
#         random_state=0
# )
# best_n_estimators, model = xgb_quick_fit(xgb6, X_train, y_train, X_val, y_val, useTrainCV=True, cv_folds=5, early_stopping_rounds=early_stopping_rounds)
# print('model \n', model)
# 
print('\n Re-tune best_n_estimators')
xgb6 = XGBClassifier(
        learning_rate=0.02,
        n_estimators=1000,
        objective='binary:logistic',
        max_depth=best_max_depth,
        gamma=best_gamma,
        subsample=best_subsample,
        colsample_bytree=best_colsample_bytree,
        tree_method='hist',
        reg_alpha=best_reg_alpha,
        reg_lambda=best_reg_lambda,
        random_state=0
)

best_n_estimators, final_model = xgb_quick_fit(xgb6, X_train, y_train, X_val, y_val, useTrainCV=True, cv_folds=5,
                                               early_stopping_rounds=early_stopping_rounds)
print('best_n_estimators', best_n_estimators)
print('final_model: \n', final_model)
# #
# #
# #
# start logging
try:
    exp_id = mlflow.create_experiment(experiment_name)
except:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
    mlflow.log_params(final_model.get_params())
    mlflow.log_params({'early stopping rounds': early_stopping_rounds})
    mlflow.log_params({'sample_frac': sample_frac, 'test_size': test_size})
    mlflow.xgboost.log_model(final_model, 'bst_test')



end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))