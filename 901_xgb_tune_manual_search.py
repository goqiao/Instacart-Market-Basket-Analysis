import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
import time
from sklearn import metrics
from utils import split_data
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

start_time = time.time()
sample_frac = 0.2
test_size = 0.2
data_folder = 'data'

# data_full_features = pd.read_pickle('{}/train_full_features.pickle'.format(data_folder))
# print('train set BEFORE sampling:')
# print(data_full_features.shape)
# data_full_features = data_full_features.sample(frac=sample_frac, random_state=0)
# print('train set AFTER sampling:')
# print(data_full_features.shape)
#
# X_train, X_val, y_train, y_val = split_data(data_full_features, test_size=test_size, data_folder=data_folder,
#                                             split_by='user_id')

X_train = pd.read_pickle('{}/X_train.pickle'.format(data_folder))
y_train = pd.read_pickle('{}/y_train.pickle'.format(data_folder))
X_val = pd.read_pickle('{}/X_val.pickle'.format(data_folder))
y_val = pd.read_pickle('{}/y_val.pickle'.format(data_folder))

def xgb_quick_fit(model, X_train, y_train, X_val, y_val, useTrainCV=True, cv_folds=5, early_stopping_rounds=30):
    if useTrainCV:
        print('?')
        xgb_param = model.get_xgb_params()
        print('??')
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        print('???')
        # xgtest = xgb.DMatrix(X_val.values)
        # train model on train with early stopping
        print('before cv')
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='logloss', early_stopping_rounds=early_stopping_rounds)
        # set rounds as early stopped rounds
        print('cv finish')
        print(cvresult.shape[0])
        print(cvresult)
        print(3)
        model.set_params(n_estimators=cvresult.shape[0])
    print(4)
    # refit data
    model.fit(X_train, y_train, eval_metric='logloss')
    print(5)
    # Predict val set:
    train_pred = model.predict(X_train)
    train_pred_prob = model.predict_proba(X_train)[:, 1]
    print(6)
    # Print model report:
    print("\n Model Report")
    print("Logloss : %.4g" % metrics.log_loss(y_train, train_pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, train_pred_prob))

    # Predict on testing data:
    val_pred = model.predict(X_val)
    val_pred_prob = model.predict_proba(X_val)[:, 1]
    print('AUC Score (Test): %f' % metrics.roc_auc_score(y_val, val_pred_prob))

"""
Step 1: Find the number of estimators for a high learning rate
- xgboost
find optimal num trees
"""
# xgb1 = XGBClassifier(
#         learning_rate=0.5,
#         n_estimators=200,
#         objective='binary:logistic',
#         max_depth=6,
#         min_child_weight=10,
#         gamma=0,
#         subsample=0.76,
#         colsample_bytree=0.8,
#         tree_method='hist',
#         random_state=0)
# print(2)
# xgb_quick_fit(xgb1, X_train, y_train, X_val, y_val, useTrainCV=True, cv_folds=5, early_stopping_rounds=30)

#Grid seach on subsample and max_features
#Choose all predictors except target & IDcols
# param_test1 = {
#     'max_depth':[4, 6, 8, 10]
#     # 'min_child_weight':range(1,6,2)
# }
# xgb2 = XGBClassifier(
#         learning_rate=0.5,
#         n_estimators=37,
#         objective='binary:logistic',
#         max_depth=6,
#         min_child_weight=10,
#         gamma=0,
#         subsample=0.76,
#         colsample_bytree=0.8,
#         tree_method='hist',
#         random_state=0
# )
# print(1)
# random_search1 = RandomizedSearchCV(xgb2, n_iter=4, param_distributions=param_test1, scoring='neg_log_loss', cv=5)
# print(2)
# random_search1.fit(X_train, y_train)
# print(3)
# print(random_search1.cv_results_)
# print(random_search1.best_score_)
# print(random_search1.best_params_)
# best_max_depth = random_search1.best_params_['max_depth']
# print('best_max_depth:', best_max_depth)

## tune gamma
# param_test2 = {
#     'gamma':np.arange(0, 1, 0.2)
#     # 'min_child_weight':range(1,6,2)
# }
#
# xgb3 = XGBClassifier(
#         learning_rate=0.5,
#         n_estimators=37,
#         objective='binary:logistic',
#         max_depth=best_max_depth,
#         min_child_weight=10,
#         gamma=0,
#         subsample=0.76,
#         colsample_bytree=0.8,
#         tree_method='hist',
#         random_state=0
# )
#
# random_search2 = RandomizedSearchCV(xgb3, n_iter=5, param_distributions=param_test2, scoring='neg_log_loss', cv=5)
# print(2)
# random_search2.fit(X_train, y_train)
# print(3)
# print(random_search2.cv_results_)
# print(random_search2.best_score_)
# best_gamma = random_search2.best_params_
# print('best_gamma', best_gamma)
## best_gamma = 0

## Tune subsample and colsample_bytree
param_test3 = {
 'subsample':[0.5, 0.75, 1],
 'colsample_bytree':[0.4, 0.6, 0.8, 1]
}

xgb4 = XGBClassifier(
        learning_rate=0.5,
        n_estimators=37,
        objective='binary:logistic',
        max_depth=6,
        min_child_weight=10,
        gamma=0,
        subsample=0.76,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=0
)

random_search3 = RandomizedSearchCV(xgb4, n_iter=10, param_distributions=param_test3, scoring='neg_log_loss', cv=5)
print(2)
random_search3.fit(X_train, y_train)
print(3)
print(random_search3.cv_results_)
print(random_search3.best_score_)
best_params = random_search3.best_params_
print(best_params)
print(best_params['subsample'])
print(best_params['colsample_bytree'])


end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))