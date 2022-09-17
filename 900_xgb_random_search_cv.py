import numpy as np
import xgboost
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from utils import split_data, print_eval_metrics, save_fig_with_timestamp
import matplotlib.pyplot as plt
import time
import mlflow
from sklearn.model_selection import RandomizedSearchCV, GroupKFold

"""
seems to use a really long time
"""
start_time = time.time()
sample_frac = 0.3
test_size = 0.2
data_folder = 'data'

# data_full_features = pd.read_pickle('{}/train_full_features.pickle'.format(data_folder))

# due to memory limit, sampling
# data_full_features = data_full_features.sample(frac=sample_frac, random_state=0)
# X_train, X_val, y_train, y_val = split_data(data_full_features, test_size=test_size, data_folder=data_folder,
#                                             split_by='user_id')

X_train = pd.read_pickle('{}/X_train.pickle'.format(data_folder))
y_train = pd.read_pickle('{}/y_train.pickle'.format(data_folder))
X_val = pd.read_pickle('{}/X_val.pickle'.format(data_folder))
y_val = pd.read_pickle('{}/y_val.pickle'.format(data_folder))

cv_split_base = X_train['user_id']
# X = data_full_features.drop('reordered', axis=1)
# y = data_full_features['reordered']
# cv_split_base = X['user_id']
# drop_cols =['order_id', 'user_id', 'product_id']
# X = X.drop(columns=drop_cols)

drop_cols = ['order_id', 'user_id', 'product_id']
X_train = X_train.drop(columns=drop_cols)
X_val = X_val.drop(columns=drop_cols)

param_grid = {'n_estimators': [100, 400, 800, 1000],
              'max_depth': [4, 6, 8, 10],
              'learning_rate':[0.02, 0.1, 0.2],
              'subsample':np.arange(0.5, 1.0, 0.2),
              'colsample_bytree':np.arange(0.4, 1.0, 0.2)}

bst = xgboost.XGBClassifier()

rs = RandomizedSearchCV(estimator=bst,
                       param_distributions=param_grid,
                       n_iter= 2,
                       scoring='neg_log_loss',
                       # n_jobs = -1,
                       cv=GroupKFold(n_splits=5).split(X_train, y_train, cv_split_base),
                       random_state=0,
                       return_train_score=True,
                       refit=True
                                   )

rs.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric ='logloss', early_stopping_rounds=30, verbose=True)
best_bst = rs.best_estimator_

print("Best parameters:", rs.best_params_)
print('CV result: \n', pd.DataFrame(rs.cv_results_))
print('Best Estimator:', best_bst)

end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))

print('end')










