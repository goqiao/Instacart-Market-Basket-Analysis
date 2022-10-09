import xgboost
import pandas as pd
import time
import mlflow
from utils import custom_refcv_drop, custom_refcv_drop_2
from sklearn.model_selection import GroupKFold

"""
customized cv to check performance
"""

start_time = time.time()
data_folder = 'data'
sample_frac = 0.4

experiment_name = 'Instacart CV'
run_name = '- high_corr custom refcv, 95 features + up order interval (6) + trend in purchase interval(2) + order interval readiness(3), + p_order_interval(12) frac 0.4'


# data
data_full_features = pd.read_pickle('{}/train_full_features.pickle'.format(data_folder))


# up_order_interval = pd.read_pickle('data/up_orders_interval.pickle')
# data_full_features = data_full_features.merge(up_order_interval, how='left'
#                         , left_on=['user_id', 'product_id'], right_index=True)


data_full_features = data_full_features.sample(frac=sample_frac, random_state=1).reset_index(drop=True)
X = data_full_features.drop('reordered', axis=1)
y = data_full_features['reordered']

drop_cols = ['order_id', 'user_id', 'product_id']
X = X.drop(columns=drop_cols)

X = custom_refcv_drop(X)
X = custom_refcv_drop_2(X)

assert X.shape[1] == 118


# splitter
cv_split_base = data_full_features['user_id']
gkf = GroupKFold(n_splits=5).split(X, y, groups=cv_split_base)
del data_full_features

# model
xgb_params = {
    'n_estimators': 1000
    , "objective": "binary:logistic"
    , "eval_metric": ['auc', 'logloss']
    , "eta": 0.1
    , "max_depth": 6
    , "min_child_weight": 10
    , "gamma": 0.70
    , "subsample": 0.76
    , "colsample_bytree": 0.8
    , "alpha": 2e-05
    , "lambda": 10
    , "tree_method": 'hist'
    , "early_stopping_rounds": 30
    , "random_state": 19
    , "predictor": 'cpu_predictor'
}
bst = xgboost.XGBClassifier(**xgb_params)

# cv: cross_val_score() method doesn't support xgb with early stopping, so use customized cv
cv_res = {'train_cv_logloss': [], 'train_cv_auc': [], 'val_cv_logloss': [], 'val_cv_auc': []}
i = 1
for train_idx, test_idx in gkf:
    print('Fold {}'.format(i))

    X_train, X_val = X.loc[train_idx], X.loc[test_idx]
    y_train, y_val = y.loc[train_idx], y.loc[test_idx]

    bst.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    res = bst.evals_result()

    cv_res['train_cv_logloss'].append(res["validation_0"]["logloss"][-1])
    cv_res['val_cv_logloss'].append(res["validation_1"]["logloss"][-1])
    cv_res['train_cv_auc'].append(res["validation_0"]["auc"][-1])
    cv_res['val_cv_auc'].append(res["validation_1"]["auc"][-1])

    i += 1

print(pd.DataFrame(cv_res))
cv_res_mean = dict(pd.DataFrame(cv_res).mean().round(5))
print(cv_res_mean)


# saving features
features = str(X.columns.values)
with open('data/features.txt', 'w') as f:
    f.write(features)
# logging to mlflow
try:
    exp_id = mlflow.create_experiment(experiment_name)
except:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
    mlflow.log_params(xgb_params)
    mlflow.log_params({'sample_frac': sample_frac})
    mlflow.log_metrics(cv_res_mean)
    mlflow.log_params({'num_features': X.shape[1]})
    mlflow.log_artifact('data/features.txt')

end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))
