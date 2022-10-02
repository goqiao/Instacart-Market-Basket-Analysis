import xgboost
import pandas as pd
import time
import mlflow
import pickle
from utils import keep_top_gain_features
from sklearn.model_selection import GroupKFold

"""
customized cv to check performance
"""

start_time = time.time()
data_folder = 'data'
sample_frac = 0.4

experiment_name = 'Instacart CV'
run_name = 'select top gain features 143 + substitution features, frac 0.4'


# data
data_full_features = pd.read_pickle('{}/train_full_features.pickle'.format(data_folder))

## substitues features
data_folder = 'data'
index_cols = ['user_id', 'product_id']
up_word2vec_substitute_purchase = pd.read_pickle('{}/up_word2vec_substitute_purchase.pickle'.format(data_folder)).set_index(index_cols)
up_word2vec_substitute_purchase_07 = pd.read_pickle('{}/up_word2vec_substitute_purchase_07.pickle'.format(data_folder)).set_index(index_cols)
product_sub_stats = pd.read_pickle('{}/product_sub_stats.pickle'.format(data_folder)).set_index('product_id')
product_sub_stats.drop('p_num_substitute', axis=1, inplace=True)
# print(product_sub_stats.head(1))

data_full_features = data_full_features.merge(up_word2vec_substitute_purchase, how='left', left_on=index_cols, right_index=True)
data_full_features = data_full_features.merge(up_word2vec_substitute_purchase_07, how='left', left_on=index_cols, right_index=True)
data_full_features = data_full_features.merge(product_sub_stats, how='left', left_on='product_id', right_index=True)
# print(data_full_features.iloc[:2, -8:])

data_full_features = data_full_features.sample(frac=sample_frac, random_state=1).reset_index(drop=True)
X = data_full_features.drop('reordered', axis=1)
y = data_full_features['reordered']

drop_cols = ['order_id', 'user_id', 'product_id']
X = X.drop(columns=drop_cols)

X = keep_top_gain_features(X)
print(1111)
print (X.shape)

# spliter
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
