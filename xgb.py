import xgboost
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import split_data, print_eval_metrics
import matplotlib.pyplot as plt
import numpy as np
import time

start_time = time.time()
data_folder = 'data'
test_size = 0.9
data_full_features = pd.read_pickle('{}/train_full_features.pickle'.format(data_folder))

X_train, X_test, y_train, y_test = split_data(data_full_features, test_size=test_size, data_folder=data_folder)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# adding feature names back for better feature importance visualization
# X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# X_train_scaled.to_pickle('{}/X_train_scaled.pickle'.format(data_folder))
y_train.fillna(0, inplace=True)
y_train.to_pickle('{}/y_train.pickle'.format(data_folder))

# XGB input data
d_train = xgboost.DMatrix(X_train_scaled, y_train, feature_names=list(X_train.columns.astype('str')), nthread=-1)
xgb_params = {
    "objective": "binary:logistic"
    , "eval_metric": "logloss"
    , "eta": 0.1
    , "max_depth": 6
    , "min_child_weight": 10
    , "gamma": 0.70
    , "subsample": 0.76
    , "colsample_bytree": 0.95
    , "alpha": 2e-05
    , "lambda": 10
}

watchlist = [(d_train, "train")]
bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)

#
print(bst.best_ntree_limit)

# feature importance
# fig, ax = plt.subplots(figsize=(30, 15))
# xgboost.plot_importance(bst, ax=ax)
# plt.show()

# train performance:
# train f-1, auc, log-loss
threshold = 0.21
y_pred_prob = bst.predict(xgboost.DMatrix(X_train_scaled, feature_names=list(X_train.columns.astype('str')), nthread=-1))
np.savetxt('{}/xgb_pred_prob.txt'.format(data_folder), y_pred_prob)

y_pred = y_pred_prob > threshold
print_eval_metrics(y_train, y_pred_prob, y_pred)

# CV performance
# from xgboost import cv
#
# xgb_cv = cv(dtrain=d_train, params=xgb_params, nfold=5,
#             num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)

# print(xgb_cv)

end_time = time.time()
print('spent {:.2f} mins'.format((end_time - start_time) / 60))
