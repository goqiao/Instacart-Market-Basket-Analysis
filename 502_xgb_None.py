import numpy as np
import xgboost
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from utils import split_data, print_eval_metrics, save_fig_with_timestamp
import matplotlib.pyplot as plt
import time
import mlflow

# takes 18 mins
start_time = time.time()

# experiment setting
experiment_name = 'Instacart Predict None'
run_name = 'eta 0.1, 1000 trees, early_stopping_rounds=8, all compressed up features'

data_folder = 'data'
sample_frac = 1
test_size = 0.2
scaling = False

data_full_features = pd.read_pickle('{}/train_None_full_features.pickle'.format(data_folder))
# data_full_features = data_full_features.sample(frac=sample_frac, random_state=0)

# TODO: drop user_id
X_train, X_val, y_train, y_val = train_test_split(data_full_features.drop('is_None', axis=1),
                                                  data_full_features['is_None'], test_size=test_size)

X_val.to_pickle('{}/X_val_None.pickle'.format(data_folder))
y_val.to_pickle('{}/y_val_None.pickle'.format(data_folder))
weight = np.sum(y_train==0)/np.sum(y_train==1)
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
    , "random_state": 19
    , "scale_pos_weight": 1
}
xgb_params['scale_pos_weight'] = weight
bst = xgboost.XGBClassifier(**xgb_params)
#TODO: imbalanced dataset, shall we give weights to 1
bst.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=8, verbose=True)

# access performance
num_best_ntrees = bst.best_ntree_limit
res = bst.evals_result()
train_logloss = res["validation_0"]["logloss"][-1]
val_logloss = res["validation_1"]["logloss"][-1]

train_auc = res["validation_0"]["auc"][-1]
val_auc = res["validation_1"]["auc"][-1]

print('saving model')
bst.save_model("{}/xgb_None_model.json".format(data_folder))

# feature importance
feature_importance = pd.DataFrame({'features': X_train.columns.astype('str'),
                                   'importance': bst.feature_importances_})
feature_importance.sort_values(by='importance', ascending=False, inplace=True)
feature_importance.to_csv('{}/feature_importance.csv'.format(data_folder), index=False)

# plot feature importance
fig = plt.figure(figsize=(25, 20))
plt.barh(feature_importance['features'], feature_importance['importance'])
plt.tight_layout()
# save_fig_with_timestamp('XGB-feature-importance')
# plt.show();  # seem to cause freeze

end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))

# start logging
try:
    exp_id = mlflow.create_experiment(experiment_name)
except:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
    mlflow.log_params(xgb_params)
    mlflow.log_params({'num_best_ntrees': num_best_ntrees})
    mlflow.log_params({'sample_frac': sample_frac, 'test_size': test_size})
    mlflow.log_metrics({'train_logloss': train_logloss, 'val_logloss': val_logloss, 'train_auc': train_auc,
                        'val_auc': val_auc, 'duration_mins': time_spent})
    mlflow.log_figure(fig, "feature importance.png")
    mlflow.xgboost.log_model(bst, 'bst_test')
    mlflow.log_artifact('{}/feature_importance.csv'.format(data_folder))


