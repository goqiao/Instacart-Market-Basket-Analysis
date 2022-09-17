import pandas as pd
import numpy as np
import shap
import xgboost
import matplotlib.pyplot as plt
import time
import mlflow
from matplotlib import rcParams
from utils import download_user_order_history
rcParams.update({'figure.autolayout': True})
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


start_time = time.time()
data_folder = 'data'
experiment_name = 'SHAP Force Plot'
pid = 39581
uid = 92665
run_name = 'userid: {} productid: {}'.format(uid, pid)
download_user_order_history(uid, pid)


xgb = xgboost.XGBClassifier()
xgb.load_model('{}/xgb_model.json'.format(data_folder))
features = pd.read_pickle('{}/X_val.pickle'.format(data_folder))
# train_pred_with_features = pd.read_pickle('{}/train_prediction_with_features.pickle'.format(data_folder))
# train_wrong_pred = train_pred_with_features.loc[train_pred_with_features['reordered'] != train_pred_with_features['pred_y']]

# train_wrong_pred = train_wrong_pred.sample(10, random_state=0)

instance = features.loc[(features.user_id == uid) & (features['product_id']==pid)]
features = features.drop(columns=['order_id', 'user_id', 'product_id'])
instance = instance.drop(columns=['order_id', 'user_id', 'product_id'])
print('dataset to look into:')
print(instance)
print(instance.shape)

booster = xgb.get_booster()
model_bytearray = booster.save_raw()[4:]
booster.save_raw = lambda : model_bytearray

# explainer = shap.TreeExplainer(booster)
features = features.fillna(-1)
explainer = shap.TreeExplainer(booster, model_output='probability', data=features.head(100).values.astype(np.float64))
shap_values = explainer.shap_values(instance)

# print(shap_values)
# print(shap_values.values)
# print(shap_values[0, :])

# fig = plt.figure(figsize=(25, 60))
# shap.plots.waterfall(shap_values)
print(shap_values.shape)

shap.waterfall_plot(expected_value=explainer.expected_value, shap_values=shap_values[0], feature_names=instance.columns)
plt.show()
plt.tight_layout()
print(explainer.expected_value)

# shap.force_plot(explainer.expected_value, shap_values, feature_names=instance.columns, matplotlib=True, link='logit')
# plt.show()

# try:
#     exp_id = mlflow.create_experiment(experiment_name)
# except:
#     exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
# with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
    # mlflow.log_figure(fig, "uid_{}_pid_{}_SHAP_explain.png".format(uid, pid))

end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))