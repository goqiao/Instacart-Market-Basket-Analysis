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

start_time = time.time()
data_folder = 'data'
experiment_name = 'SHAP Feature Importance'
run_name = 'X train'



xgb = xgboost.XGBClassifier()
xgb.load_model('{}/xgb_model.json'.format(data_folder))
df = pd.read_pickle('{}/X_val.pickle'.format(data_folder))

X_train = pd.read_pickle('data/X_train.pickle').drop(columns=['order_id', 'user_id', 'product_id'])
y_train = pd.read_pickle('data/y_train.pickle')
explainer = shap.TreeExplainer(xgb)
shap_values = explainer(X_train)
print('Expected Target Value?', y_train.mean())

fig = plt.figure()
shap.summary_plot(shap_values)
plt.tight_layout()
plt.show()

try:
    exp_id = mlflow.create_experiment(experiment_name)
except:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
with mlflow.start_run(experiment_id=exp_id, run_name=run_name):
    mlflow.log_figure(fig, "SHAP feature importance.png")

end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))