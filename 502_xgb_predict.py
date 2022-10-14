import pandas as pd
import numpy as np
import joblib
import time
import xgboost

pd.set_option("display.max_columns", None)
from utils import feature_selection

start_time = time.time()
data_folder = "data"


def xgb_predict_proba_pipe(X, data_folder="data", return_data=False, scaling=False):
    X = X.drop(["order_id", "user_id", "product_id"], axis=1)
    if "reordered" in X.columns:
        # test data
        X = X.drop("reordered", axis=1)
        X = feature_selection(X)
        assert X.columns.nunique() == 123
    if scaling:
        scaler = joblib.load("{}/xgb_scaler.joblib".format(data_folder))
        X = scaler.transform(X)

    # import trained model
    xgb = xgboost.XGBClassifier()
    xgb.load_model("{}/xgb_model.json".format(data_folder))
    predicted_prob = xgb.predict_proba(X)[:, 1]

    if return_data:
        return predicted_prob, X
    return predicted_prob


# to predict test set
test_full_features = pd.read_pickle("data/test_full_features.pickle")
test_predicted_proba = xgb_predict_proba_pipe(test_full_features, return_data=False)
test_full_features["pred_proba"] = test_predicted_proba

test_full_features[["user_id", "order_id", "product_id", "pred_proba"]].to_pickle(
    "{}/test_pred_prob_res.pickle".format(data_folder)
)


end_time = time.time()
time_spent = (end_time - start_time) / 60
print("spent {:.2f} mins".format(time_spent))
