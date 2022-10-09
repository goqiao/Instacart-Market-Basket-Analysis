import pandas as pd
import numpy as np
import joblib
import time
import xgboost
pd.set_option('display.max_columns', None)
from utils import custom_refcv_drop, custom_refcv_drop_2

start_time = time.time()
data_folder = 'data'


def xgb_predict_proba_pipe(X, data_folder='data', return_data=False, scaling=False):
    X = X.drop(['order_id', 'user_id', 'product_id'], axis=1)
    if 'reordered' in X.columns:
        # test data
        X = X.drop('reordered', axis=1)
        X = custom_refcv_drop(X)
        X = custom_refcv_drop_2(X)
        assert X.columns.nunique() == 118
    if scaling:
        scaler = joblib.load('{}/xgb_scaler.joblib'.format(data_folder))
        X = scaler.transform(X)

    # import trained model
    xgb = xgboost.XGBClassifier()
    xgb.load_model('{}/xgb_model.json'.format(data_folder))

    # predict all at once
    predicted_prob = xgb.predict_proba(X)[:, 1]

    # predict in batches
    # print(4)
    # x_shape = X.shape[0]
    # print(x_shape)
    # res1 = xgb.predict_proba(X.loc[:1500000, :])[:, 1]
    # print(res1.shape)
    # res2 = xgb.predict_proba(X.loc[1500001:3000000, :])[:, 1]
    # print(res2.shape)
    # res3 = xgb.predict_proba(X.loc[3000001:4000000, ])[:, 1]
    # print(res3.shape)
    # res4 = xgb.predict_proba(X.loc[4000001:, ])[:, 1]
    # print(res4.shape)
    # predicted_prob = np.hstack((res1, res2, res3, res4))
    # print(predicted_prob.shape)
    # print(x_shape)

    if return_data:
        return predicted_prob, X
    return predicted_prob


# to predict test set
print('predicting on test data')
test_full_features = pd.read_pickle('data/test_full_features.pickle')
test_predicted_proba = xgb_predict_proba_pipe(test_full_features, return_data=False)
# pd.DataFrame(X_test_scaled, columns=test_full_features.columns).to_pickle('{}/X_test_scaled.pickle'.format(data_folder))
test_full_features['pred_proba'] = test_predicted_proba
# test_full_features['reordered'] = (test_predicted_proba > best_threshold_f1).astype(int)
print('if output')
test_full_features[['user_id', 'order_id', 'product_id', 'pred_proba']].to_pickle('{}/test_pred_prob_res.pickle'.format(data_folder))
print('output finish')

end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))