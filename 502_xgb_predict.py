import pandas as pd
import joblib
import time
import xgboost
pd.set_option('display.max_columns', None)
from _threshold_exploration import f1_maximization
from utils import keep_top_features

start_time = time.time()
data_folder = 'data'


def xgb_predict_proba_pipe(X, data_folder='data', return_data=False, scaling=False):
    X = X.drop(['order_id', 'user_id', 'product_id'], axis=1)
    if 'reordered' in X.columns:
        # test data
        X = X.drop('reordered', axis=1)
        X = keep_top_features(X)
    if scaling:
        scaler = joblib.load('{}/xgb_scaler.joblib'.format(data_folder))
        X = scaler.transform(X)

    # import trained model
    xgb = xgboost.XGBClassifier()
    xgb.load_model('{}/xgb_model.json'.format(data_folder))
    predicted_prob = xgb.predict_proba(X)[:, 1]
    if return_data:
        return predicted_prob, X
    return predicted_prob

#
# # get best threshold based on training data
# train_full_features = pd.read_pickle('data/train_full_features.pickle')
# train_y = train_full_features['reordered']
# train_predicted_proba, X_train = xgb_predict_proba_pipe(train_full_features, return_data=True)
# # max_f1, best_threshold_f1 = f1_maximization(train_predicted_proba, train_y)
# # print(max_f1, best_threshold_f1)
#
# # save train prediction res
# train_full_features['pred_proba'] = train_predicted_proba
# # train_full_features['pred_y'] = (train_predicted_proba > best_threshold_f1).astype(int)
# train_full_features.to_pickle('{}/train_pred_prob_with_features.pickle'.format(data_folder))
#
# # get best threshold based on val data
# # pred val set
# X_val = pd.read_pickle('data/X_val.pickle')
# y_val = pd.read_pickle('data/y_val.pickle')
# val_pred_proba = xgb_predict_proba_pipe(X_val, return_data=False)
# X_val['pred_proba'] = val_pred_proba
# X_val['reordered'] = y_val
# X_val[['user_id', 'order_id', 'product_id', 'pred_proba', 'reordered']].to_pickle('{}/val_pred_prob_res.pickle'.format(data_folder))
# # max_f1, best_threshold_f1 = f1_maximization(X_val['pred_proba'], X_val['reordered'])


# to predict test set
print('predicting on test data')
test_full_features = pd.read_pickle('data/test_full_features.pickle')
test_predicted_proba, X_test_scaled = xgb_predict_proba_pipe(test_full_features, return_data=True)
# pd.DataFrame(X_test_scaled, columns=test_full_features.columns).to_pickle('{}/X_test_scaled.pickle'.format(data_folder))
test_full_features['pred_proba'] = test_predicted_proba
# test_full_features['reordered'] = (test_predicted_proba > best_threshold_f1).astype(int)
print('if run here')
test_full_features[['user_id', 'order_id', 'product_id', 'pred_proba']].to_pickle('{}/test_pred_prob_res.pickle'.format(data_folder))


end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))