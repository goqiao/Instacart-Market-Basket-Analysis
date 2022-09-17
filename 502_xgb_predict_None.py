import pandas as pd
import joblib
import time
import xgboost
pd.set_option('display.max_columns', None)
from _threshold_exploration import f1_maximization

start_time = time.time()
data_folder = 'data'

def xgb_predict_proba_pipe(X, data_folder='data', return_data=False, scaling=False):
    if 'is_None' in X.columns:
        X = X.drop(['is_None'], axis=1)
    if scaling:
        scaler = joblib.load('{}/xgb_scaler.joblib'.format(data_folder))
        X = scaler.transform(X)

    # import trained model
    xgb = xgboost.XGBClassifier()
    xgb.load_model('{}/xgb_None_model.json'.format(data_folder))
    predicted_prob = xgb.predict_proba(X)[:, 1]
    if return_data:
        return predicted_prob, X
    return predicted_prob


# train_full_features = pd.read_pickle('data/train_None_full_features.pickle')
# train_y = train_full_features['is_None']
X_val = pd.read_pickle('data/X_val_None.pickle')
y_val = pd.read_pickle('data/y_val_None.pickle')
val_predicted_proba, X_train = xgb_predict_proba_pipe(X_val, return_data=True)
# max_f1, best_threshold_f1 = f1_maximization(val_predicted_proba, y_val)
X_val['pred_None_proba'] = val_predicted_proba
X_val['reordered'] = y_val
X_val[['user_id', 'pred_None_proba', 'reordered']].to_pickle('{}/val_None_pred_res.pickle'.format(data_folder))
# save train prediction res
# train_predicted_proba = xgb_predict_proba_pipe(train_full_features, return_data=False)
# train_full_features['pred_None_proba'] = train_predicted_proba
# train_full_features[['user_id', 'pred_None_proba']].to_pickle('{}/train_None_prediction_res.pickle'.format(data_folder))
# train_full_features['pred_y'] = (train_predicted_proba > best_threshold_f1).astype(int)
# train_full_features.to_pickle('{}/train_None_prediction_with_features.pickle'.format(data_folder))


# to predict test set
print('predicting on test data')
test_full_features = pd.read_pickle('data/test_None_full_features.pickle')
test_predicted_proba, X_test_scaled = xgb_predict_proba_pipe(test_full_features, return_data=True)
# pd.DataFrame(X_test_scaled, columns=test_full_features.columns).to_pickle('{}/X_test_scaled.pickle'.format(data_folder))
test_full_features['pred_None_proba'] = test_predicted_proba
# test_full_features['pred_is_None'] = (test_predicted_proba > best_threshold_f1).astype(int)
test_full_features[['user_id', 'pred_None_proba']].to_pickle('{}/test_None_prediction_res.pickle'.format(data_folder))


end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))