import xgboost
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from utils import split_data, custom_refcv_drop, custom_refcv_drop_2
import matplotlib.pyplot as plt
import time
import mlflow
import gc

# takes 17 mins
start_time = time.time()
if __name__ == '__main__':
    # experiment setting
    experiment_name = 'Instacart'
    run_name = '- high_corr custom refcv, 95 features + order interval (6) + trend in purchase interval(2) + order interval readiness(3) , frac 0.4'

    data_folder = 'data'
    sample_frac = 0.6
    test_size = 0.2
    scaling = False
    data_full_features = pd.read_pickle('{}/train_full_features.pickle'.format(data_folder))
    print('train set BEFORE sampling:')
    print(data_full_features.shape)
    # use part of the data for speed and memory
    data_full_features = data_full_features.sample(frac=sample_frac, random_state=0).reset_index(drop=True)
    print('train set AFTER sampling:')
    print(data_full_features.shape)
    X_train, X_val, y_train, y_val = split_data(data_full_features, test_size=test_size, data_folder=data_folder,
                                                split_by='user_id')

    # release memory
    gc.collect; del data_full_features

    print(y_train.value_counts(dropna=False))
    print(y_val.value_counts(dropna=False))

    print(X_train.shape)

    if scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        print('saving scaler')
        joblib.dump(scaler, '{}/xgb_scaler.joblib'.format(data_folder))

        # adding feature names back for better feature importance visualization
        X_train = pd.DataFrame(X_train, columns=X_train.columns)
        X_val = pd.DataFrame(X_val, columns=X_val.columns)

    X_train.to_pickle('{}/X_train.pickle'.format(data_folder))
    y_train.to_pickle('{}/y_train.pickle'.format(data_folder))
    X_val.to_pickle('{}/X_val.pickle'.format(data_folder))
    y_val.to_pickle('{}/y_val.pickle'.format(data_folder))

    drop_cols = ['order_id', 'user_id', 'product_id']
    X_train = X_train.drop(columns=drop_cols)
    X_val = X_val.drop(columns=drop_cols)

    # feature selection
    X_train = custom_refcv_drop(X_train)
    X_val = custom_refcv_drop(X_val)

    X_train = custom_refcv_drop_2(X_train)
    X_val = custom_refcv_drop_2(X_val)

    print ('X_train Shape: ', X_train.shape)
    assert X_train.columns.nunique() == 118

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
        , "early_stopping_rounds": 100
        , "random_state": 19
        , "predictor": 'cpu_predictor'
    }
    print(4)
    bst = xgboost.XGBClassifier(**xgb_params)
    bst.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)

    # access performance
    num_best_ntrees = bst.best_ntree_limit
    res = bst.evals_result()
    train_logloss = res["validation_0"]["logloss"][-1]
    val_logloss = res["validation_1"]["logloss"][-1]

    train_auc = res["validation_0"]["auc"][-1]
    val_auc = res["validation_1"]["auc"][-1]

    print('saving model')
    bst.save_model("{}/xgb_model.json".format(data_folder))

    # feature importance
    feature_importance = pd.DataFrame({'features': X_train.columns.astype('str'),
                                       'importance': bst.feature_importances_})
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    feature_importance.to_csv('{}/up_feature_importance.csv'.format(data_folder), index=False)

    # plot feature importance
    fig = plt.figure(figsize=(25, 20))
    plt.barh(feature_importance['features'], feature_importance['importance'])
    plt.tight_layout()


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
        mlflow.log_params({'num_features': X_train.shape[1]})
        mlflow.log_params({'sample_frac': sample_frac, 'test_size': test_size})
        mlflow.log_metrics({'train_logloss': train_logloss, 'val_logloss': val_logloss, 'train_auc': train_auc,
                            'val_auc': val_auc, 'duration_mins': time_spent})
        mlflow.log_figure(fig, "feature importance.png")
        mlflow.xgboost.log_model(bst, 'bst_test')
        # logging this file is easy to out of memory
        # mlflow.log_artifact('{}/train_full_features.pickle'.format(data_folder))
        mlflow.log_artifact('{}/up_feature_importance.csv'.format(data_folder))
