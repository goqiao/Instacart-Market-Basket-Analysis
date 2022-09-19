import xgboost
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import split_data, print_eval_metrics, save_fig_with_timestamp
import matplotlib.pyplot as plt
import numpy as np
import time
import mlflow
import xgboost
from sklearn.preprocessing import StandardScaler
from utils import split_data, print_eval_metrics, save_fig_with_timestamp
import os
from sklearn.model_selection import GroupKFold, cross_val_score, cross_validate

start_time = time.time()
if __name__ == '__main__':
    # orders = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]])
    # y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    # userid = np.array([0, 0, 0, 2, 2, 3, 3, 4, 4])

    # split based on userid in the sense that split is made on user level and orders(rows) from the same user will be
    # split in the same fold
    # gkf = GroupKFold(n_splits=3)
    # cnt = 0
    # for train_idx, test_idx in gkf.split(orders, y, userid):
    #     cnt += 1
    #     print('{} fold'.format(cnt), '\n')
    #     print('train_idx:', train_idx, '\n')
    #     print('test_idx:', test_idx, '\n')
    #     print('train_users:', userid[train_idx], '\n')
    #     print('test_users:', userid[test_idx], '\n')
    #     print('train_x:', orders[train_idx], '\n')
    #     print('test_x:', orders[test_idx], '\n')
    #     print('train_y:', y[train_idx], '\n')
    #     print('test_y:', y[test_idx], '\n')

    # print(gkf.get_n_splits(orders, y, userid))
    # 4 users in total
    # Splitting to 2 groups by userid. 2 users are in the 1st group(train) while 2 users are in the second fold(test)
    # repeat this repeating by 3 times (3 folds)

    from sklearn import datasets, linear_model
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import LinearSVC

    # diabetes = datasets.load_diabetes()
    # X = diabetes.data[:150]
    # y = diabetes.target[:150]
    # lasso = linear_model.Lasso()
    # cv_results = cross_validate(lasso, X, y, cv=3, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
    # print(cv_results.keys())
    # print(cv_results['train_r2'].mean())
    # print(cv_results['test_r2'].mean())
    # print(cv_results['train_neg_mean_squared_error'].mean())
    # print(cv_results['test_neg_mean_squared_error'].mean())

    # experiment setting
    experiment_name = 'Instacart'
    run_name = 'XGB CV GroupKFold + New Data Pipeline'

    data_folder = 'data'
    sample_frac = 0.2
    test_size = 0.2
    data_full_features = pd.read_pickle('{}/train_full_features.pickle'.format(data_folder))
    print(data_full_features.shape)

    # use part of the data for speed
    data_full_features = data_full_features.sample(frac=sample_frac)

    X_train, X_val, y_train, y_val = split_data(data_full_features, test_size=test_size,
                                                data_folder=data_folder, drop_cols=['eval_set'])
    # y_train.fillna(0, inplace=True)
    # y_val.fillna(0, inplace=True)

    group_base = X_train['user_id']

    X_train.drop(columns=['user_id', 'product_id'])
    X_val.drop(columns=['user_id', 'product_id'])

    print(y_train.value_counts(dropna=False))
    print(y_val.value_counts(dropna=False))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # adding feature names back for better feature importance visualization
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    print(X_train_scaled.shape)
    # X_train_scaled.to_pickle('{}/X_train_scaled.pickle'.format(data_folder))
    # y_train.to_pickle('{}/y_train.pickle'.format(data_folder))


    xgb_params = {
        'n_estimators': 80
        , "objective": "binary:logistic"
        , "eval_metric": ['auc', 'logloss']
        , "eta": 0.1
        , "max_depth": 6
        , "min_child_weight": 10
        , "gamma": 0.70
        , "subsample": 0.76
        , "colsample_bytree": 0.95
        , "alpha": 2e-05
        , "lambda": 10
    }
    print(X_train_scaled.columns)
    gkf = GroupKFold(n_splits=5).split(X_train_scaled, y_train, group_base)

    bst = xgboost.XGBClassifier(**xgb_params)
    cv_results = cross_validate(bst, X_train_scaled, y_train, cv=gkf, scoring=['neg_log_loss', 'roc_auc'],
                                return_train_score=True)

    # access cv performance
    print(cv_results)

    train_logloss = -1*(cv_results["train_neg_log_loss"].mean())
    val_logloss = -1*(cv_results["test_neg_log_loss"].mean())

    train_auc = cv_results["train_roc_auc"].mean()
    val_auc = cv_results["test_roc_auc"].mean()

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
        mlflow.log_params({'sample_frac': sample_frac, 'test_size': test_size})
        mlflow.log_metrics({'train_logloss': train_logloss, 'val_logloss': val_logloss, 'train_auc': train_auc,
                            'val_auc': val_auc, 'duration_mins': time_spent})
        # mlflow.log_figure(fig, "feature importance.png")

        # need to call fit first
        # mlflow.xgboost.log_model(bst, 'bst_test')

