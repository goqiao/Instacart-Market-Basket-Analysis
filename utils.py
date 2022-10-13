from sklearn.metrics import (
    f1_score,
    log_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def print_eval_metrics(y_true, y_pred, y_pred_prob=None):
    """
    print classification evaluation metrics
    :return: None
    """

    print("f1:", f1_score(y_true, y_pred))
    print("precision:", precision_score(y_true, y_pred))
    print("recall:", recall_score(y_true, y_pred))
    print("accuracy", accuracy_score(y_true, y_pred))
    if y_pred_prob:
        print("log loss:", log_loss(y_true, y_pred_prob))
        print("roc auc:", roc_auc_score(y_true, y_pred_prob))


def read_data(
    nrows=None,
    data_folder="data",
    read_products=False,
    read_orders=False,
    read_prior=False,
    read_train=False,
    read_test=False,
):
    if read_products:
        aisles = pd.read_csv("{}/aisles.csv".format(data_folder))
        departments = pd.read_csv("{}/departments.csv".format(data_folder))
        products = pd.read_csv("{}/products.csv".format(data_folder))

        # merge products with departments and aisles data
        products = products.merge(departments, how="left").merge(aisles, how="left")

        return products

    if read_orders:
        orders = pd.read_csv(
            "{}/orders.csv".format(data_folder),
            dtype={
                "order_id": np.int32,
                "user_id": np.int64,
                "eval_set": "category",
                "order_number": np.int16,
                "order_dow": np.int8,
                "order_hour_of_day": np.int8,
                "days_since_prior_order": np.float32,
            },
            nrows=nrows,
        )
        return orders

    if read_prior:
        prior = pd.read_csv(
            "{}/order_products__prior.csv".format(data_folder),
            dtype={
                "order_id": np.int32,
                "product_id": np.uint16,
                "add_to_cart_order": np.int16,
                "reordered": np.int8,
            },
            nrows=nrows,
        )
        return prior

    if read_train:
        train = pd.read_csv(
            "{}/order_products__train.csv".format(data_folder),
            dtype={
                "order_id": np.int32,
                "product_id": np.uint16,
                "add_to_cart_order": np.int16,
                "reordered": np.int8,
            },
            nrows=nrows,
        )

        return train

    if read_test:
        test = pd.read_csv(
            "{}/order_products__test.csv".format(data_folder),
            dtype={
                "order_id": np.int32,
                "product_id": np.uint16,
                "add_to_cart_order": np.int16,
                "reordered": np.int8,
            },
            nrows=nrows,
        )

        return test


def improve_data_type(data):
    import sys

    starting_size = sys.getsizeof(data)
    i = 0
    for c, dtype in zip(data.columns, data.dtypes):
        if "int" in str(dtype):
            print(c, dtype)
            if min(data[c]) >= 0:
                max_int = max(data[c])
                if max_int <= 255:
                    data[c] = data[c].astype(np.uint8)
                elif max_int <= 65535:
                    data[c] = data[c].astype(np.uint16)
                elif max_int <= 4294967295:
                    data[c] = data[c].astype(np.uint32)
                i += 1
    print("Number of columns adjusted: {}\n".format(i))
    ## Changing known reordered col to smaller int size
    if "reordered" in data.columns:
        data["reordered"] = np.nan_to_num(data["reordered"]).astype(np.uint8)

    # data['reordered'][data['reordered'] == 0] = np.nan
    print(
        "Reduced size {:.2%}".format(float(sys.getsizeof(data)) / float(starting_size))
    )
    return data


def split_data(data_full_features, test_size, data_folder="data", split_by=None):
    from sklearn.model_selection import train_test_split

    if split_by:
        unique_dimension = data_full_features[split_by].drop_duplicates()
        selected_test = unique_dimension.sample(frac=test_size, random_state=0)
        train, test = (
            data_full_features.loc[~data_full_features[split_by].isin(selected_test)],
            data_full_features.loc[data_full_features[split_by].isin(selected_test)],
        )
        X_train, y_train, X_test, y_test = (
            train.drop("reordered", axis=1),
            train["reordered"],
            test.drop("reordered", axis=1),
            test["reordered"],
        )
    else:
        y = data_full_features["reordered"]
        X = data_full_features.drop("reordered", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0
        )

    return X_train, X_test, y_train, y_test


def plot_precision_recall_curve(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="recall")
    plt.title("precision_recall_curve")
    plt.show()


def save_fig_with_timestamp(
    file_name, fig_extension="png", resolution=300, folder="data"
):
    current_timestamp = time.strftime("%Y-%m-%d--%H-%M", time.localtime())
    path = os.path.join(
        folder, file_name + "--" + current_timestamp + "." + fig_extension
    )
    plt.savefig(path, format=fig_extension, dpi=resolution)


def p20(x):
    return x.quantile(0.2)


# 90th Percentile
def p80(x):
    return x.quantile(0.8)


def max_no_outliers(x):
    q3, q1 = x.quantile(0.75), x.quantile(0.25)
    return q3 + 1.5 * (q3 - q1)


def min_no_outliers(x):
    q3, q1 = x.quantile(0.75), x.quantile(0.25)
    return np.max([0, q1 - 1.5 * (q3 - q1)])


def trend_d1(y):
    from numpy import polyfit

    x = np.arange(1, len(y) + 1)
    if len(y) == 1:
        return np.nan
    # the coefficient
    else:
        return polyfit(x, y, 1)[0]


def skewness(y):
    from scipy.stats import skew

    if len(y) == 1:
        return np.nan
    else:
        return skew(y, axis=0, bias=False)


def compress(df, key):
    """
    compress df on key level
    """
    cols = [col for col in df.columns if "_id" not in col]
    grp = df.groupby(key)
    _df = df[[key]].drop_duplicates().set_index(key)
    for c in cols:
        tmp = grp.agg({c: ["mean", "median", "min", "max", "std"]})
        tmp.columns = [c + "-mean", c + "-median", c + "-min", c + "-max", c + "-std"]
        _df = pd.concat([_df, tmp], axis=1)

    # drop cols with 0 variance
    col_var = _df.var()
    col = col_var[col_var == 0].index
    _df.drop(col, axis=1, inplace=True)
    return _df.reset_index()


def download_user_order_history(uid, pid, label=""):
    from utils import read_data

    df = pd.read_pickle("data/prior_train_details.pickle")
    df = df.loc[df.user_id == uid]
    products = pd.read_pickle("data/products.pickle")
    df = df.merge(
        products[["product_id", "product_name", "aisle", "department"]], how="left"
    )
    df.to_csv(
        "data/{}_user_{}_order_products_history.csv".format(label, uid), index=False
    )
    if pid:
        product_name = products.loc[
            products["product_id"] == pid, "product_name"
        ].values[0]
        print(uid, product_name, " purchase history")
    df = read_data(read_orders=True)
    df = df.loc[df.user_id == uid]
    df.to_csv("data/{}_user_{}_order_history.csv".format(label, uid), index=False)

    print("user {} order history downloaded in data folder".format(uid))


def locate_by_uid_pid(df, uid=None, pid=None):
    if uid and pid:
        res = df.loc[(df.user_id == uid) & (df.product_id == pid)]
        print(res)
    if uid and not pid:
        res = df.loc[df.user_id == uid]
        print(res)


def feature_selection(df):
    # threshold corr >= 0.98
    high_corr_features = [
        "up_cart_order_median",
        "up_reorder_times",
        "up_purchase_ratio_r5",
        "up_purchase_interval_days_median",
        "up_order_interval_skewness_r5",
        "user_reorder_prod_total",
        "user_reordered_products_per_order",
        "p_sum_reordered",
        "p_sum_secondtime_purchase",
        "p_purchase_interval_days_max_woo",
        "up_overdue_days_diff_median_vs_others",
        "up_overdue_days_diff_p20_vs_others",
        "up_overdue_days_diff_p80_vs_others",
        "up_overdue_days_diff_min_vs_others",
        "up_overdue_orders_diff_mean_vs_others",
        "up_overdue_orders_diff_median_vs_others",
        "up_overdue_orders_diff_min_vs_others",
        "up_overdue_orders_percent_diff_range_vs_others",
    ]

    rfe_dropped = [
        "p_num_purchases_per_user_min",
        "up_purchases_pod_midnight",
        "p_num_purchases_per_user_p20",
        "p_is_gluten_free",
        "p_is_asian",
        "up_organic_substitute_num_purchases_r5",
        "p_is_organic",
        "p_num_products_sim_score_0.9_1.0",
        "up_norm_purchases_pod_midnight",
        "p_num_purchases_per_user_median",
        "up_organic_substitute_num_purchases",
        "up_purchases_dow_5",
        "up_purchases_dow_1",
        "p_purchase_interval_days_min_woo",
        "up_word2vec_substitute_num_purchases_r5",
        "users_purchases_dow_4",
        "up_purchases_dow_2",
        "users_asian_food_ratio",
        "up_purchases_dow_4",
        "up_purchases_pod_night",
        "up_norm_purchases_dow_1",
        "users_norm_purchases_dow_6",
        "up_purchases_dow_3",
        "up_purchases_pod_morning",
        "users_purchases_dow_5",
        "p_embedding_23",
        "users_asian_food_ratio_r5",
        "users_norm_purchases_pod_night",
        "up_norm_purchases_pod_morning",
        "users_norm_purchases_dow_5",
        "users_norm_purchases_dow_0",
        "up_norm_purchases_dow_6",
        "up_norm_purchases_dow_4",
        "up_purchases_pod_noon",
        "users_norm_purchases_pod_morning",
        "p_num_products_sim_score_0.8_0.9",
        "users_norm_purchases_dow_2",
        "p_embedding_38",
        "p_embedding_24",
        "p_num_products_sim_score_0.7_0.8",
        "user_product_total",
        "up_norm_purchases_dow_3",
        "up_norm_purchases_dow_2",
        "users_purchases_dow_0",
        "users_purchases_dow_2",
        "users_purchases_pod_night",
        "users_gluten_free_ratio_r5",
        "users_norm_purchases_dow_1",
        "p_embedding_7",
        "users_purchases_pod_midnight",
        "up_norm_purchases_dow_0",
        "p_embedding_30",
        "p_embedding_34",
        "p_embedding_32",
        "users_norm_purchases_pod_noon",
        "users_norm_purchases_dow_4",
        "up_norm_purchases_pod_noon",
        "p_embedding_10",
        "p_num_products_sim_score_0.6_0.7",
        "up_word2vec_substitute_num_purchases",
        "users_norm_purchases_dow_3",
        "users_norm_purchases_pod_midnight",
        "p_max_similarity_score",
        "p_embedding_15",
        "p_embedding_1",
        "p_embedding_29",
        "user_department_unique",
        "up_norm_purchases_dow_5",
        "users_purchases_dow_1",
        "p_embedding_4",
        "p_embedding_26",
        "user_std_order_dow",
        "p_embedding_11",
        "user_std_order_hour",
        "users_purchases_pod_noon",
        "user_mean_order_dow",
        "users_gluten_free_ratio",
        "p_embedding_37",
        "p_embedding_19",
        "users_purchases_dow_3",
        "p_embedding_33",
        "user_mean_order_hour",
        "users_purchases_dow_6",
        "user_basket_size_skew",
        "p_embedding_16",
        "p_embedding_28",
        "up_cart_order_std",
        "p_embedding_18",
        "p_embedding_5",
        "p_embedding_35",
        "p_embedding_39",
        "p_embedding_2",
        "p_embedding_36",
        "p_embedding_27",
        "uo_unique_aisle_std",
        "p_avg_first_order_num",
        "p_embedding_31",
    ]
    df.drop(high_corr_features, axis=1, inplace=True)
    df.drop(rfe_dropped, axis=1, inplace=True)
    return df
