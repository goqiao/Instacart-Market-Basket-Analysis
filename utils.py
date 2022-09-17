from sklearn.metrics import f1_score, log_loss, roc_auc_score, precision_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import precision_recall_curve


def print_eval_metrics(y_true, y_pred, y_pred_prob=None):
    """
    print classification evaluation metrics
    :return: None
    """

    print('f1:', f1_score(y_true, y_pred))
    print('precision:', precision_score(y_true, y_pred))
    print('recall:', recall_score(y_true, y_pred))
    print('accuracy', accuracy_score(y_true, y_pred))
    if y_pred_prob:
        print('log loss:', log_loss(y_true, y_pred_prob))
        print('roc auc:', roc_auc_score(y_true, y_pred_prob))


def read_data(nrows=None, data_folder='data', read_products=False, read_orders=False, read_prior=False,
              read_train=False, read_test=False):
    if read_products:
        aisles = pd.read_csv('{}/aisles.csv'.format(data_folder))
        departments = pd.read_csv('{}/departments.csv'.format(data_folder))
        products = pd.read_csv('{}/products.csv'.format(data_folder))

        # merge products with departments and aisles data
        products = products.merge(departments, how='left').merge(aisles, how='left')

        return products

    if read_orders:
        orders = pd.read_csv('{}/orders.csv'.format(data_folder), dtype={
            'order_id': np.int32,
            'user_id': np.int64,
            'eval_set': 'category',
            'order_number': np.int16,
            'order_dow': np.int8,
            'order_hour_of_day': np.int8,
            'days_since_prior_order': np.float32}, nrows=nrows)
        return orders

    if read_prior:
        prior = pd.read_csv('{}/order_products__prior.csv'.format(data_folder), dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, nrows=nrows)
        return prior

    if read_train:
        train = pd.read_csv('{}/order_products__train.csv'.format(data_folder), dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, nrows=nrows)

        return train

    if read_test:
        test = pd.read_csv('{}/order_products__test.csv'.format(data_folder), dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8}, nrows=nrows)

        return test


def improve_data_type(data):
    import sys
    starting_size = sys.getsizeof(data)
    i = 0
    for c, dtype in zip(data.columns, data.dtypes):
        if 'int' in str(dtype):
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
    print("Number of colums adjusted: {}\n".format(i))
    ## Changing known reorderd col to smaller int size
    if 'reordered' in data.columns:
        data['reordered'] = np.nan_to_num(data['reordered']).astype(np.uint8)

    # data['reordered'][data['reordered'] == 0] = np.nan
    print("Reduced size {:.2%}".format(float(sys.getsizeof(data)) / float(starting_size)))
    return data


def split_data(data_full_features, test_size, data_folder='data', split_by=None):
    from sklearn.model_selection import train_test_split

    if split_by:
        unique_dimension = data_full_features[split_by].drop_duplicates()
        selected_test = unique_dimension.sample(frac=test_size, random_state=0)
        train, test = data_full_features.loc[~data_full_features[split_by].isin(selected_test)],\
                      data_full_features.loc[data_full_features[split_by].isin(selected_test)]
        X_train, y_train, X_test, y_test = train.drop('reordered', axis=1), train['reordered'], \
                                           test.drop('reordered', axis=1), test['reordered']
    else:
        y = data_full_features['reordered']
        X = data_full_features.drop('reordered', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)


    return X_train, X_test, y_train, y_test
    # X_train.to_pickle('{}/X_train.pickle'.format(data_folder))
    # X_test.to_pickle('{}/X_test.pickle'.format(data_folder))
    # y_train.to_pickle('{}/y_train.pickle'.format(data_folder))
    # y_test.to_pickle('{}/y_test.pickle'.format(data_folder))


def plot_precision_recall_curve(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label='recall')
    plt.title('precision_recall_curve')
    plt.show()


def save_fig_with_timestamp(file_name, fig_extension="png", resolution=300, folder='data'):
    current_timestamp = time.strftime('%Y-%m-%d--%H-%M', time.localtime())
    path = os.path.join(folder, file_name + '--' + current_timestamp + '.' + fig_extension)
    plt.savefig(path, format=fig_extension, dpi=resolution)


def q20(x):
    return x.quantile(0.2)

# 90th Percentile
def q80(x):
    return x.quantile(0.8)


def max_no_outliers(x):
    q3, q1= x.quantile(0.75), x.quantile(0.25)
    return q3 + 1.5*(q3- q1)

def min_no_outliers(x):
    q3, q1= x.quantile(0.75), x.quantile(0.25)
    return np.max([0, q1 - 1.5*(q3- q1)])


def trend_d1(y):
    from numpy import polyfit
    x = np.arange(1, len(y)+1)
    # the coefficient
    return polyfit(x, y, 1)[0]

def skew(y):
    from scipy.stats import skew
    return skew(y, axis=0, bias=False)

def compress(df, key):
    """
    compress df on key level
    """
    cols = [col for col in df.columns if '_id' not in col]
    grp = df.groupby(key)
    _df = df[[key]].drop_duplicates().set_index(key)
    for c in cols:
        # TODO: improve performance
        # _df[c + '-mean'] = grp[c].mean()
        # _df[c + '-median'] = grp[c].median()
        # _df[c + '-min'] = grp[c].min()
        # _df[c + '-max'] = grp[c].max()
        # _df[c + '-std'] = grp[c].std()
        tmp = grp.agg({c:['mean', 'median', 'min', 'max', 'std']})
        tmp.columns = [c + '-mean', c + '-median', c + '-min', c + '-max', c + '-std']
        _df = pd.concat([_df, tmp], axis=1)

    # drop cols with 0 variance
    col_var = _df.var()
    col = col_var[col_var == 0].index
    _df.drop(col, axis=1, inplace=True)
    return _df.reset_index()


def keep_top_features(df):
    users_high_im = ['user_age_days_on_platform', 'user_total_orders', 'user_mean_days_order_interval', 'user_std_days_order_interval',
                     'user_reorder_ratio', 'user_reorder_rate', 'user_next_order_readiness', 'user_order_freq']
    users_med_im = ['user_product_total', 'user_product_unique', 'user_days_not_purchase', 'uo_basket_size_mean', 'uo_unique_aisle_mean',
                    'uo_unique_department_mean', 'uo_reorered_products_mean', 'uo_reordered_products_std', 'uo_reorder_ratio_mean',
                    ]
    users_low_im = ['user_mean_order_dow', 'user_std_order_dow', 'user_mean_order_hour', 'user_std_order_hour',
                    'aisle_nunique', 'user_department_unique', 'user_reorder_prod_total', 'user_order_num_sum_exclude_1st',
                    'uo_basket_size_std', 'uo_unique_aisle_std', 'uo_unique_department_std', 'uo_reorder_ratio_std', ]

    products_high_im = ['p_num_purchases_per_user_q80', 'p_num_purchases_per_user_median', 'p_reorder_rate',
                        'p_reorder_proba', 'p_num_purchases_per_user_mean', 'p_purchase_interval_days_q80']

    products_med_im = ['p_sum_reordered', 'p_num_purchases', 'p_unique_buyers', 'p_mean_add_cart_num', 'p_sum_secondtime_purchase',
                       'p_sum_onetime_purchase', 'p_avg_first_reorder_diff', 'p_std_first_reorder_diff', 'p_num_purchases_per_user_std',
                       'p_purchase_interval_days_mean', 'p_purchase_interval_days_median', 'p_purchase_interval_days_max_woo']

    products_low_im = ['p_std_add_cart_num', 'p_avg_first_order_num', 'p_std_first_order_num', 'p_avg_first_reorder_num',
                       'p_std_first_reorder_num', 'p_num_purchases_per_user_max', 'p_num_purchases_per_user_min', 'p_num_purchases_per_user_q20',
                       'p_purchase_interval_days_std', 'p_purchase_interval_days_q20', 'p_purchase_interval_days_min_woo']

    up_high_im = ['up_purchase_time', 'up_reorder_times', 'up_purchase_times_r5', 'up_purchase_ratio_r5',
                  'up_purchase_interval_days_mean_r5', 'up_purchase_interval_days_median_r5', 'up_days_since_last_purchase_max_r5',
                  'up_days_since_last_purchase_min_r5', 'up_purhcase_proba', 'up_purhcase_proba_r5', 'up_num_days_not_purchase',
                  ]
    up_med_im = ['up_cart_order_mean', 'up_cart_order_min', 'up_cart_order_median', 'up_first_order', 'up_last_order',
                 'up_mean_order_num', 'up_std_order_num', 'up_purchase_interval_days_mean', 'up_purchase_interval_days_median',
                 'up_purchase_interval_days_max', 'up_purchase_interval_days_min']
    up_low_im = ['up_cart_order_std', 'up_cart_order_sum', 'up_cart_order_max', ]

    cols = users_high_im + users_med_im + users_low_im + products_high_im + products_med_im + products_low_im \
           + up_high_im + up_med_im + up_low_im


    return df[cols]


def download_user_order_history(uid, pid, label=''):
    from utils import read_data

    df = pd.read_pickle('data/prior_train_details.pickle')
    df = df.loc[df.user_id == uid]
    products = pd.read_pickle('data/products.pickle')
    df = df.merge(products[['product_id', 'product_name', 'aisle', 'department']], how='left')
    df.to_csv('data/{}_user_{}_order_products_history.csv'.format(label, uid), index=False)
    if pid:
        product_name = products.loc[products['product_id']==pid, 'product_name'].values[0]
        print(uid, product_name, ' purchase history')
    df = read_data(read_orders=True)
    df = df.loc[df.user_id == uid]
    df.to_csv('data/{}_user_{}_order_history.csv'.format(label, uid), index=False)

    print('user {} order history downloaded in data folder'.format(uid))

def locate_by_uid_pid(df, uid=None, pid=None):
    if uid and pid:
        res = df.loc[(df.user_id==uid) & (df.product_id==pid)]
        print(res)
    if uid and not pid:
        res = df.loc[df.user_id==uid]
        print(res)







