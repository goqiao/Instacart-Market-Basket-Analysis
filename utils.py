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


def group_existing_features(top_percent=0.4, mid_percent=0.3):
    fi = pd.read_csv('data/up_feature_importance.csv')
    num_features = fi.shape[0]

    top_percent = top_percent
    mid_percent = mid_percent

    def fi_group(i):
        if i < num_features * top_percent:
            return 'high_fi'
        elif i < num_features * (mid_percent + top_percent):
            return 'middle_fi'
        else:
            return 'low_fi'

    fi['importance_group'] = fi.reset_index()['index'].apply(lambda x: fi_group(x))
    fi['feature_group'] = fi['features'].str.extract(r'([a-z]+)_.*')

    # TODO: users and aisle can be removed after rerun
    fi['feature_group'] = fi['feature_group'].str.replace('users', 'user', regex=False)
    fi['feature_group'] = fi['feature_group'].str.replace('uo', 'user', regex=False)
    fi['feature_group'] = fi['feature_group'].str.replace('aisle', 'user', regex=False)

    print('Feature Groups:')
    print(fi['feature_group'].value_counts(dropna=False))

    groups = fi[['features', 'feature_group', 'importance_group']].groupby(['feature_group', 'importance_group'])[
        'features'].apply(lambda x: x.to_list()).reset_index()
    groups['fi_group'] = groups['feature_group'] + '_' + groups['importance_group']
    print(groups[['fi_group', 'features']])
    groups[['fi_group', 'features']].to_pickle('data/fi_group.pickle')

def keep_top_features(df, keep_high_mid_fi=True, keep_high_fi=False):
    # TODO: this function need to be hard coded after finalizing code. Other people won't have the fi csv file in the first run
    # fi_group = group_existing_features(top_percent=0.4, mid_percent=0.3).set_index('fi_group')
    # fi_group = pd.read_pickle('data/fi_group.pickle').set_index('fi_group')
    # print(fi_group)
    p_high = ['p_ratio_2nd_to_onetime_purchases', 'p_reorder_rate', 'p_num_purchases_per_user_q80', 'p_num_purchases_per_user_median',
              'p_purchase_interval_days_q80', 'p_num_purchases_per_user_mean', 'p_num_purchases_per_user_std',
              'p_purchase_interval_days_mean', 'p_sum_reordered', 'p_num_purchases', 'p_avg_first_reorder_diff',
              'p_mean_add_cart_num', 'p_sum_secondtime_purchase', 'p_purchase_interval_days_max_woo', 'p_purchase_interval_days_std',
              'p_purchase_interval_days_min_woo', 'p_purchase_interval_days_median', 'p_unique_buyers', 'p_purchase_interval_days_q20',
              'p_sum_onetime_purchase', 'p_avg_first_reorder_num']
    p_middle = ['p_std_add_cart_num', 'p_is_organic', 'p_std_first_order_num', 'p_std_first_reorder_diff', 'p_avg_first_order_num', 'p_num_purchases_per_user_max']
    p_low = ['p_std_first_reorder_num', 'p_num_purchases_per_user_q20', 'p_is_gluten_free', 'p_is_asian', 'p_num_purchases_per_user_min']

    product_embedding_middle = ['product_embedding_2', 'product_embedding_33', 'product_embedding_30', 'product_embedding_34',
                                'product_embedding_26', 'product_embedding_3']
    product_embedding_low = ['product_embedding_24', 'product_embedding_32', 'product_embedding_9', 'product_embedding_28',
                             'product_embedding_20', 'product_embedding_1', 'product_embedding_14', 'product_embedding_0',
                             'product_embedding_31', 'product_embedding_15', 'product_embedding_27', 'product_embedding_6',
                             'product_embedding_22','product_embedding_23', 'product_embedding_8', 'product_embedding_16',
                             'product_embedding_13',  'product_embedding_12','product_embedding_10', 'product_embedding_5',
                             'product_embedding_11', 'product_embedding_18', 'product_embedding_17', 'product_embedding_29',
                             'product_embedding_7', 'product_embedding_19', 'product_embedding_35', 'product_embedding_21',
                             'product_embedding_4', 'product_embedding_25']

    user_high = ['user_days_not_purchase', 'user_reorder_rate', 'user_max_order', 'user_total_orders', 'user_next_order_readiness',
                 'uo_reorder_ratio_mean', 'uo_reorered_products_mean', 'user_age_days_on_platform', 'user_product_unique',
                 'user_reordered_products_per_order', 'uo_reordered_products_std', 'user_mean_days_order_interval',
                 'user_order_freq_days_mean', 'users_organic_ratio', 'user_aisle_unique', 'user_cluster_2', 'uo_unique_aisle_mean', 'user_cluster_1']

    user_middle = ['users_asian_food_ratio', 'user_product_total', 'uo_unique_aisle_std', 'uo_reorder_ratio_std',
                   'users_organic_ratio_r5', 'uo_unique_department_mean', 'user_std_days_order_interval',
                   'users_purchases_dow_2', 'user_reorder_prod_total', 'user_basket_size_trend_d1',
                   'user_orders_days_interval_trend', 'uo_basket_size_mean', 'uo_unique_department_std',
                   'users_asian_food_ratio_r5', 'users_norm_purchases_pod_midnight', 'uo_basket_size_std',
                   'users_purchases_pod_night', 'users_gluten_free_ratio_r5', 'user_std_order_dow',
                   'user_basket_size_skew', 'users_norm_purchases_dow_6', 'user_department_unique',
                   'users_norm_purchases_dow_0', 'users_purchases_pod_midnight', 'users_purchases_pod_morning',
                   'users_norm_purchases_pod_morning', 'users_norm_purchases_dow_2', 'users_purchases_dow_3',
                   'users_purchases_pod_noon', 'user_mean_order_hour', 'users_norm_purchases_pod_night',
                   'users_purchases_dow_5', 'users_purchases_dow_0', 'users_purchases_dow_6', 'user_std_order_hour', 'users_purchases_dow_1']
    user_low = ['users_norm_purchases_pod_noon', 'users_gluten_free_ratio', 'users_norm_purchases_dow_1',
                'user_cluster_0', 'users_norm_purchases_dow_3', 'users_norm_purchases_dow_5', 'user_mean_order_dow',
                'users_norm_purchases_dow_4', 'users_purchases_dow_4', 'user_cluster_3']


    up_high = ['up_num_purchases_r5', 'up_purchase_proba_r5', 'up_purchase_ratio_r5', 'up_purchase_proba', 'up_num_orders_since_last_purchase',
               'up_reorder_times', 'up_num_purchases_diff_p_min', 'up_purchase_interval_days_max_r5', 'up_num_purchases_diff_p_q20',
               'up_overdue_days_mean', 'up_num_purchases', 'up_num_purchases_diff_p_q80', 'up_overdue_days_max', 'up_readiness_p_min_woo',
               'up_purchase_interval_days_min', 'up_first_order', 'up_readiness_p_q20', 'up_readiness_p_median', 'up_aisle_purchase_skew',
               'up_num_days_not_purchase', 'up_num_purchases_diff_p_mean', 'up_readiness_p_q80', 'up_overdue_days_min', 'up_std_order_num',
               'up_readiness_p_max_woo', 'up_readiness_p_range', 'up_cart_order_median', 'up_purchase_interval_days_median',
               'up_purchase_interval_days_mean', 'up_purchase_interval_days_mean_r5', 'up_purchase_readiness', 'up_purchase_interval_days_min_r5',
               'up_mean_order_num', 'up_cart_order_mean', 'up_purchase_interval_days_max', 'up_readiness_p_mean', 'up_cart_order_min',
               'up_purchases_dow_3', 'up_purchase_interval_days_median_r5', 'up_norm_purchases_pod_morning', 'up_cart_order_sum', 'up_purchases_dow_5',
               'up_subsitute_num_purchases_r5', ]
    up_middle = ['up_purchases_dow_6', 'up_department_purchase_skew', 'up_aisle_purchase_trend_d1', 'up_last_order',
                 'up_cart_order_max', 'up_organic_substitute_num_purchases', 'up_purchases_pod_noon', 'up_purchases_pod_night',
                 'up_num_purchases_diff_p_max', 'up_cart_order_std', 'up_purchases_pod_morning', 'up_purchases_dow_4', 'up_substitute_num_purchases']
    up_low = ['up_norm_purchases_dow_2', 'up_norm_purchases_dow_0', 'up_purchases_dow_2', 'up_norm_purchases_dow_4',
              'up_norm_purchases_pod_night', 'up_department_purchase_trend_d1', 'up_norm_purchases_pod_noon', 'up_norm_purchases_dow_3',
              'up_norm_purchases_dow_5', 'up_norm_purchases_dow_6', 'up_purchases_dow_0', 'up_purchases_pod_midnight', 'up_norm_purchases_pod_midnight',
              'up_purchases_dow_1', 'up_norm_purchases_dow_1', 'up_organic_substitute_num_purchases_r5']


    if keep_high_mid_fi:
        # assert len(p_high + p_middle + user_high + user_middle + up_high + up_middle + p_low + user_low + up_low) == 165
        to_keep = p_high + p_middle + user_high + user_middle + up_high + up_middle + product_embedding_middle
    elif keep_high_fi:
        to_keep = p_high + user_high + up_high + product_embedding_middle
    return df[to_keep]


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







