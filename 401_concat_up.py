import pandas as pd
import numpy as np
import gc
from utils import improve_data_type
import time

pd.set_option('display.max_rows', 90)

# takes 15 mins
start_time = time.time()
data_folder = 'data'

def make_data(data_folder='data', make_set='train'):
    idx_cols = ['user_id', 'product_id']
    base = pd.read_pickle('data/base.pickle').set_index(idx_cols)

    # read pre-created features
    up_agg = pd.read_pickle('data/up_agg.pickle').set_index(idx_cols)
    up_purchase_r5 = pd.read_pickle('data/up_purchase_r5.pickle').set_index(idx_cols)
    up_days_since_last_purchase = pd.read_pickle('data/up_days_since_last_purchase.pickle').set_index(idx_cols)
    up_days_since_last_purchase_r5 = pd.read_pickle('data/up_days_since_last_purchase_r5.pickle').set_index(idx_cols)
    up_purchase_proba = pd.read_pickle('data/up_purchase_proba.pickle').set_index(idx_cols)
    up_purchase_proba_r5 = pd.read_pickle('data/up_purchase_proba_r5.pickle').set_index(idx_cols)
    up_days_not_purchase = pd.read_pickle('data/up_days_not_purchase.pickle').set_index(idx_cols)
    up_aisles_purchase_trend = pd.read_pickle('data/up_aisles_purchase_trend.pickle').set_index(idx_cols)
    up_departments_purchase_trend = pd.read_pickle('data/up_departments_purchase_trend.pickle').set_index(idx_cols)
    # up_substitute_purchase = pd.read_pickle('data/up_substitute_purchase.pickle').set_index(idx_cols)
    up_organic_substitute_purchase = pd.read_pickle('data/up_organic_substitute_purchase.pickle').set_index(idx_cols)
    up_order_time = pd.read_pickle('data/up_order_time.pickle').set_index(idx_cols)
    up_word2vec_substitute_purchase = pd.read_pickle('data/up_word2vec_substitute_purchase.pickle').set_index(idx_cols)
    up_word2vec_substitute_purchase_07 = pd.read_pickle('data/up_word2vec_substitute_purchase_07.pickle').set_index(idx_cols)
    up_purchase_interval_trend = pd.read_pickle('data/up_purchase_interval_trend.pickle')
    up_order_interval = pd.read_pickle('data/up_orders_interval.pickle')

    idx_cols = ['user_id']
    users_features = pd.read_pickle('data/users_features.pickle').set_index(idx_cols)
    users_basket_size_trend = pd.read_pickle('data/users_basket_size_trend.pickle').set_index(idx_cols)
    users_orders_interval_trend = pd.read_pickle('data/users_orders_interval_trend.pickle').set_index(idx_cols)
    users_organic_purchases = pd.read_pickle('data/users_organic_purchases.pickle').set_index(idx_cols)
    users_organic_purchases_r5 = pd.read_pickle('data/users_organic_purchases_r5.pickle').set_index(idx_cols)
    users_order_time = pd.read_pickle('data/users_order_time.pickle').set_index(idx_cols)
    # users_cluster = pd.read_pickle('data/user_kmeans_cluster.pickle').set_index(idx_cols)

    idx_cols = ['product_id']
    product_features_basic_agg = pd.read_pickle('data/product_features_basic_agg.pickle').set_index(idx_cols)
    product_special_features = pd.read_pickle('data/product_special_features.pickle').set_index(idx_cols)
    products_purchases_features = pd.read_pickle('data/products_purchases_features.pickle').set_index(idx_cols)
    product_purchase_cycle = pd.read_pickle('data/product_purchase_cycle.pickle').set_index(idx_cols)
    products_embedding = pd.read_pickle('data/product_embedding.pickle').set_index(idx_cols)
    product_sub_stats = pd.read_pickle('data/product_sub_stats.pickle').set_index(idx_cols)
    product_order_interval = pd.read_pickle('data/p_order_interval.pickle')

    if make_set == 'train':
        # get train users <user_id, order_id, product_id, reordered>
        base = base.loc[base['eval_set'] == 'train']
        # base = base.sample(frac=sample_frac, random_state=0)
    elif make_set == 'test':
        # get test users <user_id, order_id, product_id, reordered>
        base = base.loc[base['eval_set'] == 'test']

    # add reordered label to the train data
    data_full_features = base.join(up_agg, on=['user_id', 'product_id'], how='left').drop('eval_set', axis=1)
    print(data_full_features.index.nunique())

    print('join1')
    # attach all features to the unique user_id and product_id tuple
    data_full_features = data_full_features.join(up_purchase_r5, on=['user_id', 'product_id'], how='left')
    del up_purchase_r5


    print('join2')
    data_full_features = data_full_features.join(up_days_since_last_purchase, on=['user_id', 'product_id'],
                                                  how='left')
    del up_days_since_last_purchase
    print('join3')
    data_full_features = data_full_features.join(up_days_since_last_purchase_r5, on=['user_id', 'product_id'],
                                                  how='left')
    del up_days_since_last_purchase_r5
    print('join4')
    data_full_features = data_full_features.join(up_purchase_proba, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.join(up_purchase_proba_r5, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.join(up_days_not_purchase, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.join(up_aisles_purchase_trend, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.join(up_departments_purchase_trend, on=['user_id', 'product_id'], how='left')
    # data_full_features = data_full_features.join(up_substitute_purchase, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.join(up_organic_substitute_purchase, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.join(up_order_time, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.join(up_word2vec_substitute_purchase, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.join(up_word2vec_substitute_purchase_07, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.join(up_purchase_interval_trend, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.join(up_order_interval, on=['user_id', 'product_id'], how='left')

    data_full_features = data_full_features.join(users_features, on='user_id', how='inner')
    data_full_features = data_full_features.join(users_basket_size_trend, on='user_id', how='inner')
    data_full_features = data_full_features.join(users_orders_interval_trend, on='user_id', how='inner')
    data_full_features = data_full_features.join(users_organic_purchases, on='user_id', how='left')
    data_full_features = data_full_features.join(users_organic_purchases_r5, on='user_id', how='left')
    data_full_features = data_full_features.join(users_order_time, on='user_id', how='left')
    # data_full_features = data_full_features.join(users_cluster, on='user_id', how='left')
    
    # release memory
    gc.collect()
    del up_purchase_proba, up_purchase_proba_r5, up_days_not_purchase, up_aisles_purchase_trend, up_departments_purchase_trend,
    up_organic_substitute_purchase, up_order_time, up_word2vec_substitute_purchase, up_word2vec_substitute_purchase_07, up_purchase_interval_trend,
    up_order_interval
    del users_features, users_basket_size_trend, users_orders_interval_trend, users_organic_purchases, users_organic_purchases_r5, users_order_time

    # optimize data types
    data_full_features = improve_data_type(data_full_features)


    print('join5')
    data_full_features = data_full_features.join(product_features_basic_agg, on='product_id', how='left')
    data_full_features = data_full_features.join(products_purchases_features, on='product_id', how='left')
    data_full_features = data_full_features.join(product_purchase_cycle, on='product_id', how='left')
    data_full_features = data_full_features.join(product_special_features, on='product_id', how='left')
    data_full_features = data_full_features.join(products_embedding, on='product_id', how='left')
    data_full_features = data_full_features.join(product_sub_stats, on='product_id', how='left')
    data_full_features = data_full_features.join(product_order_interval, on='product_id', how='left')

    print('join5 done')
    gc.collect(); del product_features_basic_agg, products_purchases_features, product_purchase_cycle, product_special_features,
    products_embedding, product_sub_stats, product_order_interval

    # create extra features
    df_temp = pd.DataFrame()
    data_full_features['num_orders_not_purchase'] = data_full_features['user_total_orders'] - data_full_features['up_last_order']

    # compared with users' own purchase interval
    df_temp['up_overdue_days_diff_mean_vs_self'] = data_full_features['up_num_days_not_purchase'] - data_full_features['up_purchase_interval_days_mean']
    df_temp['up_overdue_days_diff_max_vs_self'] = data_full_features['up_num_days_not_purchase'] - data_full_features['up_purchase_interval_days_max']
    df_temp['up_overdue_days_diff_min_vs_self'] = data_full_features['up_num_days_not_purchase'] - data_full_features['up_purchase_interval_days_min']
    df_temp['up_overdue_days_percent_range_vs_self'] = (data_full_features['up_num_days_not_purchase']/(data_full_features['up_purchase_interval_days_max'] \
                                                                    - data_full_features['up_purchase_interval_days_min'])).replace(np.inf, np.nan)

    # compared with users's own order interval
    df_temp['up_overdue_orders_diff_mean_vs_self'] = data_full_features['num_orders_not_purchase'] - data_full_features['up_mean_order_interval']
    df_temp['up_overdue_orders_diff_max_vs_self'] = data_full_features['num_orders_not_purchase'] - data_full_features['up_max_order_interval']
    df_temp['up_overdue_orders_diff_min_vs_self'] = data_full_features['num_orders_not_purchase'] - data_full_features['up_min_order_interval']


    # compared with other people's purchase interval
    df_temp['up_overdue_days_diff_median_vs_others'] = data_full_features['up_num_days_not_purchase'] - data_full_features['p_purchase_interval_days_median']
    df_temp['up_overdue_days_diff_q20_vs_others'] = data_full_features['up_num_days_not_purchase'] - data_full_features['p_purchase_interval_days_q20']
    df_temp['up_overdue_days_diff_q80_vs_others'] = data_full_features['up_num_days_not_purchase'] - data_full_features['p_purchase_interval_days_q80']
    df_temp['up_overdue_days_diff_max_vs_others'] = data_full_features['up_num_days_not_purchase'] - data_full_features['p_purchase_interval_days_max_woo']
    df_temp['up_overdue_days_diff_min_vs_others'] = data_full_features['up_num_days_not_purchase'] - data_full_features['p_purchase_interval_days_min_woo']
    df_temp['up_overdue_days_percent_range_vs_others'] = (data_full_features['up_num_days_not_purchase']/(data_full_features['p_purchase_interval_days_max_woo'] - \
                                                                                                 data_full_features['p_purchase_interval_days_min_woo'])).replace(np.inf, np.nan)
    # compared with other people's order interval
    df_temp['up_overdue_orders_diff_mean_vs_others'] = data_full_features['num_orders_not_purchase'] - data_full_features['p_order_interval_mean']
    df_temp['up_overdue_orders_diff_median_vs_others'] = data_full_features['num_orders_not_purchase'] - data_full_features['p_order_interval_median']
    df_temp['up_overdue_orders_diff_max_vs_others'] = data_full_features['num_orders_not_purchase'] - data_full_features['p_order_interval_max_woo']
    df_temp['up_overdue_orders_diff_min_vs_others'] = data_full_features['num_orders_not_purchase'] - data_full_features['p_order_interval_min_woo']
    df_temp['up_overdue_orders_percent_diff_range_vs_others'] = (data_full_features['num_orders_not_purchase'] - (data_full_features['p_order_interval_max_woo']
                                                                                                                            - data_full_features['p_order_interval_min_woo'])).replace(np.inf, np.nan)


    # compared with other people's purchase times
    df_temp['up_num_purchases_diff_p_mean'] = data_full_features['up_num_purchases']  - data_full_features['p_num_purchases_per_user_mean']
    df_temp['up_num_purchases_diff_p_max'] = data_full_features['up_num_purchases'] - data_full_features[
        'p_num_purchases_per_user_max']
    # df_temp['up_num_purchases_diff_p_min'] = data_full_features['up_num_purchases'] - data_full_features[
    #     'p_num_purchases_per_user_min']  # highly correlated with up_num_purchases_diff_p_q20
    df_temp['up_num_purchases_diff_p_q20'] = data_full_features['up_num_purchases'] - data_full_features[
        'p_num_purchases_per_user_q20']
    df_temp['up_num_purchases_diff_p_q80'] = data_full_features['up_num_purchases'] - data_full_features[
        'p_num_purchases_per_user_q80']
  
    data_full_features = pd.concat([data_full_features, df_temp], axis=1)



    # clean memory
    gc.collect()
    del df_temp
    data_full_features = improve_data_type(data_full_features.reset_index())

    # change columns type to string
    print(data_full_features.dtypes)
    data_full_features.columns = data_full_features.columns.astype('str')
    assert data_full_features.shape[1] == 237
    print('finish creating {} data set'.format(make_set))
    data_full_features.to_pickle('data/{}_full_features.pickle'.format(make_set))


make_data(data_folder='data', make_set='train')
make_data(data_folder='data', make_set='test')
end_time = time.time()
print('spent {:.2f} mins'.format((end_time - start_time) / 60))
