import pandas as pd
import numpy as np
import gc
from utils import read_data, improve_data_type
import time

pd.set_option('display.max_rows', 90)

# takes 1.27 mins
start_time = time.time()
data_folder = 'data'
nrows = None
# sample_frac = 0.1  # have to sample on training set to avoid out of memory

def make_data(data_folder='data', make_set='train'):
    base = pd.read_pickle('{}/base.pickle'.format(data_folder))

    # read pre-created features
    up_agg = pd.read_pickle('{}/up_agg.pickle'.format(data_folder))
    up_purchase_r5 = pd.read_pickle('{}/up_purchase_r5.pickle'.format(data_folder))
    up_days_since_last_purchase = pd.read_pickle('{}/up_days_since_last_purchase.pickle'.format(data_folder))
    up_days_since_last_purchase_r5 = pd.read_pickle('{}/up_days_since_last_purchase_r5.pickle'.format(data_folder))
    up_purchase_proba = pd.read_pickle('{}/up_purchase_proba.pickle'.format(data_folder))
    up_purchase_proba_r5 = pd.read_pickle('{}/up_purchase_proba_r5.pickle'.format(data_folder))
    up_days_not_purchase = pd.read_pickle('{}/up_days_not_purchase.pickle'.format(data_folder))
    up_aisles_purchase_trend = pd.read_pickle('{}/up_aisles_purchase_trend.pickle'.format(data_folder))
    up_departments_purchase_trend = pd.read_pickle('{}/up_departments_purchase_trend.pickle'.format(data_folder))
    up_substitute_purchase = pd.read_pickle('{}/up_substitute_purchase.pickle'.format(data_folder))
    up_organic_substitute_purchase = pd.read_pickle('{}/up_organic_substitute_purchase.pickle'.format(data_folder))
    up_order_time = pd.read_pickle('{}/up_order_time.pickle'.format(data_folder))

    users_features = pd.read_pickle('{}/users_features.pickle'.format(data_folder))
    users_basket_size_trend = pd.read_pickle('{}/users_basket_size_trend.pickle'.format(data_folder))
    users_orders_interval_trend = pd.read_pickle('{}/users_orders_interval_trend.pickle'.format(data_folder))
    users_organic_purchases = pd.read_pickle('{}/users_organic_purchases.pickle'.format(data_folder))
    users_organic_purchases_r5 = pd.read_pickle('{}/users_organic_purchases_r5.pickle'.format(data_folder))
    users_order_time = pd.read_pickle('{}/users_order_time.pickle'.format(data_folder))
    users_cluster = pd.read_pickle('{}/user_kmeans_cluster.pickle'.format(data_folder))

    product_features_basic_agg = pd.read_pickle('{}/product_features_basic_agg.pickle'.format(data_folder))
    product_organic_features = pd.read_pickle('{}/product_organic_features.pickle'.format(data_folder))
    products_purchases_features = pd.read_pickle('{}/products_purchases_features.pickle'.format(data_folder))
    product_purchase_cycle = pd.read_pickle('{}/product_purchase_cycle.pickle'.format(data_folder))
    products_embedding = pd.read_pickle('{}/word2vec_prods_embedding_renamed.pickle'.format(data_folder))

    if make_set == 'train':
        # get train users <user_id, order_id, product_id, reordered>
        base = base.loc[base['eval_set'] == 'train']
        # base = base.sample(frac=sample_frac, random_state=0)
    elif make_set == 'test':
        # get test users <user_id, order_id, product_id, reordered>
        base = base.loc[base['eval_set'] == 'test']

    # add reordered label to the train data
    data_full_features = base.merge(up_agg, on=['user_id', 'product_id'], how='left').drop('eval_set', axis=1)
    print(data_full_features[['user_id', 'product_id']].nunique())

    # select sub features to avoid out of memory
    # selected_users_features = ['user_id',
    #                            # '_user_mean_days_since_prior_order',
    #                            '_user_product_total',
    #                            '_user_total_orders',
    #                            '_user_product_unique', '_uo_avg_products', '_uo_mean_reorered_products',
    #                            '_uo_mean_reorder_ratio', '_user_unique_product_rate',
    #                            '_user_reorder_rate', '_user_reorder_ratio']

    # selected_product_features = ['product_id', '_p_purchase_times', '_p_unique_buyers', '_p_mean_add_cart_num',
    #                              '_p_std_add_cart_num', '_p_reorder_rate', '_p_ratio_2nd_to_onetime_purchases',
    #                              # '_p_sum_days_since_prior_order', '_p_mean_days_interval',
    #                              '_p_avg_first_reorder_diff', '_p_purchases_time_per_buyer',
    #                              '_p_sum_secondtime_purchase']

    print('merge1')
    # attach all features to the unique user_id and product_id tuple
    data_full_features = data_full_features.merge(up_purchase_r5, on=['user_id', 'product_id'], how='left')
    del up_purchase_r5
    # fill None with -1 as 0 has a meaning
    print('merge2')
    # data_full_features = data_full_features.merge(up_days_since_last_purchase, on=['user_id', 'product_id'],
    #                                               how='left').fillna(0)
    data_full_features = data_full_features.merge(up_days_since_last_purchase, on=['user_id', 'product_id'],
                                                  how='left')
    del up_days_since_last_purchase
    print('merge3')
    # data_full_features = data_full_features.merge(up_days_since_last_purchase_r5, on=['user_id', 'product_id'],
    #                                               how='left').fillna(-1)
    data_full_features = data_full_features.merge(up_days_since_last_purchase_r5, on=['user_id', 'product_id'],
                                                  how='left')
    del up_days_since_last_purchase_r5
    print('merge4')
    data_full_features = data_full_features.merge(up_purchase_proba, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.merge(up_purchase_proba_r5, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.merge(up_days_not_purchase, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.merge(up_aisles_purchase_trend, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.merge(up_departments_purchase_trend, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.merge(up_substitute_purchase, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.merge(up_organic_substitute_purchase, on=['user_id', 'product_id'], how='left')
    data_full_features = data_full_features.merge(up_order_time, on=['user_id', 'product_id'], how='left')

    # data_full_features = data_full_features.merge(users_features[selected_users_features], on='user_id', how='inner')
    data_full_features = data_full_features.merge(users_features, on='user_id', how='inner')
    data_full_features = data_full_features.merge(users_basket_size_trend, on='user_id', how='inner')
    data_full_features = data_full_features.merge(users_orders_interval_trend, on='user_id', how='inner')
    data_full_features = data_full_features.merge(users_organic_purchases, on='user_id', how='left')
    data_full_features = data_full_features.merge(users_organic_purchases_r5, on='user_id', how='left')
    data_full_features = data_full_features.merge(users_order_time, on='user_id', how='left')
    data_full_features = data_full_features.merge(users_cluster, on='user_id', how='left')

    # data_full_features = data_full_features.merge(product_features[selected_product_features], on='product_id',
    #                                               how='inner')
    data_full_features = data_full_features.merge(product_features_basic_agg, on='product_id', how='inner')
    # release memory
    gc.collect(); del up_purchase_proba, up_purchase_proba_r5, up_days_not_purchase, users_features, product_features_basic_agg

    # data_full_features.to_pickle('data/merge_by_chunk_full.pickle')
    print('merge5')
    data_full_features = data_full_features.merge(products_purchases_features, on='product_id', how='left')
    data_full_features = data_full_features.merge(product_purchase_cycle, on='product_id', how='left')
    data_full_features = data_full_features.merge(product_organic_features, on='product_id', how='left')

    data_full_features = data_full_features.merge(products_embedding, on='product_id', how='left')

    print('merge5 done')
    gc.collect(); del products_purchases_features
    # create extra features
    data_full_features['up_num_orders_since_last_purchase'] = data_full_features['user_total_orders'] - \
                                                            data_full_features['up_last_order']

    # # TODO: is _up_order_rate biased, esp for those products that users just start to try out?
    # data_full_features['_up_order_rate'] = data_full_features['_up_num_purchases'] / data_full_features[
    #     '_user_total_orders']

    # data_full_features['up_order_rate_since_first_order'] = data_full_features['up_num_purchases'] / (
    #             data_full_features['user_total_orders'] - data_full_features['up_first_order'] + 1)

    # data_full_features['up_order_rate_since_first_order'] = (data_full_features['up_num_purchases'] -1) / (
    #             data_full_features['user_total_orders'] - data_full_features['up_first_order'])
    #
    # data_full_features['up_order_rate_since_first_order'] = data_full_features['up_order_rate_since_first_order'].replace(
    #     {float('-inf'): -1, float('inf'): -1}
    # )

    # if _up_overdue_days_mean > 0 and close to 0, time to reorder
    # if _up_overdue_days_mean < 0, pass the mean no-purchase window, might not reorder it
    data_full_features['up_overdue_days_mean'] = data_full_features['up_purchase_interval_days_mean'] - data_full_features['up_num_days_not_purchase']

    # if _up_overdue_days_max < 0, pass the max no-purchase window, might not reorder it
    data_full_features['up_overdue_days_max'] = data_full_features['up_purchase_interval_days_max'] - data_full_features['up_num_days_not_purchase']

    # _up_overdue_days_min close to or >0, likely to reorder,
    # if _up_overdue_days_min < 0, might too early for users to reorder
    data_full_features['up_overdue_days_min'] = data_full_features['up_num_days_not_purchase'] - data_full_features[
        'up_purchase_interval_days_min']
    print(data_full_features.shape)

    data_full_features['up_purchase_readiness'] = (data_full_features['up_num_days_not_purchase']/(data_full_features['up_purchase_interval_days_max'] \
                                                                    - data_full_features['up_purchase_interval_days_min'])).replace(np.inf, np.nan)


    # Are users close to lose interests on a product?
    data_full_features['up_num_purchases_diff_p_mean'] = data_full_features['up_num_purchases'] \
                                                          - data_full_features['p_num_purchases_per_user_mean']
    data_full_features['up_num_purchases_diff_p_max'] = data_full_features['up_num_purchases'] - data_full_features[
        'p_num_purchases_per_user_max']
    data_full_features['up_num_purchases_diff_p_min'] = data_full_features['up_num_purchases'] - data_full_features[
        'p_num_purchases_per_user_min']
    data_full_features['up_num_purchases_diff_p_q20'] = data_full_features['up_num_purchases'] - data_full_features[
        'p_num_purchases_per_user_q20']
    data_full_features['up_num_purchases_diff_p_q80'] = data_full_features['up_num_purchases'] - data_full_features[
        'p_num_purchases_per_user_q80']

    # Users individual cycle with general product cycles
    data_full_features['up_readiness_p_mean'] = data_full_features['up_num_days_not_purchase'] - data_full_features['p_purchase_interval_days_mean']
    data_full_features['up_readiness_p_median'] = data_full_features['up_num_days_not_purchase'] - data_full_features['p_purchase_interval_days_median']
    data_full_features['up_readiness_p_q20'] = data_full_features['up_num_days_not_purchase'] - data_full_features['p_purchase_interval_days_q20']
    data_full_features['up_readiness_p_q80'] = data_full_features['up_num_days_not_purchase'] - data_full_features['p_purchase_interval_days_q80']
    data_full_features['up_readiness_p_max_woo'] = data_full_features['up_num_days_not_purchase'] - data_full_features['p_purchase_interval_days_max_woo']
    data_full_features['up_readiness_p_min_woo'] = data_full_features['up_num_days_not_purchase'] - data_full_features['p_purchase_interval_days_min_woo']
    data_full_features['up_readiness_p_range'] = (data_full_features['up_num_days_not_purchase']/(data_full_features['p_purchase_interval_days_max_woo'] - \
                                                                                                 data_full_features['p_purchase_interval_days_min_woo'])).replace(np.inf, np.nan)

    # fill Null
    # data_full_features.fillna(0, inplace=True)
    print(data_full_features.isnull().sum())

    print(min(data_full_features['p_num_purchases']) >= 0)
    # clean memory
    gc.collect()
    data_full_features = improve_data_type(data_full_features)

    # change columns type to string
    data_full_features.columns = data_full_features.columns.astype('str')
    print(data_full_features.columns)
    print('{} set shape'.format(make_set))
    data_full_features.to_pickle('{}/{}_full_features.pickle'.format(data_folder, make_set))


make_data(data_folder='data', make_set='train')
make_data(data_folder='data', make_set='test')
end_time = time.time()
print('spent {:.2f} mins'.format((end_time - start_time) / 60))
