import pandas as pd
import numpy as np
import time

# parameters:
data_folder = 'data'
nrows = None
start_time = time.time()

prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))


product_1_temp = prior_order_details.groupby(['product_id']).agg({'order_id': ['nunique'],
                                                                  'user_id': ['nunique'],
                                                                  'add_to_cart_order': ['mean', 'std'],
                                                                  'reordered': ['mean', 'sum'],
                                                                  # 'days_since_prior_order': ['sum', 'mean', 'std'],
                                                                  '_up_purchase_order': lambda x: np.sum(x==2)})
                                                                 # TODO: add more users purchases times

product_1_temp.columns = product_1_temp.columns.get_level_values(0) + '_' + product_1_temp.columns.get_level_values(1)

# TODO: _p_sum_days_since_prior_order, _p_mean_days_interval, _p_std_days_interval are not meaningful here and have low fi
product_1 = product_1_temp.rename(columns={'order_id_nunique': 'p_num_purchases',
                                           'user_id_nunique': 'p_unique_buyers',
                                           'add_to_cart_order_mean': 'p_mean_add_cart_num',
                                           'add_to_cart_order_std': 'p_std_add_cart_num',
                                           'reordered_mean': 'p_reorder_rate',
                                           'reordered_sum': 'p_sum_reordered',
                                           # 'days_since_prior_order_sum': '_p_sum_days_since_prior_order',
                                           # 'days_since_prior_order_mean': '_p_mean_days_interval',
                                           # 'days_since_prior_order_std': '_p_std_days_interval',
                                           '_up_purchase_order_<lambda>': 'p_sum_secondtime_purchase'})
product_1.reset_index(inplace=True)

# product_1['p_sum_onetime_purchase'] = product_1['p_num_purchases'] - product_1['p_sum_reordered'] # duplicate with p_unique_buyers
product_1['p_ratio_2nd_to_onetime_purchases'] = product_1['p_sum_secondtime_purchase']/product_1['p_unique_buyers']
# subsitute with p_reorder_rate
# product_1['p_ratio_onetime_purchases'] = product_1['p_sum_onetime_purchase']/product_1['p_num_purchases']


# fist order and first reorder
first_order = prior_order_details.loc[prior_order_details['reordered'] == 0].groupby(['product_id', 'user_id']).agg(
    {'order_number': 'min'})
first_reorder = prior_order_details.loc[prior_order_details['reordered'] == 1].groupby(['product_id', 'user_id']).agg(
    {'order_number': 'min'})

first_reorder_diff = (first_reorder - first_order).reset_index()
# first_reorder_diff.fillna(-1, inplace=True)
first_reorder_diff = first_reorder_diff.groupby('product_id').agg({'order_number': ['mean', 'std']})
first_reorder_diff.columns = first_reorder_diff.columns.get_level_values(
    0) + '_' + first_reorder_diff.columns.get_level_values(1)

first_reorder_diff = first_reorder_diff.rename(columns={'order_number_mean': 'p_avg_first_reorder_diff',
                                                        'order_number_std': 'p_std_first_reorder_diff'}).reset_index()


p_first_order = first_order.reset_index().groupby('product_id').agg({'order_number': ['mean', 'std']})

p_first_order.columns = p_first_order.columns.get_level_values(0) + '_' + p_first_order.columns.get_level_values(1)

p_first_order = p_first_order.rename(columns={'order_number_mean': 'p_avg_first_order_num',
                                              'order_number_std': 'p_std_first_order_num'}).reset_index()


p_first_reorder = first_reorder.reset_index().groupby('product_id').agg({'order_number': ['mean', 'std']})

p_first_reorder.columns = p_first_reorder.columns.get_level_values(0) + '_' + p_first_reorder.columns.get_level_values(
    1)

p_first_reorder = p_first_reorder.rename(columns={'order_number_mean': 'p_avg_first_reorder_num',
                                                  'order_number_std': 'p_std_first_reorder_num'}).reset_index()
print(p_first_reorder.shape)
print(p_first_reorder.head())

product_features = product_1.merge(p_first_order, how='left').merge(p_first_reorder, how='left').merge(
    first_reorder_diff, how='left')


# Done: remove duplicate with p_purchase_times_per_user_mean
# product_features['_p_purchases_time_per_buyer'] = product_features['_p_purchase_times'] / product_features[
#     '_p_unique_buyers']
print(product_features.shape)

# some products are only ordered once or have never been re-orderd, so features related to their std or averages are 0
print(product_features.isnull().sum())
product_features.fillna(0, inplace=True)


product_features.to_pickle('{}/product_features_basic_agg.pickle'.format(data_folder))

end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))