import pandas as pd
import time


# parameters:
start_time = time.time()
data_folder = 'data'
nrows = None

"""
features used in the final prediction
"""

prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))

T = 5
prior_order_details_r5 = prior_order_details.loc[
    (prior_order_details['last_nth_order'] >= 1) & (prior_order_details['last_nth_order'] <= T),]

_up = prior_order_details_r5.groupby(['user_id', 'product_id']).agg({
                                                                  'add_to_cart_order': ['mean', 'std', 'sum', 'min', 'max', 'median'],
                                                                  }).reset_index()

_up.columns = ['user_id', 'product_id', '_up_cart_order_mean_r5', '_up_cart_order_std_r5', '_up_cart_order_sum_r5',
               '_up_cart_order_min_r5', '_up_cart_order_max_r5', '_up_cart_order_median_r5']


# merge with base
base_up = pd.read_pickle('{}/base.pickle'.format(data_folder))[['user_id', 'product_id']]
up_cart_order_r5 = base_up.merge(_up, on=['user_id', 'product_id'], how='left')
up_cart_order_r5.to_pickle('{}/up_cart_order_r5.pickle'.format(data_folder))

end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))




