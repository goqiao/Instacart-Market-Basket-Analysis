import pandas as pd
import numpy as np
import time
import gc


# parameters:
start_time = time.time()
data_folder = 'data'

up_purchase_time = pd.read_pickle('{}/days_since_last_purchase.pickle'.format(data_folder))

key = ['user_id', 'product_id']
up = up_purchase_time.groupby(key)['_up_days_since_last_purchase'].mean().to_frame()
up.columns = ['up_purchase_interval_days_mean']

up['up_purchase_interval_days_median'] = up_purchase_time.groupby(key)['_up_days_since_last_purchase'].median()
up['up_purchase_interval_days_max'] = up_purchase_time.groupby(key)['_up_days_since_last_purchase'].max()
up['up_purchase_interval_days_min'] = up_purchase_time.groupby(key)['_up_days_since_last_purchase'].min()

up.reset_index().to_pickle('{}/up_days_since_last_purchase.pickle'.format(data_folder))
gc.collect(); del up


print('n5')
# product purchasing cycles in last 5 orders
T = 5
prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))
orders_r5 = prior_order_details[prior_order_details['last_nth_order'].isin(np.arange(1, T+1))].order_id
up_purchase_time_r5 = up_purchase_time.loc[up_purchase_time.order_id.isin(orders_r5)]

key = ['user_id', 'product_id']
up_r5 = up_purchase_time_r5.groupby(key)['_up_days_since_last_purchase'].median().to_frame()
up_r5.columns = ['up_purchase_interval_days_median_r5']

up_r5['up_purchase_interval_days_max_r5'] = up_purchase_time_r5.groupby(key)['_up_days_since_last_purchase'].max()
up_r5['up_purchase_interval_days_min_r5'] = up_purchase_time_r5.groupby(key)['_up_days_since_last_purchase'].min()

up_r5.reset_index().to_pickle('{}/up_days_since_last_purchase_r5.pickle'.format(data_folder))


end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))

