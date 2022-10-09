import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import gc
import time

pd.set_option('display.max_colwidth', None)
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from utils import trend_d1, skewness


# parameters:
start_time = time.time()
data_folder = 'data'

up_purchase_time = pd.read_pickle('{}/days_since_last_purchase.pickle'.format(data_folder))

up_orders = up_purchase_time[['user_id', 'product_id', 'order_number']]
up_orders['order_interval'] = up_orders.groupby(['user_id', 'product_id'])['order_number'].diff()
up_orders.head()

up_orders = up_orders.loc[~up_orders['order_interval'].isnull()]
up_orders_interval = up_orders.groupby(['user_id', 'product_id']).agg({'order_interval':[list, 'mean', 'max', 'min', 'median', trend_d1, skewness]})
up_orders_interval.columns = ['up_order_intervals', 'up_mean_order_interval', 'up_max_order_interval', 'up_min_order_interval', 'up_median_order_interval',
                                'up_order_interval_trend_d1', 'up_order_interval_skewness']
up_orders_interval.drop('up_order_intervals', axis=1).to_pickle('data/up_orders_interval.pickle')

# R5 version
T = 5
prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))
orders_r5 = prior_order_details[prior_order_details['last_nth_order'].isin(np.arange(1, T+1))].order_id
up_purchase_time_r5 = up_purchase_time.loc[up_purchase_time.order_id.isin(orders_r5)]

up_orders = up_purchase_time_r5[['user_id', 'product_id', 'order_number']]
up_orders['order_interval'] = up_orders.groupby(['user_id', 'product_id'])['order_number'].diff()
up_orders.head()

up_orders = up_orders.loc[~up_orders['order_interval'].isnull()]
up_orders_interval = up_orders.groupby(['user_id', 'product_id']).agg({'order_interval':[list, 'mean', 'max', 'min', 'median', trend_d1, skewness]})
up_orders_interval.columns = ['up_order_intervals_r5', 'up_mean_order_interval_r5', 'up_max_order_interval_r5', 'up_min_order_interval_r5', 'up_median_order_interval_r5',
                                'up_order_interval_trend_d1_r5', 'up_order_interval_skewness_r5']
up_orders_interval.drop('up_order_intervals', axis=1).to_pickle('data/up_orders_interval.pickle')


end_time = time.time()
print('spent {:.2f} mins'.format((end_time - start_time) / 60))



