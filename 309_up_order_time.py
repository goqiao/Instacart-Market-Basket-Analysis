import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from utils import download_user_order_history, read_data

start_time = time.time()
data_folder = 'data'
nrows = None

orders = pd.read_pickle('data/orders.pickle')[['order_id', 'user_id', 'order_dow', 'part_of_day']]
prior_order_details = pd.read_pickle('data/prior_order_details.pickle')[['order_id', 'user_id', 'product_id']]

prior_order_details = prior_order_details.merge(orders, on= ['order_id', 'user_id'], how='left')

prior_order_details['up'] = prior_order_details['user_id'].astype('str') + ' ' + prior_order_details['product_id'].astype('str')
# print(prior_order_details.head())

dow = pd.crosstab(prior_order_details['up'], prior_order_details['order_dow']).add_prefix('up_purchases_dow_')
dow_norm = pd.crosstab(prior_order_details['up'], prior_order_details['order_dow'], normalize='index').add_prefix('up_norm_purchases_dow_')

part_of_day = pd.crosstab(prior_order_details['up'], prior_order_details['part_of_day']).add_prefix('up_purchases_pod_')
part_of_day_norm = pd.crosstab(prior_order_details['up'], prior_order_details['part_of_day'], normalize='index').add_prefix('up_norm_purchases_pod_')

up_order_time = pd.concat([dow, dow_norm, part_of_day, part_of_day_norm], axis=1).reset_index()
up_order_time['user_id'] = up_order_time['up'].str.split(expand=True)[0]
# up_order_time['user_id'] = up_order_time['up'].map(lambda x: x.split()[0])

up_order_time['product_id'] = up_order_time['up'].str.split(expand=True)[1]
# up_order_time['product_id'] = up_order_time['up'].map(lambda x: x.split()[1])
up_order_time[['user_id', 'product_id']] = up_order_time[['user_id', 'product_id']].astype('int')


print(up_order_time.head())
up_order_time.drop('up', axis=1).to_pickle('data/up_order_time.pickle')



end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))