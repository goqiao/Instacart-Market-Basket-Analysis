import pandas as pd
import numpy as np
import time
import seaborn as sns
from utils import q20, q80

'''
summary: how many times each product been purchased so far? 
If on average, it's been bought 3 times per user, average users usuall won't place more than 3 orders on it
'''

# parameters:
start_time = time.time()
data_folder = 'data'
nrows = None

base_prods = pd.read_pickle('{}/base.pickle'.format(data_folder))['product_id'].to_frame().drop_duplicates()
prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))

# prior_order_details = prior_order_details.loc[prior_order_details['product_id'] == 4781,]
# print(prior_order_details.shape)
# print(prior_order_details.head())

prod_users = prior_order_details.groupby(['product_id', 'user_id']).agg(
    {'order_number': 'size'}).rename(columns={'order_number': 'num_purchases_per_user'}).reset_index()

# print(prod_users['num_purchases_per_user'].describe([.2, .25, .5, .75, .8, .9, .95]))

# sns.boxplot(prod_users['num_purchases_per_user'])
print('cal stats..')
# TODO: num_purchases_per_old_buyers
prod = prod_users.groupby(['product_id']).agg({'num_purchases_per_user':['mean', 'std', 'median', 'max', 'min', q20, q80]}).reset_index()
prod.columns = ['product_id', 'p_num_purchases_per_user_mean','p_num_purchases_per_user_std',
                'p_num_purchases_per_user_median',
                'p_num_purchases_per_user_max', 'p_num_purchases_per_user_min',
                'p_num_purchases_per_user_q20', 'p_num_purchases_per_user_q80']
print(prod.head())

base_prods = base_prods.merge(prod, how='left')

base_prods.to_pickle('{}/products_purchases_features.pickle'.format(data_folder))

end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))
