import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from utils import read_data

"""
when order interval = 0, 
- lost orders 
- or forget to buy sth in the previous orders
- or didn't buy enough of sth

2% of prior orders have days_since_prior_order ==0, make improvement on these data probably won't improve performance much
"""

order_prods = pd.read_pickle('data/order_prods_list.pickle')
# print(order_prods.head())

orders = pd.read_pickle('data/prior_order_details.pickle')[['order_id', 'user_id', 'days_since_prior_order']].drop_duplicates()
orders['t-1_order_id'] = orders['order_id'].shift()
orders_interval_0 = orders.loc[orders.days_since_prior_order == 0]
orders_interval_0 = orders_interval_0.merge(order_prods.rename(columns={'order_id':'t-1_order_id', 'products':'t-1_products'}),
                                            on = 't-1_order_id', how='left')
orders_interval_0 = orders_interval_0.merge(order_prods, on='order_id', how='left')
print(orders_interval_0.shape)
print(orders_interval_0.order_id.nunique())
print(orders_interval_0.order_id.nunique()/orders.order_id.nunique())

orders_interval_0['new_products'] = orders_interval_0.apply(lambda x: set(x['products']).difference(set(x['t-1_products'])), axis=1)

# print(orders_interval_0.head())
orders_interval_0['num_new_products'] = orders_interval_0['new_products'].apply(lambda x: len(x))
orders_interval_0['num_products_t'] = orders_interval_0['products'].apply(lambda x: len(x))
orders_interval_0['num_same_products'] = orders_interval_0['num_products_t'] - orders_interval_0['num_new_products']
print(orders_interval_0.describe())
sns.histplot(orders_interval_0, x='num_new_products')
plt.show(block=True)


# orders_interval_0.loc[orders_interval_0.num_new_products == 0]
print(orders_interval_0.head())