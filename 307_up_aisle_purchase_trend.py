import pandas as pd
import numpy as np
import time
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from utils import trend_d1, skew
import gc

# parameters:
start_time = time.time()
data_folder = 'data'
nrows = None


prior = pd.read_pickle('data/prior_order_details.pickle')[['user_id', 'order_number', 'product_id']]
# prior = prior.loc[prior.user_id==83412]
products = pd.read_pickle('data/products.pickle')[['product_id', 'aisle']]
prior = prior.merge(products, how='left')

users_aisles = prior.groupby(['user_id', 'aisle', 'order_number']).agg({'product_id':'nunique'}).reset_index()
users_aisles.columns = ['user_id', 'aisle', 'order_number', 'num_products']
users_orders = prior[['user_id', 'order_number']].drop_duplicates()
del prior, products; gc.collect()

df = users_aisles[['user_id', 'aisle']].drop_duplicates().merge(users_orders, on='user_id', how='outer')
print(df.head())
users_aisles = df.merge(users_aisles, how='left', on=['user_id', 'aisle', 'order_number']).fillna(0)
del df; gc.collect()
print(users_aisles.head())

# users_aisles.to_pickle('data/up_aisles_tmp.pickle')
users_aisles_trend = users_aisles.groupby(['user_id', 'aisle']).agg({'num_products': [trend_d1, skew]}).reset_index()

users_aisles_trend.columns = ['user_id', 'aisle', 'up_aisle_purchase_trend_d1', 'up_aisle_purchase_skew']
del users_aisles; gc.collect()
print(users_aisles_trend.head())

# merge with base
base = pd.read_pickle('data/base.pickle')[['user_id', 'product_id']]
# base = base.loc[base.user_id == 83412]
products = pd.read_pickle('data/products.pickle')[['product_id', 'aisle']]
base = base.merge(products, how='left')

users_aisles_trend = base.merge(users_aisles_trend, on=['user_id', 'aisle'])
# users_aisles_trend.columns = ['user_id', 'product_id', 'aisle', 'up_aisle_purchase_trend_d1']
users_aisles_trend[['user_id', 'product_id', 'up_aisle_purchase_trend_d1', 'up_aisle_purchase_skew']].to_pickle(
                                                            '{}/up_aisles_purchase_trend.pickle'.format(data_folder))
print(users_aisles_trend.loc[(users_aisles_trend.user_id == 1) & (users_aisles_trend.product_id == 39657)])


end_time = time.time()
print('spent {:.2f} mins'.format((end_time - start_time) / 60))
