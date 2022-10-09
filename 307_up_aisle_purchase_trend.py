import pandas as pd
import time
from utils import trend_d1, skewness
import gc


# parameters:
start_time = time.time()
data_folder = 'data'


prior = pd.read_pickle('data/prior_order_details.pickle')[['user_id', 'order_number', 'product_id']]
products = pd.read_pickle('data/products.pickle')[['product_id', 'aisle']]
prior = prior.merge(products, how='left')

users_aisles = prior.groupby(['user_id', 'aisle', 'order_number']).agg({'product_id':'nunique'}).reset_index()
users_aisles.columns = ['user_id', 'aisle', 'order_number', 'num_products']
users_orders = prior[['user_id', 'order_number']].drop_duplicates()
del prior, products; gc.collect()

df = users_aisles[['user_id', 'aisle']].drop_duplicates().merge(users_orders, on='user_id', how='outer')
users_aisles = df.merge(users_aisles, how='left', on=['user_id', 'aisle', 'order_number']).fillna(0)
del df; gc.collect()

users_aisles_trend = users_aisles.groupby(['user_id', 'aisle']).agg({'num_products': ['mean', trend_d1, skewness]}).reset_index()
users_aisles_trend.columns = ['user_id', 'aisle', 'up_mean_prods_same_aisle_per_order', 'up_aisle_purchase_trend_d1', 'up_aisle_purchase_skew']
del users_aisles; gc.collect()

# merge with base
base = pd.read_pickle('data/base.pickle')[['user_id', 'product_id']]
products = pd.read_pickle('data/products.pickle')[['product_id', 'aisle']]
base = base.merge(products, how='left')

users_aisles_trend = base.merge(users_aisles_trend, on=['user_id', 'aisle'])
users_aisles_trend[['user_id', 'product_id', 'up_mean_prods_same_aisle_per_order', 'up_aisle_purchase_trend_d1', 'up_aisle_purchase_skew']].to_pickle(
                                                            '{}/up_aisles_purchase_trend.pickle'.format(data_folder))



end_time = time.time()
print('spent {:.2f} mins'.format((end_time - start_time) / 60))
