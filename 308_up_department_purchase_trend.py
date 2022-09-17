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
products = pd.read_pickle('data/products.pickle')[['product_id', 'department']]
prior = prior.merge(products, how='left')

users_departments = prior.groupby(['user_id', 'department', 'order_number']).agg({'product_id':'nunique'}).reset_index()
users_departments.columns = ['user_id', 'department', 'order_number', 'num_products']
users_orders = prior[['user_id', 'order_number']].drop_duplicates()
del prior, products; gc.collect()

df = users_departments[['user_id', 'department']].drop_duplicates().merge(users_orders, on='user_id', how='outer')
print(df.head())
users_departments = df.merge(users_departments, how='left', on=['user_id', 'department', 'order_number']).fillna(0)
del df; gc.collect()
print(users_departments.head())

users_departments.to_pickle('data/up_departments_tmp.pickle')
users_departments_trend = users_departments.groupby(['user_id', 'department']).agg({'num_products': [trend_d1, skew]}).reset_index()

users_departments_trend.columns = ['user_id', 'department', 'up_department_purchase_trend_d1', 'up_department_purchase_skew']
del users_departments; gc.collect()
print(users_departments_trend.head())

# merge with base
base = pd.read_pickle('data/base.pickle')[['user_id', 'product_id']]
# base = base.loc[base.user_id == 83412]
products = pd.read_pickle('data/products.pickle')[['product_id', 'department']]
base = base.merge(products, how='left')

users_departments_trend = base.merge(users_departments_trend, on=['user_id', 'department'])
# users_departments_trend.columns = ['user_id', 'product_id', 'department', 'up_department_purchase_trend_d1']
users_departments_trend[['user_id', 'product_id', 'up_department_purchase_trend_d1', 'up_department_purchase_skew']].to_pickle(
                                                            '{}/up_departments_purchase_trend.pickle'.format(data_folder))
print(users_departments_trend.loc[(users_departments_trend.user_id == 1) & (users_departments_trend.product_id == 39657)])


end_time = time.time()
print('spent {:.2f} mins'.format((end_time - start_time) / 60))
