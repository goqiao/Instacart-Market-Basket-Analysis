import pandas as pd
import numpy as np
import time
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from utils import trend_d1
import concurrent.futures
import gc

"""
run out of memory, due to parent memory?
"""

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

def multi(uid):
    users_aisles_uid = users_aisles.loc[users_aisles.user_id == uid]
    users_orders_uid = users_orders.loc[users_orders.user_id == uid]

    df = users_aisles_uid[['user_id', 'aisle']].drop_duplicates().merge(users_orders_uid, on='user_id', how='outer')
    users_aisles_uid = df.merge(users_aisles_uid, how='left', on=['user_id', 'aisle', 'order_number']).fillna(0)
    users_aisles_trend_uid = users_aisles_uid.groupby(['user_id', 'aisle']).apply(
        lambda x: trend_d1(x['num_products'])).reset_index()
    users_aisles_trend_uid.columns = ['user_id', 'aisle', 'aisle_trend_d1']
    return users_aisles_trend_uid


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        uids = users_orders['user_id'].drop_duplicates()
        results = executor.map(multi, uids)
    users_aisles_trend = pd.concat([result for result in results])
    users_aisles_trend.to_pickle('data/users_aisles_trend.pickle')

    # base = pd.read_pickle('data/base.pickle')[['user_id', 'product_id']]
    # products = pd.read_pickle('data/products.pickle')[['product_id', 'aisle']]
    # base = base.merge(products, how='left')
    # users_aisles_trend = base.merge(users_aisles_trend, on=['user_id', 'aisle'], how='left')
    # users_aisles_trend.columns = [['user_id', 'product_id', 'aisle', 'up_aisle_purchase_trend_d1']]
    # users_aisles_trend[['user_id', 'product_id', 'up_aisle_purchase_trend_d1']].to_pickle('{}/users_aisles_purchase_trend.pickle'.format(data_folder))
    # print(users_aisles_trend.isnull().sum())

    # TODO: check effect of indenting this block
    end_time = time.time()
    print('spent {:.2f} mins'.format((end_time - start_time) / 60))
    print(2)