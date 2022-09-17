import pandas as pd
import numpy as np
import gc
from utils import read_data, improve_data_type, compress
import time

pd.set_option('display.max_rows', 90)

'''
users:
- user num orders
users behaviour trend features
https://www.kaggle.com/competitions/instacart-market-basket-analysis/discussion/37336
'''
start_time = time.time()
data_folder = 'data'
nrows = None


def make_data_None(data_folder='data', make_set='train'):
    base = pd.read_pickle('{}/base.pickle'.format(data_folder))

    # read pre-created features
    users_features = pd.read_pickle('{}/users_features.pickle'.format(data_folder))

    if make_set == 'train':
        # get train users <user_id, None> where None indicates if users reordered
        base_None = base.loc[base['eval_set'] == 'train'].groupby('user_id').agg({'reordered': 'max'}).reset_index()
        base_None.columns = ['user_id', 'is_None']
        base_None['is_None'] = base_None['is_None'].replace({1: 0, 0: 1})
    elif make_set == 'test':
        # get test users <user_id, None> where None is empty
        base_None = base.loc[base['eval_set'] == 'test'].groupby('user_id').agg({'reordered': 'max'}).reset_index()
        base_None.columns = ['user_id', 'is_None']
        base_None['is_None'] = base_None['is_None'].replace({1: 0, 0: 1})

    data_full_features = base_None.merge(users_features, on='user_id', how='left')

    # user x item features
    up_basic = pd.read_pickle('{}/up_agg.pickle'.format(data_folder))
    up_cart_order_r5 = pd.read_pickle('{}/up_cart_order_r5.pickle'.format(data_folder))
    up_purchase_r5 = pd.read_pickle('{}/up_purchase_r5.pickle'.format(data_folder))
    up_days_since_last_purchase = pd.read_pickle('{}/up_days_since_last_purchase.pickle'.format(data_folder))
    up_days_since_last_purchase_r5 = pd.read_pickle('{}/up_days_since_last_purchase_r5.pickle'.format(data_folder))
    up_purchase_proba = pd.read_pickle('{}/up_purchase_proba.pickle'.format(data_folder))
    up_purchase_proba_r5 = pd.read_pickle('{}/up_purchase_proba_r5.pickle'.format(data_folder))
    up_days_not_purchase = pd.read_pickle('{}/up_days_not_purchase.pickle'.format(data_folder))

    up = up_basic.merge(up_cart_order_r5, on=['user_id', 'product_id'], how='left')
    up = up.merge(up_purchase_r5,  on=['user_id', 'product_id'], how='left')
    up = up.merge(up_days_since_last_purchase, on=['user_id', 'product_id'], how='left')
    up = up.merge(up_days_since_last_purchase_r5, on=['user_id', 'product_id'], how='left')
    up = up.merge(up_purchase_proba, on=['user_id', 'product_id'], how='left')
    up = up.merge(up_purchase_proba_r5, on=['user_id', 'product_id'], how='left')
    up = up.merge(up_days_not_purchase, on=['user_id', 'product_id'], how='left')

    gc.collect(); del up_basic, up_cart_order_r5, up_purchase_r5, up_days_since_last_purchase, up_days_since_last_purchase_r5,
    up_purchase_proba, up_purchase_proba_r5
    key = 'user_id'
    up_compressed = compress(up, key)
    data_full_features = data_full_features.merge(up_compressed, on='user_id', how='left')

    # product features
    product_features = pd.read_pickle('{}/product_features.pickle'.format(data_folder))
    base_up = pd.read_pickle('{}/base.pickle'.format(data_folder))[['user_id', 'product_id']]
    product_features = base_up.merge(product_features, on='product_id', how='left')

    key = 'user_id'
    compressed_product_features = compress(product_features, key)
    data_full_features = data_full_features.merge(compressed_product_features, on='user_id', how='left')

    data_full_features.to_pickle('{}/{}_None_full_features.pickle'.format(data_folder, make_set))

    print(make_set, ':')
    print(data_full_features.shape)


make_data_None(data_folder='data', make_set='train')
make_data_None(data_folder='data', make_set='test')

end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))
