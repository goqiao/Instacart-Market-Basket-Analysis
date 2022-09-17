import pandas as pd
import numpy as np
import time
from utils import read_data
import gc

"""
summary:make essential dataframes and features that will be repeatedly used later
- products: merged with aisles and departments info
- prior_order_details: <user, order, product> level info for prior orders, with feature _up_purchase_order
- base: <'user_id', 'order_id', 'product_id', 'reordered'> for train and test orders
- up_days_since_last_purchase: <user, order, product> level info with features about products purchase days
"""
# parameters:
start_time = time.time()
data_folder = 'data'
nrows = None

# make products data set
aisles = pd.read_csv('{}/aisles.csv'.format(data_folder))
departments = pd.read_csv('{}/departments.csv'.format(data_folder))
products = pd.read_csv('{}/products.csv'.format(data_folder))
# merge with departments and aisles data
products = products.merge(departments, how='left').merge(aisles, how='left')
products.to_pickle('{}/products.pickle'.format(data_folder))

# make prior order details. The order history of all users
orders = read_data(data_folder=data_folder, nrows=nrows, read_orders=True)
prior = read_data(data_folder=data_folder, nrows=nrows, read_prior=True)
train = read_data(data_folder=data_folder, nrows=nrows, read_train=True)

# make orders pickle: add reverse order number + part of day
orders['order_number_reverse'] = orders.groupby('user_id')['order_number'].transform(np.max) + 1 - orders[
    'order_number']

def part_of_day(t):
    if t < 6:
        return 'midnight'
    elif t < 12:
        return 'morning'
    elif t < 18:
        return 'noon'
    else:
        return 'night'

orders['part_of_day'] = orders['order_hour_of_day'].map(part_of_day)
orders.to_pickle('{}/orders.pickle'.format(data_folder))

# orders:user/order level info, prior: product level info
prior_order_details = orders[['order_id', 'user_id', 'order_number', 'days_since_prior_order']].merge(prior,
                                                                                                      on='order_id',
                                                                                                      how='inner')

prior_order_details['_up_purchase_order'] = prior_order_details.groupby(['user_id', 'product_id'])[
    'order_number'].rank()
prior_order_details['last_nth_order'] = prior_order_details.groupby('user_id')['order_number'].transform(
    np.max) + 1 - prior_order_details['order_number']
prior_order_details['user_max_order'] = prior_order_details.groupby('user_id')['order_number'].transform(
    lambda x: x.max())
prior_order_details.to_pickle('{}/prior_order_details.pickle'.format(data_folder))

# make train & prediction base dataset <'user_id', 'order_id', 'product_id'>. order_id is the last order from users
# locate train & test orders
base = orders.loc[
    orders['eval_set'].isin(['train', 'test']), ['user_id', 'order_id', 'eval_set']]

# for train and test users, find all previously bought prods
base = base.merge(prior_order_details[['user_id', 'product_id']].drop_duplicates(), on='user_id')

# left join for <user_id, product_id>, find reordered label. Null for test users
base = base.merge(train[['order_id', 'product_id', 'reordered']], how='left')
# base = <'user_id', 'order_id', 'product_id', 'eval_set', 'days_since_prior_order', 'reordered'>

# for products that are not reordered in the train orders, fill 0
base.loc[base['eval_set'] == 'train', 'reordered'] = base.loc[base['eval_set'] == 'train', 'reordered'].fillna(0)
base.to_pickle('{}/base.pickle'.format(data_folder))

# make up_days_since_last_purchase (up short for user_product)
prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))
user_orders = prior_order_details[['user_id', 'order_number', 'days_since_prior_order']].drop_duplicates()
user_orders.sort_values(by=['user_id', 'order_number'], inplace=True)
user_orders['nth_day_since_customer'] = user_orders.groupby('user_id')['days_since_prior_order'].cumsum().fillna(0)

col = ['user_id', 'product_id', 'order_number', ]
_up = prior_order_details[col + ['order_id']].merge(user_orders, on=['user_id', 'order_number'])
gc.collect();
del prior_order_details, user_orders
_up.sort_values(by=col, inplace=True)

# diff() is slow in groupby, see discussion https://stackoverflow.com/questions/53150700/why-the-groupby-diff-is-so-slower
# _up['_up_days_since_last_purchase'] = _up.groupby(['user_id', 'product_id'])['nth_day_since_customer'].diff()
_up['_up_days_since_last_purchase'] = _up['nth_day_since_customer'] - \
                                      _up.groupby(['user_id', 'product_id'])['nth_day_since_customer'].shift()
_up.to_pickle('{}/days_since_last_purchase.pickle'.format(data_folder))

# make order_prods_list
prior = pd.read_pickle('data/prior_order_details.pickle')
order_prods = prior.groupby('order_id').apply(lambda x: x['product_id'].to_list()).reset_index()
order_prods.columns = ['order_id', 'products']
order_prods.to_pickle('{}/order_prods_list.pickle'.format(data_folder))

# make organic substitution dataset
products = pd.read_pickle('data/products.pickle')[['product_id', 'product_name']]
products['product_organic_name'] = 'Organic ' + products['product_name']
products_sub_pairs = products.loc[
    products['product_organic_name'].isin(products['product_name']), ['product_id', 'product_name',
                                                                      'product_organic_name']]
products_sub_pairs.columns = ['product_id', 'product_name', 'substitute_product_name']
products_sub_pairs = products_sub_pairs.merge(products[['product_id', 'product_name']].rename(columns=
                                                                                              {
                                                                                                  'product_id': 'substitute_id',
                                                                                                  'product_name': 'substitute_product_name'})
                                              , how='left')
products_sub_pairs = products_sub_pairs[['product_id', 'product_name', 'substitute_id', 'substitute_product_name']]
products_sub_pairs.to_pickle('data/products_organic_substitution.pickle')

# make product_features
products = pd.read_pickle('data/products.pickle')
products['p_is_organic'] = products['product_name'].str.lower().str.contains('organic')
products['p_is_gluten_free'] = (products['product_name'].str.lower().str.contains('gluten')) \
                             & (products['product_name'].str.lower().str.contains('free'))
products['p_is_asian'] = products['product_name'].str.lower().str.contains('asian')

products[['product_id', 'p_is_organic', 'p_is_gluten_free', 'p_is_asian']].to_pickle(
    '{}/product_organic_features.pickle'.format(data_folder))

end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))
