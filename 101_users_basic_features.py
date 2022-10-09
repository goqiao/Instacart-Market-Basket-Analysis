import pandas as pd
from utils import read_data
import time

pd.set_option('display.max_columns', None)

# parameters:
data_folder = 'data'
nrows = None
start_time = time.time()

orders = read_data(data_folder=data_folder, nrows=nrows, read_orders=True)
prior = read_data(data_folder=data_folder, nrows=nrows, read_prior=True)

# get users level metrics
users_1 = orders.loc[orders['eval_set'] == 'prior'].groupby('user_id').agg(
    {'days_since_prior_order': ['sum', 'mean', 'std'],
     'order_number': 'max',
     'order_dow': ['mean', 'std'],
     'order_hour_of_day': ['mean', 'std']
     })

# flatten hierarchical index in column names
users_1.columns = users_1.columns.get_level_values(0) + '_' + users_1.columns.get_level_values(1)
# TODO: add function flatten column names

users_1.rename(columns={
                        'days_since_prior_order_sum': 'user_age_days_on_platform',
                        'days_since_prior_order_mean': 'user_mean_days_order_interval',
                        'days_since_prior_order_std': 'user_std_days_order_interval',
                        'order_number_max': 'user_total_orders',
                        'order_dow_mean': 'user_mean_order_dow',
                        'order_dow_std': 'user_std_order_dow',
                        'order_hour_of_day_mean': 'user_mean_order_hour',
                        'order_hour_of_day_std': 'user_std_order_hour'}, inplace=True)
users_1.reset_index(inplace=True)

# merge prior with orders to get user_id
# attach user level info to prior. We ignored train and test orders by joining with prior dataset
prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))
products = pd.read_pickle('{}/products.pickle'.format(data_folder))
prior_order_details = prior_order_details.merge(products, how='left')
print(prior_order_details.shape)
prior_order_details.head(2)

# TODO: drop count on aisle and department as they are the same for the product count
users_2 = prior_order_details.groupby('user_id').agg({'product_id': ['count', 'nunique'],
                                                      'aisle': ['nunique'],
                                                      'department': ['nunique'],
                                                      'reordered': 'sum',
                                                      })

users_2.columns = users_2.columns.get_level_values(0) + '_' + users_2.columns.get_level_values(1)

users_2.rename(columns={'product_id_count': 'user_product_total',
                        'product_id_nunique': 'user_product_unique',
                        'aisle_nunique': 'user_aisle_unique',
                        'department_nunique': 'user_department_unique',
                        'reordered_sum': 'user_reorder_prod_total',
                        }, inplace=True)

users_2.reset_index(inplace=True)

# create feature _user_days_not_purchase
users_2 = users_2.merge(orders.loc[orders['eval_set'] != 'prior', ['user_id', 'days_since_prior_order']],
                        how='left').rename(columns={'days_since_prior_order': 'user_days_not_purchase'})

## aggregate on users_orders(uo) level
users_3_temp = prior_order_details.groupby(['user_id', 'order_id']).agg({'product_id': ['count'],
                                                                         'aisle': ['nunique'],
                                                                         'department': ['nunique'],
                                                                         'reordered': ['sum', 'mean']})

users_3_temp.columns = users_3_temp.columns.get_level_values(0) + '_' + users_3_temp.columns.get_level_values(1)

users_3 = users_3_temp.reset_index().drop(columns='order_id').groupby('user_id').agg(['mean', 'std'])
users_3.columns = users_3.columns.get_level_values(0) + '_' + users_3.columns.get_level_values(1)
# TODO: _uo_products_mean vs _user_mean_order_size, both measures basket size?
users_3.rename(columns={'product_id_count_mean': 'uo_basket_size_mean',
                        'product_id_count_std': 'uo_basket_size_std',
                        'aisle_nunique_mean': 'uo_unique_aisle_mean',
                        'aisle_nunique_std': 'uo_unique_aisle_std',
                        'department_nunique_mean': 'uo_unique_department_mean',
                        'department_nunique_std': 'uo_unique_department_std',
                        'reordered_sum_mean': 'uo_reordered_products_mean',
                        'reordered_sum_std': 'uo_reordered_products_std',
                        'reordered_mean_mean': 'uo_reorder_ratio_mean',
                        'reordered_mean_std': 'uo_reorder_ratio_std'}, inplace=True)
users_3.reset_index(inplace=True)

# merge users level features
users_features = users_1.merge(users_2).merge(users_3)

users_features['user_reorder_rate'] = users_features['user_reorder_prod_total'] / users_features[
    'user_product_total']

users_features['user_next_order_readiness'] = users_features['user_days_not_purchase'] - users_features['user_mean_days_order_interval']
users_features['user_order_freq_days_mean'] = users_features['user_age_days_on_platform']/users_features['user_total_orders']

users_features.to_pickle('{}/users_features.pickle'.format(data_folder))

end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))
