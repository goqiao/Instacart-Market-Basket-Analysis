import pandas as pd
from utils import read_data

# parameters:
data_folder = 'data'
nrows = None

prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))
_up = prior_order_details.groupby(['user_id', 'product_id']).agg({'order_id': 'count',
                                                                  'add_to_cart_order': ['mean', 'std', 'sum', 'min', 'max', 'median'],
                                                                  'reordered': 'sum',
                                                                  'order_number': ['min','max', 'mean', 'std'],
                                                                  })

# TODO: _up_max_days_since_prior, _up_mean_days_interval, _up_std_days_interval are not meaningful and low feature importances
_up.columns = _up.columns.get_level_values(0) + '_' + _up.columns.get_level_values(1)
up_agg = _up.rename(columns={'order_id_count': 'up_num_purchases',
                             'add_to_cart_order_mean': 'up_cart_order_mean', 'add_to_cart_order_std': 'up_cart_order_std',
                             'add_to_cart_order_sum': 'up_cart_order_sum', 'add_to_cart_order_min': 'up_cart_order_min',
                             'add_to_cart_order_max': 'up_cart_order_max', 'add_to_cart_order_median': 'up_cart_order_median',
                             'reordered_sum': 'up_reorder_times',
                             'order_number_min': 'up_first_order',
                             'order_number_max': 'up_last_order',
                             'order_number_mean': 'up_mean_order_num',
                             'order_number_std': 'up_std_order_num',
                             }).reset_index()

# TODO : _up_reorder_ratio is either 0 or 1 and has least feature importance
# up_agg['_up_reorder_ratio'] = up_agg['_up_reorder_times'] / (up_agg['_up_num_purchases'] - 1)

# TODO: Not clear about meaning of _up_order_interval, consider removing
# up_agg['_up_order_interval'] = up_agg['_up_mean_order_num'] / up_agg['_up_num_purchases']

orders = read_data(data_folder=data_folder, nrows=nrows, read_orders=True)

# note: _user_days_since_last_order is repaced by _user_days_not_purchase in the users_features
# up_agg = up_agg.merge(orders.loc[orders['eval_set'] != 'prior', ['user_id', 'days_since_prior_order']], on='user_id',
#                       how='left')
# up_agg.rename(columns={'days_since_prior_order': '_user_days_since_last_order'}, inplace=True)

print(up_agg.shape)
print(up_agg[['user_id', 'product_id']].nunique())

# merge with base
base_up = pd.read_pickle('{}/base.pickle'.format(data_folder))[['user_id', 'product_id']]
up_agg_final = base_up.merge(up_agg, on=['user_id', 'product_id'], how='left')

# std metrics are null due to only 1 record: _up_std_cart_order, _up_std_order_num
up_agg.fillna(-1, inplace=True)

up_agg.to_pickle('{}/up_agg.pickle'.format(data_folder))

