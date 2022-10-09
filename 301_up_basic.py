import pandas as pd
import time
from utils import read_data

# parameters:
data_folder = 'data'
nrows = None
start_time = time.time()


prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))
_up = prior_order_details.groupby(['user_id', 'product_id']).agg({'order_id': 'count',
                                                                  'add_to_cart_order': ['std', 'sum', 'min', 'max', 'median'],
                                                                  'reordered': 'sum',
                                                                  'order_number': ['min','max', 'mean', 'std'],
                                                                  })

_up.columns = _up.columns.get_level_values(0) + '_' + _up.columns.get_level_values(1)
up_agg = _up.rename(columns={'order_id_count': 'up_num_purchases',
                              'add_to_cart_order_std': 'up_cart_order_std',
                             'add_to_cart_order_sum': 'up_cart_order_sum', 'add_to_cart_order_min': 'up_cart_order_min',
                             'add_to_cart_order_max': 'up_cart_order_max', 'add_to_cart_order_median': 'up_cart_order_median',
                             'order_number_min': 'up_first_order',
                             'order_number_max': 'up_last_order',
                             'order_number_mean': 'up_mean_order_num',
                             'order_number_std': 'up_std_order_num',
                             }).reset_index()

orders = read_data(data_folder=data_folder, nrows=nrows, read_orders=True)

print(up_agg.shape)
print(up_agg[['user_id', 'product_id']].nunique())

# merge with base
base_up = pd.read_pickle('{}/base.pickle'.format(data_folder))[['user_id', 'product_id']]
up_agg_final = base_up.merge(up_agg, on=['user_id', 'product_id'], how='left')

# std metrics are null due to only 1 record: _up_std_cart_order, _up_std_order_num
up_agg.fillna(-1, inplace=True)

up_agg.to_pickle('{}/up_agg.pickle'.format(data_folder))


end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))