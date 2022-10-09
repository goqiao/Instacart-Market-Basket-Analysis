import pandas as pd
import time
from utils import max_no_outliers, min_no_outliers, q20, q80


# parameters:
start_time = time.time()

cols = ['user_id', 'product_id', 'order_number']
prior_order_details = pd.read_pickle('data/prior_order_details.pickle')[cols]
df = prior_order_details.sort_values(by=['user_id', 'product_id', 'order_number'])

df['up_order_interval'] = df['order_number'] - df.groupby(['user_id', 'product_id'])['order_number'].shift()
order_interval = df.groupby(['product_id']).agg({'up_order_interval':['mean', 'std', 'median', q20, q80, max_no_outliers, min_no_outliers]})
order_interval.columns = ['p_order_interval_mean', 'p_order_interval_std', 'p_order_interval_median', 'p_order_interval_q20',
                        'p_order_interval_q80', 'p_order_interval_max_woo', 'p_order_interval_min_woo']

order_interval.to_pickle('data/p_order_interval.pickle')

end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))