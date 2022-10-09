import pandas as pd
import numpy as np
import time
import gc
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import seaborn as sns
from utils import max_no_outliers, min_no_outliers, q20, q80
'''
Summary: how long do users purchase a product? toilet paper and apples might have different purchase cycles
'''

# parameters:
start_time = time.time()
data_folder = 'data'

up_purchase_time = pd.read_pickle('{}/days_since_last_purchase.pickle'.format(data_folder))
# up_purchase_time = up_purchase_time.loc[up_purchase_time['product_id']==196]

product_purchase_cycle = up_purchase_time.groupby('product_id').agg(
    {'_up_days_since_last_purchase': ['mean', 'std', 'median', q20, q80, max_no_outliers, min_no_outliers]}).reset_index()
product_purchase_cycle.columns = ['product_id', 'p_purchase_interval_days_mean', 'p_purchase_interval_days_std',
                                 'p_purchase_interval_days_median', 'p_purchase_interval_days_q20',
                                 'p_purchase_interval_days_q80', 'p_purchase_interval_days_max_woo',
                                 'p_purchase_interval_days_min_woo']


# cola = up_purchase_time.loc[up_purchase_time['product_id']==196]
# # sns.boxplot(cola['_up_days_since_last_purchase'])
# print(cola['_up_days_since_last_purchase'].describe([.2, 0.5, .75, .8, .9, .1]))

base = pd.read_pickle('{}/base.pickle'.format(data_folder))['product_id'].to_frame().drop_duplicates()
product_purchase_cycle = base.merge(product_purchase_cycle, how='left')
product_purchase_cycle.to_pickle('{}/product_purchase_cycle.pickle'.format(data_folder))
# print(product_purchase_time.shape)
# print(product_purchase_time.product_id.nunique())

# print(product_purchase_time.loc[product_purchase_time['product_id']==196])

end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))