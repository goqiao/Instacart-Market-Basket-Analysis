import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from utils import trend_d1

# parameters:
start_time = time.time()
data_folder = 'data'
nrows = None

orders = pd.read_pickle('data/orders.pickle')
orders = orders.loc[~orders['days_since_prior_order'].isnull(), ]
orders.sort_values(by=['user_id', 'order_number'], inplace=True)
print(orders.head())

orders_interval = orders.groupby('user_id').apply(lambda x: trend_d1(x['days_since_prior_order'])).reset_index()
orders_interval.columns =['user_id', 'user_orders_days_interval_trend']

orders_interval.to_pickle('data/users_orders_interval_trend.pickle')
print(2)

