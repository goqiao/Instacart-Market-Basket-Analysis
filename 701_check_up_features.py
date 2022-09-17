import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

start_time = time.time()
data_folder = 'data'



# to check up features
base = pd.read_pickle('{}/base.pickle'.format(data_folder))
up_agg = pd.read_pickle('{}/up_agg.pickle'.format(data_folder))
up_purchase_r5 = pd.read_pickle('{}/up_purchase_r5.pickle'.format(data_folder))
up_days_since_last_purchase = pd.read_pickle('{}/up_days_since_last_purchase.pickle'.format(data_folder))
up_days_since_last_purchase_r5 = pd.read_pickle('{}/up_days_since_last_purchase_r5.pickle'.format(data_folder))
up_purchase_proba = pd.read_pickle('{}/up_purchase_proba.pickle'.format(data_folder))
up_purchase_proba_r5 = pd.read_pickle('{}/up_purchase_proba_r5.pickle'.format(data_folder))
up_days_not_purchase = pd.read_pickle('{}/up_days_not_purchase.pickle'.format(data_folder))

cols=['user_id', 'product_id']
base = base.loc[(base.user_id==1) & (base.product_id==196)]
up =base.merge(up_agg, on=cols).merge(up_purchase_r5, on=cols).merge(up_days_since_last_purchase, on=cols).merge(
up_days_since_last_purchase_r5, on=cols).merge(up_purchase_proba, on=cols).merge(up_purchase_proba_r5, on=cols).merge(up_days_not_purchase, on=cols)

print(up)


# need to evaluate why up_days_since_last_purchase_mean_r5 is predictive