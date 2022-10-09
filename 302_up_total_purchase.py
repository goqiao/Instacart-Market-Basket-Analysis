import pandas as pd
import time


# parameters:
data_folder = 'data'
nrows = None
start_time = time.time()

prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))

T = 5
orders_within_5 = prior_order_details.loc[
    (prior_order_details['last_nth_order'] >= 1) & (prior_order_details['last_nth_order'] <= T),]

cols = ['user_id', 'product_id']
up_purchase_r5 = orders_within_5.groupby(cols, as_index=False).size()
up_purchase_r5.columns = ['user_id', 'product_id', 'up_num_purchases_r5']

df = prior_order_details[['user_id', 'product_id', 'order_number', 'user_max_order']]
df = df.drop(columns='order_number').drop_duplicates()

df = df.merge(up_purchase_r5, on=cols, how='left')
df['up_purchase_ratio_r5'] = df['up_num_purchases_r5']/df['user_max_order'].apply(lambda x: min(T, x))
df.fillna(0, inplace=True)

df[['user_id', 'product_id', 'up_num_purchases_r5', 'up_purchase_ratio_r5']].to_pickle('{}/up_purchase_r5.pickle'.format(data_folder))


end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))