import pandas as pd
pd.set_option('display.max_columns', 20)

# parameters:
data_folder = 'data'
nrows = None

prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))
# prior_order_details['last_nth_order'] = prior_order_details.groupby('user_id')['order_number'].transform(
#     np.max) + 1 - prior_order_details['order_number']

T = 5
orders_within_5 = prior_order_details.loc[
    (prior_order_details['last_nth_order'] >= 1) & (prior_order_details['last_nth_order'] <= T),]

cols = ['user_id', 'product_id']
up_purchase_r5 = orders_within_5.groupby(cols, as_index=False).size()
up_purchase_r5.columns = ['user_id', 'product_id', 'up_num_purchases_r5']
print(up_purchase_r5.head())
# up_purchases = prior_order_details.groupby(cols).size().reset_index()
# up_purchases.columns = ['user_id', 'product_id', 'up_num_purchases']

# up_size = up_purchases.merge(up_purchase_r5, on=cols, how='left')
# print(up_size.isnull().sum())
# up_size_54035 = up_size.loc[up_size['user_id'] == 139223, ]
# print(up_size_54035)

# 139223
df = prior_order_details[['user_id', 'product_id', 'order_number', 'user_max_order']]
df = df.drop(columns='order_number').drop_duplicates()

df = df.merge(up_purchase_r5, on=cols, how='left')
df['up_purchase_ratio_r5'] = df['up_num_purchases_r5']/df['user_max_order'].apply(lambda x: min(T, x))
df.fillna(0, inplace=True)
# print(df.isnull().sum())
# print(df.loc[df['user_id'] == 139223])
# print(df.loc[df['user_id'] == 54035])
print(df[cols].nunique())
print('Adding Features:', df.columns.values)
print(df.head())
df.to_pickle('{}/up_purchase_r5.pickle'.format(data_folder))
