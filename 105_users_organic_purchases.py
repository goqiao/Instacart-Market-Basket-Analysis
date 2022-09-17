import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# parameters:
data_folder = 'data'
nrows = None

prior_order_details = pd.read_pickle('{}/prior_order_details.pickle'.format(data_folder))[['user_id', 'product_id', 'last_nth_order']]
product_features = pd.read_pickle('{}/product_organic_features.pickle'.format(data_folder))[
                                        ['product_id', 'p_is_organic', 'p_is_gluten_free', 'p_is_asian']]
prior_order_details = prior_order_details.merge(product_features, how='left')

user_organic_counts = prior_order_details.groupby('user_id').agg({'product_id':'size',
                                                                 'p_is_organic':'sum',
                                                                 'p_is_gluten_free':'sum',
                                                                 'p_is_asian':'sum'}).reset_index()
user_organic_counts['user_organic_ratio'] = user_organic_counts['p_is_organic']/user_organic_counts['product_id']
user_organic_counts['user_gluten_free_ratio'] = user_organic_counts['p_is_gluten_free']/user_organic_counts['product_id']
user_organic_counts['user_asian_food_ratio'] = user_organic_counts['p_is_asian']/user_organic_counts['product_id']
user_organic_counts[['user_id', 'user_organic_ratio', 'user_gluten_free_ratio', 'user_asian_food_ratio']].to_pickle(
                                                                                '{}/users_organic_purchases.pickle'.format(data_folder))
print(user_organic_counts.head())


T = 5
prior_order_details = prior_order_details.loc[(prior_order_details.last_nth_order >=1) & (prior_order_details.last_nth_order <= T)]
user_organic_counts_r5 = prior_order_details.groupby('user_id').agg({'product_id':'size',
                                                                     'p_is_organic':'sum',
                                                                     'p_is_gluten_free':'sum',
                                                                     'p_is_asian':'sum'}).reset_index()
user_organic_counts_r5['user_organic_ratio_r5'] = user_organic_counts_r5['p_is_organic']/user_organic_counts_r5['product_id']
user_organic_counts_r5['user_gluten_free_ratio_r5'] = user_organic_counts_r5['p_is_gluten_free']/user_organic_counts_r5['product_id']
user_organic_counts_r5['user_asian_food_ratio_r5'] = user_organic_counts_r5['p_is_asian']/user_organic_counts_r5['product_id']
user_organic_counts_r5[['user_id', 'user_organic_ratio_r5', 'user_gluten_free_ratio_r5'
                    , 'user_asian_food_ratio_r5']].to_pickle('{}/users_organic_purchases_r5.pickle'.format(data_folder))
print(user_organic_counts_r5.head())
