import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# parameters:
start_time = time.time()
data_folder = 'data'
nrows = None


products = pd.read_pickle('data/products.pickle')[['product_id', 'product_name']]
substitutes_name = [('Large Brown Eggs','Organic Large Brown Grade AA Cage Free Eggs'),
                        ('Large Brown Eggs','Organic Large Grade AA Brown Eggs'),
                        ('Green Seedless Grapes','Grape White/Green Seedless'),
                        ('Organic Large Grade AA Brown Eggs','Large Brown Eggs'),
                        ('Organic Broccoli','Organic Broccoli Crowns'),
                        ('Banana','Organic Banana'),
                        # ('Organic Reduced Fat Milk','Organic Reduced Fat Milk'),
                        ('Banana','Bag of Organic Bananas'),
                        ('Half And Half Cream','Organic Half & Half'),
                        ('Strawberries','Organic Strawberries')
]
substitutes_id = [(products.set_index('product_name').at[l[0], 'product_id'],
                         products.set_index('product_name').at[l[1], 'product_id']) for l in substitutes_name]
# print(substitutes_id)
substitutes_id_rev = [(l[1], l[0]) for l in substitutes_id]

substitute_list = substitutes_id + substitutes_id_rev

substitute = pd.DataFrame(substitute_list, columns=['product_id', 'substitute_id'])
# substitute.merge(products, left_on='subsitute_a', right_on ='product_name'])
base = pd.read_pickle('data/base.pickle')[['user_id', 'product_id']]
# base = base.merge(products, on='product_id', how='left')
# base = base.loc[base.user_id == 111429, ]
#

user_substitute = base.merge(substitute, on=['product_id'], how='left')
# <user_id, product_id, substitute_id>
# print(user_substitute)

up_purcahse = pd.read_pickle('data/up_agg.pickle')[['user_id', 'product_id', 'up_num_purchases']]
# up_purcahse = up_purcahse.merge(products, how='inner')
up_purcahse.columns = ['user_id', 'substitute_id', 'up_substitute_num_purchases']
user_substitute = user_substitute.merge(up_purcahse,  left_on=['user_id','substitute_id']
                                        , right_on=['user_id', 'substitute_id'], how='left')

up_purchases_r5 = pd.read_pickle('data/up_purchase_r5.pickle')[['user_id', 'product_id', 'up_num_purchases_r5']]
# up_purchases_r5 = up_purchases_r5.merge(products, how='inner')
up_purchases_r5.columns = ['user_id', 'substitute_id', 'up_subsitute_num_purchases_r5']
user_substitute = user_substitute.merge(up_purchases_r5,  left_on=['user_id','substitute_id']
                                        , right_on=['user_id', 'substitute_id'], how='left')
# print(user_substitute.head())
user_substitute.fillna(0, inplace=True)
user_substitute = user_substitute.groupby(['user_id', 'product_id']).agg({'up_substitute_num_purchases':'sum',
                                                                          'up_subsitute_num_purchases_r5':'sum'}).reset_index()
print(user_substitute.loc[user_substitute.user_id == 111429 ].head())

print(user_substitute[['up_substitute_num_purchases', 'up_subsitute_num_purchases_r5']].describe())
print(user_substitute.isnull().sum())
user_substitute.to_pickle('{}/up_substitute_purchase.pickle'.format(data_folder))


end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))
