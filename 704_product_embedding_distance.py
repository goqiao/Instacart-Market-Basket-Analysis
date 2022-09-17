import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from utils import read_data

start_time = time.time()
data_folder = 'data'


prod_embedding = pd.read_pickle('data/word2vec_prods_embedding.pickle')
prods = pd.read_pickle('data/products.pickle')[['product_id', 'product_name']]
prod_embedding = prod_embedding.merge(prods)
# print(prod_embedding.head(2))
# print(prod_embedding.shape)

orgnic_stra = 21137
stra = 16797

print(prod_embedding.loc[prod_embedding.product_id==orgnic_stra].reset_index(drop=True).drop('product_name', axis=1)
      - prod_embedding.loc[prod_embedding.product_id==stra].reset_index(drop=True).drop('product_name', axis=1))
print(prod_embedding.loc[prod_embedding.product_name=='Organic Green Grapes'].reset_index(drop=True).drop('product_name', axis=1)
      - prod_embedding.loc[prod_embedding.product_name=='Green Grapes'].reset_index(drop=True).drop('product_name', axis=1))






end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))