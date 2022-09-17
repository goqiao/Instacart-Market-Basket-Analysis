import pandas as pd
from utils import read_data
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set(rc={'figure.figsize':(25, 18)})
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

"""
No strong correlation between num_prods_same_aisle and aisle_reorder_rate, num_prods_same_department and 
department_reorder_rate because both of their pearson correlation < 0.4. The former is -0.01 and the latter is -0.22
"""


start_time = time.time()

products = pd.read_pickle('data/products.pickle')
aisle_competition = products.groupby('aisle').agg({'product_id':'nunique'}).reset_index()
aisle_competition.columns = ['aisle', 'num_products_same_aisle']

department_competition = products.groupby('department').agg({'product_id':'nunique'}).reset_index()
department_competition.columns = ['department', 'num_products_same_department']

# products = products.merge(aisle_competition, how='left').merge(department_competition, how='left')

prior_train_details = pd.read_pickle('data/prior_train_details.pickle')
prior_train_details = prior_train_details.merge(products)
#
# aisle_reorder = prior_train_details.groupby('aisle').agg({'reordered': 'mean'}).reset_index().rename(columns={'reordered':'reorder_rate'})
# aisle_reorder.sort_values(by='reorder_rate', ascending=False, inplace=True)
#
# aisle_reorder = aisle_reorder.merge(aisle_competition)
# print(aisle_reorder[['reorder_rate', 'num_products_same_aisle']].corr())
# print(aisle_reorder.head())

department_reorder = prior_train_details.groupby('department').agg({'reordered': 'mean'}).reset_index().rename(columns={'reordered':'reorder_rate'})
department_reorder.sort_values(by='reorder_rate', ascending=False, inplace=True)

department_reorder = department_reorder.merge(department_competition)
print(department_reorder[['reorder_rate', 'num_products_same_department']].corr())
print(department_reorder.head())
sns.relplot(data=department_reorder, x='num_products_same_department', y='reorder_rate', hue='department', palette='GnBu_r')
plt.show(block=True)

end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))
