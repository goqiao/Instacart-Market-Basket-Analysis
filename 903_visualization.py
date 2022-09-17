import pandas as pd
from utils import read_data
import time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set(rc={'figure.figsize':(25, 18)})
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

start_time = time.time()

"""
create visualization for the EDA part of the report
"""

# Number of Orders by Days Since Prior Order
# orders = read_data(read_orders=True)
# sns.displot(data=orders, x='days_since_prior_order', kind='hist')
# plt.xlabel('Days Since Prior Orders')
# plt.ylabel('Num Orders')
# plt.title('Number of Orders by Days Since Prior Order', weight='bold')
# plt.tight_layout()
# plt.grid(False)
# plt.show(block=True)

# # Order Day of the week and hour of the day
# orders = read_data(read_orders=True)
# orders['order_dow'].replace({0: 'Sat', 1:'Sun', 2:'Mon', 3:'Tue', 4:'Wed', 5:'Thur', 6:'Fri'}, inplace=True)
# print(orders.head())
# orders_heatmap = pd.crosstab(orders['order_hour_of_day'], orders['order_dow'], normalize=True)
# plt.figure(figsize=(10,8))
# orders_heatmap = orders_heatmap[['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri']]
# print(orders_heatmap.head())
# sns.heatmap(data=orders_heatmap, cmap='GnBu')
# plt.xlabel('Order Day of Week (DoW)')
# plt.ylabel('Order Hour of Day')
# plt.title('Percentage of Orders by DoW and Hour of Day', weight='bold')
# plt.tight_layout()
# plt.show(block=True)

# # Num Orders by Basket Size
# prior_train_details = pd.read_pickle('data/prior_train_details.pickle')
# order_size = prior_train_details.groupby('order_id').agg({'product_id':'nunique'}).rename(columns={'product_id':'basket_size'}).reset_index()
# print(order_size.head())
# basket_size_dist = order_size.groupby('basket_size').agg({'order_id':'nunique'}).reset_index()
# print(basket_size_dist.head())
# plt.figure(figsize=(14,8))
# sns.barplot(x='basket_size', y='order_id', data=basket_size_dist)
# plt.xlabel('Num of Items in Basket')
# plt.ylabel('Num Orders')
# plt.title('Number of Orders by Basket Size', weight='bold')
# plt.tight_layout()
# plt.xticks(ticks=range(0, 145, 2), labels=range(1, 146, 2), rotation='vertical')
# plt.show(block=True)

# top ordered products
# prior_train_details = pd.read_pickle('data/prior_train_details.pickle')
# products = pd.read_pickle('data/products.pickle')
# prior_train_details = prior_train_details.merge(products)
# top_products = prior_train_details.groupby('aisle').agg({'order_id':'nunique'}).reset_index().rename(columns={'order_id':'purchase_times'})
# top_products.sort_values(by='purchase_times', ascending=False, inplace=True)
# print(top_products[['aisle', 'purchase_times']])

# top reordered aisles
# prior_train_details = pd.read_pickle('data/prior_train_details.pickle')
# products = pd.read_pickle('data/products.pickle')
# prior_train_details = prior_train_details.merge(products)
# reorder = prior_train_details.groupby('aisle').agg({'reordered': 'mean'}).reset_index().rename(columns={'reordered':'reorder_rate'})
# reorder.sort_values(by='reorder_rate', ascending=False, inplace=True)
# print(reorder)


# add to cart order vs reorder rate  (consider add)
# prior_train_details = pd.read_pickle('data/prior_train_details.pickle')
# # print(prior_train_details.head())
# stats = prior_train_details.groupby('add_to_cart_order').agg({'reordered':'mean',
#                                                       'order_id':'size'})
# stats = stats.loc[stats['order_id'] > 300, ]
# # print(stats.sort_values(by='order_id', ascending=True).head(20))
# stats['reordered'].plot(figsize=(10, 6))
# plt.show()


# Pareto analysis on product sales (can add)
prior_train_details = pd.read_pickle('data/prior_train_details.pickle')
products = pd.read_pickle('data/products.pickle')[['product_id', 'product_name']]
prior_train_details = prior_train_details.merge(products, how='left')
product_sales = prior_train_details.groupby('product_name').agg({'order_id':'size'})
product_sales.sort_values(by='order_id', ascending=False, inplace=True)
product_sales['cumsum'] = product_sales['order_id'].cumsum()
# product_sales.plot(kind='bar')
# plt.show()
product_sales[['cumsum']].reset_index().plot(figsize=(10, 6))
plt.xlabel('products')
plt.ylabel('sales')
plt.title('Cumulative Sales of Products')
plt.show()






end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))