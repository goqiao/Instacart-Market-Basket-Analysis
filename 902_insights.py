import pandas as pd
import numpy as np
import shap
import xgboost
import matplotlib.pyplot as plt
import time
import mlflow
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# sample_size = None
print(1)
train = pd.read_pickle('data/train_full_features.pickle').sample(frac=0.3, random_state=0)
print(2)
train.to_hdf('data/train_full_features_sampled_03.h5', key='stage', mode='w')
print(3)
train.to_pickle('data/train_full_features_sampled_03.pickle')

# train = pd.read_hdf('data/train_full_features_sampled_03.h5')

# X = pd.read_pickle('data/X_train.pickle').reset_index(drop=True)
# y = pd.read_pickle('data/y_train.pickle').reset_index(drop=True)
# train = pd.concat([X, y], axis=1)

# high reorder rate on gluten_free food?
# no, the opposite, gluten-free food might have lower reorder rate, need hypothesis testing
# print(pd.crosstab(train['is_gluten_free'], train['reordered'], normalize=True))



# up_num_purchases_r5 and reorder rate？
# yes, users purchase the product more often, likely higher reorder rate. chi-squared test
# cross_counts = pd.crosstab(train['up_num_purchases_r5'], train['reordered'])
# cross = pd.crosstab(train['up_num_purchases_r5'], train['reordered'], normalize='index')
# print(cross)
# cross.plot(kind='bar', stacked=True)
# plt.xlabel('up_number_purchases_r5')
# plt.ylabel('percentage')
# plt.title('Reorder Rate by up_number_purchase_r5')
# # add % on plot
# for n, x in enumerate([*cross.index.values]):
#     for (proportion, y_loc) in zip(cross.loc[x],
#                                    cross.loc[x].cumsum()):
#         plt.text(x=n - 0.22,
#                  y=(y_loc - proportion) + (proportion / 2),
#                  s=f'{np.round(proportion * 100, 1)}%',
#                  color="black",
#                  fontsize=8,
#                  fontweight="bold")
# plt.show(block=True)

# add % and number on plot
# for n, x in enumerate([*cross_counts.index.values]):
#     for (proportion, count, y_loc) in zip(cross.loc[x],
#                                           cross_counts.loc[x],
#                                           cross.loc[x].cumsum()):
#         plt.text(x=n - 0.22,
#                  y=(y_loc - proportion) + (proportion / 2),
#                  s=f'{count}\n({np.round(proportion * 100, 1)}%)',
#                  color="black",
#                  fontsize=8,
#                  fontweight="bold")
# plt.show(block=True)


# user_next_readiness and reorder rate?
# user_next_readiness > 0, pass the historical mean interval
# user_next_readiness <0, need to wait a few more days
# reordered distribution is more tailed than the non-reordered distribution
# reordered distribution centers at 0, while non-reordered distribution peaks at -1/-2
# not symmetric, non-reordered distribution bulk on the right. Passed mean interval, less liekly to reorder
# re-ordered distribution bulk on the left, users might reorder sooner than the mean interval
# bumpy when user_next_readiness > 0?
# sns.displot(x='user_next_order_readiness', data=train, hue='reordered', kde=True)
# plt.show(block=True)


# up_overdue_days_mean, similar to user_next_order_readiness, but smooth
# sns.displot(x='up_overdue_days_mean', data=train, hue='reordered', kde=True)
# plt.show(block=True)



# for users who purchase both generic and organic version of a product, if they purchase the organic subsitutes
# more than 4 times, their probability of purchasing the generic version reduce to 0
# cross = pd.crosstab(train['up_organic_substitute_num_purchases_r5'], train['reordered'], normalize='index')
# # cross.plot(kind='bar', stacked=True)
# cross.sort_index(ascending=False, inplace=True)
# cross[[1]].plot(kind='barh', legend=None)
# plt.ylabel('up_organic_substitute_num_purchases_r5')
# plt.xlabel('reorder rate')
# plt.title('Reorder Rate by Purchase of Organic Substitutes')
# plt.legend('None')
# plt.show()
# print(train.groupby(['up_organic_substitute_num_purchases_r5']).agg({'order_id':'size'}))
# cross_counts = pd.crosstab(train['up_organic_substitute_num_purchases_r5'], train['reordered'], normalize=False)
# cross_counts.plot(kind='bar', logy=True)
# plt.show()
# add % on plot
# for n, x in enumerate([*cross.index.values]):
#     for (proportion, y_loc) in zip(cross.loc[x],
#                                    cross.loc[x].cumsum()):
#         plt.text(x=n - 0.22,
#                  y=(y_loc - proportion) + (proportion / 2),
#                  s=f'{np.round(proportion * 100, 1)}%',
#                  color="black",
#                  fontsize=8,
#                  fontweight="bold")
# plt.show(block=True)
# print(train.loc[train.up_organic_substitute_num_purchases_r5==5, ['user_id', 'product_id']].head())



# effect of products
# sns.displot(data=train, x='users_asian_food_ratio_r5', hue='reordered')
# plt.show(block=True)


