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
train = pd.read_pickle('data/train_full_features.pickle').sample(frac=0.3, random_state=0)

# up_num_purchases_within5 and reorder rate？
# yes, users purchase the product more often, likely higher reorder rate. chi-squared test
# cross_counts = pd.crosstab(train['up_num_purchases_within5'], train['reordered'])
# cross = pd.crosstab(train['up_num_purchases_within5'], train['reordered'], normalize='index')
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