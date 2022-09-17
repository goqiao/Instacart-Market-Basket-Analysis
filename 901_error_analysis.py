import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from utils import download_user_order_history, get_features_pred_by_uid_pid
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


start_time = time.time()
data_folder = 'data'

df = pd.read_pickle('{}/val_pred_res.pickle'.format(data_folder))
X_val = pd.read_pickle('data/X_val.pickle')
products = pd.read_pickle('data/products.pickle')[['product_id', 'product_name', 'aisle','department']]
# merge
df = df.merge(products, on=['product_id'], how='left').merge(X_val, on=['order_id', 'product_id'], how='left')
# print_eval_metrics(train_pred_with_features['reordered'], train_pred_with_features['pred_prob'], train_pred_with_features['pred_y'])
# print(df.head())

# train_wrong_pred = train_pred_with_features.loc[train_pred_with_features['reordered'] != train_pred_with_features['pred_y']]
fp = df.loc[(df['reordered'] == 0) & (df['pred_y'] == 1)]
fn = df.loc[(df['reordered'] == 1) & (df['pred_y'] == 0)]
# print(fp.sample(n=5, random_state=0))

# sns.histplot(fp, x='up_num_purchases')
# plt.show(block=True)
# peak at 2, 3

# sns.histplot(fn, x='up_num_purchases')
# plt.show(block=True)
# peak at 1

# print(fp.shape)
# print(fp.loc[fp.up_num_purchases == 1, ].shape)
# print(fp.loc[fp.up_num_purchases == 1, ].sample(5, random_state=0))


print(fn.shape)
print(fn.loc[fn.up_num_purchases == 1, ].shape)
print(fn.loc[fn.up_num_purchases == 1, ].sample(5, random_state=0))

pid = 39581
uid = 92665

get_features_pred_by_uid_pid(df, uid, pid)
download_user_order_history(uid, label='fp')

# fp_summary = fp.groupb
# y('aisle').agg({'product_id': ['size', 'nunique']}).reset_index()
# print(fp_summary.head())
# fp_summary.columns = ['aisle', 'fp_size', 'fp_unique_prods']
# fp_summary.sort_values(by='fp_size', ascending=False, inplace=True)
# print(fp_summary)
# sns.displot(x='aisle', data=fp)
# plt.show(block=True)

end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))