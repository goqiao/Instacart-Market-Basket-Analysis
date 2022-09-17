import pandas as pd
import time

trim_num_products = False
low_b = 1
data_folder = 'data'
test_res = pd.read_pickle('{}/test_prediction_res.pickle'.format(data_folder))

test_res['product_id'] = test_res['product_id'].astype('str')
submit = test_res.loc[test_res['reordered'] == 1].groupby('order_id')['product_id'].apply(
    lambda x: ' '.join(set(x))).to_frame()

if trim_num_products:
    # if the predicted num reordered products less than low_b, predict as no reordering
    submit['num_reordered_prods'] = test_res.loc[test_res['reordered'] == 1].groupby('order_id')['product_id'].nunique()
    submit = submit.loc[submit['num_reordered_prods'] > low_b, ].drop(columns=['num_reordered_prods'])

sample_submit = pd.read_csv('{}/sample_submission.csv'.format(data_folder))
submit.reset_index(inplace=True); submit.columns = sample_submit.columns
submit_final = sample_submit[['order_id']].merge(submit, on='order_id', how='left').fillna('None')

print(sum(submit_final['products'].apply(lambda x: x == 'None')))
submit_final.to_csv('{}/my_submission.csv'.format(data_folder), index=False)
