import pandas as pd
import time
from _max_f1 import get_best_prediction
import concurrent.futures
pd.set_option('display.max_columns', None)
from _threshold_exploration import f1_maximization

start_time = time.time()
data_folder = 'data'
# find threshold
None_pred = pd.read_pickle('data/test_None_prediction_res.pickle')[['user_id', 'pred_None_proba']]
up_pred = pd.read_pickle('data/test_prediction_res.pickle')

up_pred = up_pred.merge(None_pred, on='user_id', how='left')
up = up_pred.groupby('order_id')['product_id'].apply(list).to_frame()
up['pred_proba'] = up_pred.groupby('order_id')['pred_proba'].apply(list)
up['pred_proba_None'] = up_pred.groupby('order_id')['pred_None_proba'].mean()
up.reset_index(inplace=True)
print(up.head())
print(2)

def multi(i):
    # if i%1000==0:
    #     print('{:.3f} min'.format((time.time()-start_time)/60))
    items = up.loc[i,'product_id']
    preds = up.loc[i,'pred_proba']
    pNone = up.loc[i,'pred_proba_None']
    ret = get_best_prediction(items, preds, pNone)
    return ret


# pool = mp.Pool()
# callback = pool.map(multi, range(up.shape[0]))
# up['products'] = callback

# for loop
# products = []
# for i in range(up.shape[0]):
#     products.append(multi(i))

if __name__ == '__main__':
    # pool = mp.Pool()
    # callback = pool.map(multi, range(1000))
    # print('callback', callback)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(multi, range(up.shape[0]))
    up['products'] = [result for result in results]
    t2 = (time.time() - start_time)/60
    print(t2)


    # TODO: how to find individual threshold as in
    # https://asagar60.medium.com/instacart-market-basket-analysis-part-2-fe-modelling-1dc02c2b028b

    up.to_pickle('data/combined_prediction_res.pickle')
    up[['order_id', 'products']].to_csv('data/my_submission_faron.csv', index=False)

    end_time = time.time()
    print('code using {:.2f} mins'.format((end_time - start_time) / 60))
# max_f1, best_threshold_f1 = f1_maximization(up_pred['pred_proba'], up_pred['reordered'])


# # test set
# None_pred = pd.read_pickle('data/test_None_prediction_res.pickle')
# up_pred = pd.read_pickle('data/test_prediction_res.pickle')
#
# up_pred = up_pred.merge(None_pred, on='user_id', how='left')
# print(up_pred.head())
# up_pred['pred_proba'] = (1 - up_pred['pred_None_proba']) * up_pred['pred_proba']
# print(up_pred.head())
#
#
# up_pred['reordered'] = (up_pred['pred_proba'] > best_threshold_f1).astype(int)
# up_pred[['order_id', 'product_id', 'reordered']].to_pickle('{}/combined_test_prediction_res.pickle'.format(data_folder))
# print(up_pred.shape)
