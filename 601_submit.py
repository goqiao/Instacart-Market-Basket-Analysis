import pandas as pd
import time
from max_f1 import get_best_prediction
import concurrent.futures
pd.set_option('display.max_columns', None)

start_time = time.time()
data_folder = 'data'


up_pred = pd.read_pickle('data/test_pred_prob_res.pickle')
up = up_pred.groupby('order_id')['product_id'].apply(list).to_frame()
up['pred_proba'] = up_pred.groupby('order_id')['pred_proba'].apply(list)
up.reset_index(inplace=True)


def multi(i):
    # if i%1000==0:
    #     print('{:.3f} min'.format((time.time()-start_time)/60))
    items = up.loc[i,'product_id']
    preds = up.loc[i,'pred_proba']
    ret = get_best_prediction(items, preds, pNone=None)
    return ret


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(multi, range(up.shape[0]))

    up['products'] = [result for result in results]
    up[['order_id', 'products']].to_csv('data/submission.csv', index=False)

    end_time = time.time()
    print('spent {:.2f} mins'.format((end_time - start_time) / 60))
