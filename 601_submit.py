import pandas as pd
import time
from max_f1 import get_best_prediction
import concurrent.futures
pd.set_option('display.max_columns', None)

start_time = time.time()
data_folder = 'data'
# find threshold

up_pred = pd.read_pickle('data/test_pred_prob_res.pickle')
up = up_pred.groupby('order_id')['product_id'].apply(list).to_frame()
up['pred_proba'] = up_pred.groupby('order_id')['pred_proba'].apply(list)
up.reset_index(inplace=True)


def multi(i):
    # if i%1000==0:
    #     print('{:.3f} min'.format((time.time()-start_time)/60))
    items = up.loc[i,'product_id']
    preds = up.loc[i,'pred_proba']
    # pNone = up.loc[i,'pred_proba_None']
    ret = get_best_prediction(items, preds, pNone=None)
    return ret


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(multi, range(up.shape[0]))

    up['products'] = [result for result in results]
    # TODO: how to find individual threshold as in
    # https://asagar60.medium.com/instacart-market-basket-analysis-part-2-fe-modelling-1dc02c2b028b
    # up.to_pickle('data/combined_prediction_res.pickle')
    up[['order_id', 'products']].to_csv('data/submission.csv', index=False)

    end_time = time.time()
    print('spent {:.2f} mins'.format((end_time - start_time) / 60))
