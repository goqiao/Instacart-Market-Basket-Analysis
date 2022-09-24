import pandas as pd
import time
from max_f1 import get_best_prediction
import concurrent.futures
from utils import print_eval_metrics
pd.set_option('display.max_columns', None)

start_time = time.time()
data_folder = 'data'

up_pred_prob = pd.read_pickle('data/val_pred_prob_res.pickle')[['order_id', 'product_id', 'pred_proba', 'reordered']]
up_prob = up_pred_prob.groupby('order_id')['product_id'].apply(list).to_frame()
up_prob['pred_proba'] = up_pred_prob.groupby('order_id')['pred_proba'].apply(list)
up_prob.reset_index(inplace=True)


def multi(i):
    # if i%1000==0:
    #     print('{:.3f} min'.format((time.time()-start_time)/60))
    items = up_prob.loc[i, 'product_id']
    preds = up_prob.loc[i, 'pred_proba']
    # pNone = up.loc[i,'pred_proba_None']
    ret = get_best_prediction(items, preds, pNone=None)
    return ret


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(multi, range(up_prob.shape[0]))
    up_prob['products'] = [result for result in results]

    val_pred_reorder_long = pd.melt(pd.concat([up_prob['order_id'], up_prob['products'].str.split(expand=True)], axis=1),
                                    id_vars='order_id', value_name='product_id').drop('variable', axis=1)

    val_pred_reorder_long = val_pred_reorder_long.loc[val_pred_reorder_long.product_id.apply(lambda x: x is not None)
                                                      & (val_pred_reorder_long.product_id != 'None')]
    val_pred_reorder_long['product_id'] = val_pred_reorder_long['product_id'].astype('uint16')
    val_pred_reorder_long['pred_y'] = 1

    val_pred = up_pred_prob.merge(val_pred_reorder_long, on=['order_id', 'product_id'], how='left')
    val_pred['pred_y'].fillna(0, inplace=True)
    print_eval_metrics(val_pred['reordered'], val_pred['pred_y'])

    val_pred.to_pickle('data/val_pred_res.pickle')

    end_time = time.time()
    print('code using {:.2f} mins'.format((end_time - start_time) / 60))