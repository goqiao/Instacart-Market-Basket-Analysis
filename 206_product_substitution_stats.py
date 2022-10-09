import pandas as pd
import time
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

"""
use data file generated in 205_product_word2vec_substitute.py
create similarity score buckets. For each product, count number of substitute it has in each bucket
"""

start_time = time.time()
data_folder = 'data'

sub = pd.read_pickle('data/word2vec_substitute.pickle')
sub_stats1 = sub.groupby('product_id').agg({'substitute_id':'nunique',
                                           'similarity_score':'max'})
sub_stats1.columns = ['p_num_substitute', 'p_max_similarity_score']


sub['similarity_score_bucket'] = pd.cut(sub['similarity_score'], right=False, bins=[.5, .6, .7, .8, .9, 1])
sub_stats2 = pd.crosstab(index=sub['product_id'], columns=sub['similarity_score_bucket']).add_prefix('p_num_products_sim_score')

sub_stats = pd.concat([sub_stats1, sub_stats2], axis=1)
# rename as special characters, '[, )' are not supported in xgboost
sub_stats.rename(columns={'p_num_products_sim_score[0.5, 0.6)': 'p_num_products_sim_score_0.5_0.6',
                      'p_num_products_sim_score[0.6, 0.7)': 'p_num_products_sim_score_0.6_0.7',
                      'p_num_products_sim_score[0.7, 0.8)': 'p_num_products_sim_score_0.7_0.8',
                      'p_num_products_sim_score[0.8, 0.9)': 'p_num_products_sim_score_0.8_0.9',
                      'p_num_products_sim_score[0.9, 1.0)': 'p_num_products_sim_score_0.9_1.0'
                      }, inplace=True)

sub_stats.reset_index().to_pickle('data/product_sub_stats.pickle')


end_time = time.time()
print('code using {:.2f} mins'.format((end_time - start_time) / 60))

