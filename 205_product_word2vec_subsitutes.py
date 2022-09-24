import pandas as pd
from gensim.models import Word2Vec
import time
import concurrent.futures
from itertools import chain

"""
use results of word2vec algorithm to find the most similar (alternative) products for each prodcut. 
To increase accuracy, also filter on:
- products and their alternatives have to be in the same department
- word2vec similarity score > 0.5

takes ~90 mins to run
"""

model = Word2Vec.load('data/word2vec.model')
products = pd.read_pickle('data/products.pickle')[['product_id', 'product_name', 'aisle', 'department']].set_index('product_id')
products.index = products.index.astype('str')
vocab_len = len(model.wv)

start_time = time.time()
threshold = 0.5

def multi(pid, department):
    # print(pid, department)
    tmp = []
    if pid in model.wv.key_to_index:
        # print(pid)
        for pair in model.wv.most_similar(pid, topn=vocab_len):
            sub_pid, sub_sim_score = pair[0], pair[1]
            sub_department = products.at[sub_pid, 'department']
            # print(sub_department, department)
            if sub_department == department and sub_sim_score > threshold:
                # print([pid, sub_pid, sub_sim_score])
                tmp.append([pid, sub_pid, sub_sim_score])
        return tmp
    else:
        return []


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(multi, products.index.values, products.department.values)
    # res = [result for result in results]

    # results.to_pickle('data/multi_processed_results.pickle')
    res = list(chain.from_iterable(results))

    res = pd.DataFrame(res, columns=['product_id', 'substitute_id', 'similarity_score'])
    res[['product_id','substitute_id']] = res[['product_id','substitute_id']].astype('int')
    res.to_pickle('data/word2vec_substitute.pickle')

    end_time = time.time()
    print('spent {:.2f} mins'.format((end_time - start_time) / 60))