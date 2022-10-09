import pandas as pd
import numpy as np
from utils import read_data
import gensim
from sklearn.decomposition import PCA
import time


"""
capture product relationships using word2vec algorithm, where each order is a sentence and the products inside the 
order as words. This is a potential algorithm to find alternative products. 
"""
start_time = time.time()
data_folder = 'data'
nrows = None
prior = read_data(data_folder=data_folder, nrows=nrows, read_prior=True)
train = read_data(data_folder=data_folder, nrows=nrows, read_train=True)

train['product_id'] = train['product_id'].astype('str')
prior['product_id'] = prior['product_id'].astype('str')

train_products = train.groupby('order_id')['product_id'].apply(lambda x: x.tolist()).reset_index()
prior_products = prior.groupby('order_id')['product_id'].apply(lambda x: x.tolist()).reset_index()

products = pd.concat([train_products, prior_products])
max_num_prods = len(max(products['product_id'], key=len))

# train Word2Vec with sentence as orders of products, and each product id as word
# min_count = 2 to exclude one-product orders
# window=max_num_prods because sequence of products added to cart are independent
# vector_size=100, the dimensions of word vectors, can be reduced to PCA later
model = gensim.models.Word2Vec(sentences=products['product_id'], min_count=2, vector_size=100, window=max_num_prods, workers=8)

# get word embedding
prods_embedding = pd.DataFrame(index=model.wv.key_to_index.keys(),
                              data=model.wv.vectors).rename_axis('product_id')


# to choose how many components we need to retain 80% variance
full_variance_explained = PCA().fit(prods_embedding.values).explained_variance_ratio_
n_components = np.argmax(np.cumsum(full_variance_explained) >= 0.80) + 1

# pca project data
pc = PCA(n_components, random_state=0).fit_transform(prods_embedding.values)


# update 100 dimensions with reduced dimensions
product_embedding = pd.DataFrame(index=prods_embedding.index, data=pc, columns=np.arange(pc.shape[1])).add_prefix('p_embedding_').reset_index()

# convert product_id type back to int
product_embedding.product_id = product_embedding.product_id.astype(int)
product_embedding.to_pickle('data/product_embedding.pickle')


end_time = time.time()
time_spent = (end_time - start_time) / 60
print('spent {:.2f} mins'.format(time_spent))


