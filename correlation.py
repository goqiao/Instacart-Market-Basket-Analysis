import pandas as pd
import numpy as np
import pickle

corr_thresh = 0.98

data_full_features = pd.read_pickle("data/train_full_features.pickle").sample(
    frac=0.01, random_state=1
)
corr_m = data_full_features.corr().abs()
corr_m.shape
corr_m.to_pickle("data/corr_matrix.pickle")

upper_tri = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(np.bool))
to_drop = [
    column for column in upper_tri.columns if any(upper_tri[column] > corr_thresh)
]

# save high-corr cols
with open("data/high_corr_features.pickle".format(corr_thresh), "wb") as f:
    pickle.dump(to_drop, f)
