import pandas as pd
import time


"""
number of purchases each user on substitution products. 
substitution is defined as similar products inferred by word2vec algorithm
"""

# parameters:
start_time = time.time()
data_folder = "data"
nrows = None
two_way = True


substitute = pd.read_pickle("data/word2vec_substitute.pickle")
substitute = substitute.loc[
    substitute["similarity_score"] > 0.8, ["product_id", "substitute_id"]
]

base = pd.read_pickle("data/base.pickle")[["user_id", "product_id"]]


user_substitute = base.merge(substitute, on=["product_id"], how="inner")
user_substitute.set_index(["user_id", "substitute_id"], inplace=True)
# <user_id, product_id, substitute_id>
del base, substitute


up_purchase = pd.read_pickle("data/up_agg.pickle")[
    ["user_id", "product_id", "up_num_purchases"]
]
up_purchase.columns = [
    "user_id",
    "substitute_id",
    "up_word2vec_substitute_num_purchases",
]
up_purchase.set_index(["user_id", "substitute_id"], inplace=True)
# user_substitute = user_substitute.merge(up_purchase,  left_on=['user_id', 'substitute_id']
#                                         , right_on=['user_id', 'substitute_id'], how='left')

user_substitute = user_substitute.join(
    up_purchase, on=["user_id", "substitute_id"], how="left"
)
del up_purchase


# r5
up_purchases_r5 = pd.read_pickle("data/up_purchase_r5.pickle")[
    ["user_id", "product_id", "up_num_purchases_r5"]
]
up_purchases_r5.columns = [
    "user_id",
    "substitute_id",
    "up_word2vec_substitute_num_purchases_r5",
]
up_purchases_r5.set_index(["user_id", "substitute_id"], inplace=True)
# user_substitute = user_substitute.merge(up_purchases_r5,  left_on=['user_id','substitute_id']
#                                         , right_on=['user_id', 'substitute_id'], how='left')


user_substitute = user_substitute.join(
    up_purchases_r5, on=["user_id", "substitute_id"], how="left"
)
user_substitute.fillna(0, inplace=True)

user_substitute = (
    user_substitute.groupby(["user_id", "product_id"])
    .agg(
        {
            "up_word2vec_substitute_num_purchases": "sum",
            "up_word2vec_substitute_num_purchases_r5": "sum",
        }
    )
    .reset_index()
)


user_substitute.to_pickle(
    "{}/up_word2vec_substitute_purchase.pickle".format(data_folder)
)


end_time = time.time()
print("code using {:.2f} mins".format((end_time - start_time) / 60))
