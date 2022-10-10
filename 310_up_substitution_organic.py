import pandas as pd
import time


# parameters:
start_time = time.time()
data_folder = "data"
nrows = None
two_way = True

substitute = pd.read_pickle("data/products_organic_substitution.pickle")[
    ["product_id", "substitute_id"]
]
if two_way:
    substitute_rev = pd.read_pickle("data/products_organic_substitution.pickle")[
        ["product_id", "substitute_id"]
    ].rename(columns={"product_id": "substitute_id", "substitute_id": "product_id"})
    substitute = pd.concat([substitute, substitute_rev], axis=0)
base = pd.read_pickle("data/base.pickle")[["user_id", "product_id"]]

user_substitute = base.merge(substitute, on=["product_id"], how="left")
# <user_id, product_id, substitute_id>
print(
    user_substitute.loc[
        (user_substitute.user_id == 91812) & (user_substitute.product_id == 31915)
    ]
)
print(user_substitute.loc[(user_substitute.user_id == 91812)])

up_purchase = pd.read_pickle("data/up_agg.pickle")[
    ["user_id", "product_id", "up_num_purchases"]
]
up_purchase.columns = [
    "user_id",
    "substitute_id",
    "up_organic_substitute_num_purchases",
]
user_substitute = user_substitute.merge(
    up_purchase,
    left_on=["user_id", "substitute_id"],
    right_on=["user_id", "substitute_id"],
    how="left",
)

up_purchases_r5 = pd.read_pickle("data/up_purchase_r5.pickle")[
    ["user_id", "product_id", "up_num_purchases_r5"]
]
up_purchases_r5.columns = [
    "user_id",
    "substitute_id",
    "up_organic_substitute_num_purchases_r5",
]
user_substitute = user_substitute.merge(
    up_purchases_r5,
    left_on=["user_id", "substitute_id"],
    right_on=["user_id", "substitute_id"],
    how="left",
)
user_substitute.fillna(0, inplace=True)
user_substitute = (
    user_substitute.groupby(["user_id", "product_id"])
    .agg(
        {
            "up_organic_substitute_num_purchases": "sum",
            "up_organic_substitute_num_purchases_r5": "sum",
        }
    )
    .reset_index()
)

print(
    user_substitute[
        [
            "up_organic_substitute_num_purchases",
            "up_organic_substitute_num_purchases_r5",
        ]
    ].describe()
)
print(user_substitute.isnull().sum())
user_substitute.to_pickle(
    "{}/up_organic_substitute_purchase.pickle".format(data_folder)
)


end_time = time.time()
print("code using {:.2f} mins".format((end_time - start_time) / 60))
