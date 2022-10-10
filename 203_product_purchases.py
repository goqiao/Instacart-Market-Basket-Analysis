import pandas as pd
import time
from utils import q20, q80

"""
On average, how many times each product tend to be purchased?
"""

# parameters:
start_time = time.time()
data_folder = "data"


base_prods = (
    pd.read_pickle("{}/base.pickle".format(data_folder))["product_id"]
    .to_frame()
    .drop_duplicates()
)
prior_order_details = pd.read_pickle(
    "{}/prior_order_details.pickle".format(data_folder)
)

prod_users = (
    prior_order_details.groupby(["product_id", "user_id"])
    .agg({"order_number": "size"})
    .rename(columns={"order_number": "num_purchases_per_user"})
    .reset_index()
)


print("cal stats..")
prod = (
    prod_users.groupby(["product_id"])
    .agg({"num_purchases_per_user": ["mean", "std", "median", "max", "min", q20, q80]})
    .reset_index()
)
prod.columns = [
    "product_id",
    "p_num_purchases_per_user_mean",
    "p_num_purchases_per_user_std",
    "p_num_purchases_per_user_median",
    "p_num_purchases_per_user_max",
    "p_num_purchases_per_user_min",
    "p_num_purchases_per_user_q20",
    "p_num_purchases_per_user_q80",
]
print(prod.head())

base_prods = base_prods.merge(prod, how="left")
base_prods.to_pickle("{}/products_purchases_features.pickle".format(data_folder))

end_time = time.time()
print("code using {:.2f} mins".format((end_time - start_time) / 60))
