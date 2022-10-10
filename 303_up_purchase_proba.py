import pandas as pd
import numpy as np
import time

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

"""
After users make the first purchase of a product, they can choose to purchase the product again when they place 
orders. What's the probability that they purchases the same product?
"""

# parameters:
start_time = time.time()
data_folder = "data"

prior_order_details = pd.read_pickle(
    "{}/prior_order_details.pickle".format(data_folder)
)

key = ["user_id", "product_id"]
_up = (
    prior_order_details.groupby(key)
    .agg({"order_number": ["min", "size"]})
    .reset_index()
)
_up.columns = key + ["up_first_order", "up_num_purchases"]

chances = prior_order_details[key + ["user_max_order"]].drop_duplicates().merge(_up)
chances["num_up_purchases_chances"] = (
    chances["user_max_order"] - chances["up_first_order"]
)

chances["up_reorder_tendency_proba"] = (chances["up_num_purchases"] - 1) / chances[
    "num_up_purchases_chances"
]
chances.replace({float("-inf"): np.nan, float("inf"): np.nan}, inplace=True)

chances[["user_id", "product_id", "up_reorder_tendency_proba"]].to_pickle(
    "{}/up_reorder_tendency_proba.pickle".format(data_folder)
)

## recent 5 orders

T = 5
prior_order_details = prior_order_details.loc[
    prior_order_details["last_nth_order"].isin(np.arange(1, T + 1))
]
key = ["user_id", "product_id"]
_up_r5 = (
    prior_order_details.groupby(key)
    .agg({"order_number": ["min", "size"]})
    .reset_index()
)
_up_r5.columns = key + ["up_first_order_r5", "up_num_purchases_r5"]

chances_r5 = (
    prior_order_details[key + ["user_max_order"]].drop_duplicates().merge(_up_r5)
)

chances_r5["num_up_purchases_chances_r5"] = (
    chances_r5["user_max_order"] - chances_r5["up_first_order_r5"]
)


chances_r5["up_reorder_tendency_proba_r5"] = (
    chances_r5["up_num_purchases_r5"] - 1
) / chances_r5["num_up_purchases_chances_r5"]
chances_r5.replace({float("-inf"): np.nan, float("inf"): np.nan}, inplace=True)

chances_r5[["user_id", "product_id", "up_reorder_tendency_proba_r5"]].to_pickle(
    "{}/up_reorder_tendency_proba_r5.pickle".format(data_folder)
)

end_time = time.time()
print("code using {:.2f} mins".format((end_time - start_time) / 60))
