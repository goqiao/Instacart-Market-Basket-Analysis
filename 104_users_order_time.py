import time
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


start_time = time.time()
data_folder = "data"

orders = pd.read_pickle("data/orders.pickle")[
    ["order_id", "user_id", "order_dow", "part_of_day"]
]
prior_order_details = pd.read_pickle("data/prior_order_details.pickle")[
    ["order_id", "user_id", "product_id"]
]

prior_order_details = prior_order_details.merge(
    orders, on=["order_id", "user_id"], how="left"
)

# cal metrics
dow = pd.crosstab(
    prior_order_details["user_id"], prior_order_details["order_dow"]
).add_prefix("user_purchases_dow_")
dow_norm = pd.crosstab(
    prior_order_details["user_id"], prior_order_details["order_dow"], normalize="index"
).add_prefix("user_norm_purchases_dow_")

part_of_day = pd.crosstab(
    prior_order_details["user_id"], prior_order_details["part_of_day"]
).add_prefix("user_purchases_pod_")
part_of_day_norm = pd.crosstab(
    prior_order_details["user_id"],
    prior_order_details["part_of_day"],
    normalize="index",
).add_prefix("user_norm_purchases_pod_")

users_order_time = pd.concat(
    [dow, dow_norm, part_of_day, part_of_day_norm], axis=1
).reset_index()

users_order_time.to_pickle("data/users_order_time.pickle")


end_time = time.time()
print("code using {:.2f} mins".format((end_time - start_time) / 60))
