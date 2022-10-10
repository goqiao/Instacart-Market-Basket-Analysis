import pandas as pd
import time


start_time = time.time()
data_folder = "data"
nrows = None

up_purchase_time = pd.read_pickle(
    "{}/days_since_last_purchase.pickle".format(data_folder)
)

up_purchase_time = up_purchase_time.drop_duplicates(
    ["user_id", "product_id"], keep="last"
)[["user_id", "product_id", "nth_day_since_customer"]]

orders = pd.read_pickle("{}/orders.pickle".format(data_folder))
users_age_last_orders = orders.loc[
    orders["eval_set"] != "prior", ["user_id", "days_since_prior_order"]
]
users_age_before_last = (
    up_purchase_time.groupby(["user_id"])
    .agg({"nth_day_since_customer": "max"})
    .rename({"nth_day_since_customer": "user_age_before_last"}, axis=1)
    .reset_index()
)
users_age = pd.merge(
    users_age_last_orders, users_age_before_last, on="user_id", how="inner"
)
users_age["user_age"] = (
    users_age["user_age_before_last"] + users_age["days_since_prior_order"]
)

up_purchase_time = up_purchase_time.merge(users_age, on="user_id")
up_purchase_time["up_num_days_not_purchase"] = (
    up_purchase_time["user_age"] - up_purchase_time["nth_day_since_customer"]
)

up_purchase_time[["user_id", "product_id", "up_num_days_not_purchase"]].to_pickle(
    "{}/up_days_not_purchase.pickle".format(data_folder)
)


end_time = time.time()
print("code using {:.2f} mins".format((end_time - start_time) / 60))
