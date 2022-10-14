import pandas as pd
import time
from utils import trend_d1, skew

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# parameters:
start_time = time.time()
data_folder = "data"

prior = pd.read_pickle("data/prior_order_details.pickle")
orders = prior[
    ["user_id", "order_id", "order_number", "days_since_prior_order"]
].drop_duplicates()
orders.sort_values(by=["user_id", "order_number"], ascending=True)

products = prior.groupby("order_id").agg({"product_id": "nunique"}).reset_index()
products.columns = ["order_id", "num_products"]

orders = orders.merge(products, how="left")
users_basket_trend = (
    orders.groupby("user_id").agg({"num_products": [trend_d1, skew]}).reset_index()
)
users_basket_trend.columns = [
    "user_id",
    "user_basket_size_trend_d1",
    "user_basket_size_skew",
]


base = pd.read_pickle("data/base.pickle")[["user_id"]].drop_duplicates()
users_basket_trend = base.merge(users_basket_trend, how="left")
users_basket_trend.to_pickle("{}/users_basket_size_trend.pickle".format(data_folder))


end_time = time.time()
print("code using {:.2f} mins".format((end_time - start_time) / 60))
