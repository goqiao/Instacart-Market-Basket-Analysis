import pandas as pd
import time
from utils import trend_d1, skewness


# parameters:
start_time = time.time()
data_folder = "data"

up_purchase_time = pd.read_pickle(
    "{}/days_since_last_purchase.pickle".format(data_folder)
).reset_index(drop=True)

key = ["user_id", "product_id"]
up_purchase_time = up_purchase_time.loc[
    ~up_purchase_time._up_days_since_last_purchase.isnull()
]
up_time = up_purchase_time.groupby(key).agg(
    {"_up_days_since_last_purchase": [list, "mean", trend_d1, skewness]}
)
up_time.columns = [
    "up_purchase_interval",
    "up_purchase_interval_mean",
    "up_purchase_interval_trend_d1",
    "up_purchase_interval_skewness",
]

up_time[["up_purchase_interval_trend_d1", "up_purchase_interval_skewness"]].to_pickle(
    "data/up_purchase_interval_trend.pickle"
)


end_time = time.time()
print("code using {:.2f} mins".format((end_time - start_time) / 60))
