import os

## 00x prep data
os.system("python 000_make_data.py")


## 1xx create user level features
os.system("python 101_users_basic_features.py")
os.system("python 102_users_order_time.py")
os.system("python 103_users_organic_purchases.py")
os.system("python 104_users_purchase_interval_trend.py")
os.system("python 105_users_basket_size_trend.py")
os.system("python 106_user_kmeans.py")


### 2xx create product level features
os.system("python 201_product_basic.py")
os.system("python 202_product_purchases.py")
os.system("python 203_product_purchase_interval.py")
os.system("python 204_product_order_interval.py")
os.system("python 205_product_word2vec.py")
os.system("python 206_product_substitution_stats.py")
os.system("python 207_product_word2vec_substitute.py")


### 3xx create product level features
os.system("python 301_up_basic.py")
os.system("python 302_up_total_purchase.py")
os.system("python 303_up_purchase_proba.py")
os.system("python 304_up_purchase_interval.py")
os.system("python 305_up_days_not_purchase.py")
os.system("python 306_up_order_interval.py")
os.system("python 307_up_aisle_purchase_trend.py")
os.system("python 308_up_department_purchase_trend.py")
os.system("python 309_up_purchase_interval_trend.py")
os.system("python 310_up_substitution_organic.py")
os.system("python 311_up_substitution_word2vec.py")
os.system("python 312_up_substitution_word2vec_threshold_07.py")
os.system("python 313_up_position_cart.py")
os.system("python 314_up_order_time.py")


### 4xx concat features
os.system("python 401_concat_up.py")


### 5xx train model and predict
os.system("python 500_xgb_cv.py")  # optional
os.system("python 501_xgb.py")
os.system("python 502_xgb_predict.py")
os.system("python 503_test_performance.py")  # optional


### 6xx output
os.system("python 601_submit.py")
