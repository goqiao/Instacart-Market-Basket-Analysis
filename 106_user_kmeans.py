import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# get user basic features
users_basic = pd.read_pickle("data/users_features.pickle").drop(
    ["user_days_not_purchase", "user_next_order_readiness"], axis=1
)

# get order details
prior_order_details = pd.read_pickle("data/prior_order_details.pickle")[
    ["order_id", "user_id", "product_id"]
]
products = pd.read_pickle("data/products.pickle")[["product_id", "aisle"]]
prior_order_details = prior_order_details.merge(products, how="left")

# count num purchases on each aisle for every user
users_aisles_cnts = pd.crosstab(
    index=prior_order_details["user_id"], columns=prior_order_details["aisle"]
).reset_index()

# merge features
users = users_basic.merge(users_aisles_cnts, on="user_id", how="inner")
print("user feature shape:")
print(users.shape)

# scaling
train = users.drop("user_id", axis=1)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train.values)

# train Kmeans
kmeans = KMeans(n_clusters=4, random_state=4).fit(train_scaled)

# get cluster center on each feature dimension
cluster_res = kmeans.predict(train_scaled)
users["user_cluster"] = cluster_res

# compare the centers of each cluster
# cluster_centers = pd.DataFrame(kmeans.cluster_centers_.transpose(), index=users.columns)
# color_list = ['g', 'b', 'orange', 'r']
# cluster_centers.plot(kind='bar', color=color_list, figsize=(25, 15))
# plt.show()

# plot cluster distribution
users["user_cluster"].value_counts(dropna=False).plot(kind="bar")
plt.show()

user_cluster_res = pd.get_dummies(
    users[["user_id", "user_cluster"]], prefix="user_cluster", columns=["user_cluster"]
)
print(user_cluster_res.shape)
user_cluster_res.to_pickle("data/user_kmeans_cluster.pickle")
