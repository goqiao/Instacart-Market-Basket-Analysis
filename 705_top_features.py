import xgboost
import matplotlib.pyplot as plt

bst = xgboost.XGBClassifier()
bst.load_model('data/xgb_model.json')
# print(bst.get_params())

# comparison among different feature importances
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(34, 8))
# xgboost.plot_importance(bst, importance_type='weight', max_num_features=30, show_values=False, ax=ax1)
# xgboost.plot_importance(bst, importance_type='gain', max_num_features=30, show_values=False, ax=ax2)
# xgboost.plot_importance(bst, importance_type='cover', max_num_features=30, show_values=False, ax=ax3)
# ax1.set_xlabel('F Score - Weights')
# ax2.set_xlabel('F Score - Gain')
# ax3.set_xlabel('F Score - Cover')
# ax2.set_ylabel('')
# ax3.set_ylabel('')
# ax1.set_title('')
# ax2.set_title('')
# ax3.set_title('')
# f.suptitle('Top 30 Feature Importance')
# plt.tight_layout()
# plt.show()

# top 30
plt.figure(figsize=(20, 8))
xgboost.plot_importance(bst, importance_type='gain', max_num_features=30, show_values=False)
plt.title('Top 30 Feature Importance in Gain')
plt.xlabel('Gain')
plt.tight_layout()
plt.show()
