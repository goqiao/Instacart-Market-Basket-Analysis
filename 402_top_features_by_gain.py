import pandas as pd
import pickle

"""
Select top 70% features and group them by feature types: user, product and user x product features
"""

fi = pd.read_csv('data/up_feature_importance_drop_corr.csv') # data is sorted with most important features at top
num_features = fi.shape[0]

top_percent = 0.4
mid_percent = 0.3

def fi_group(rank):
    if rank < num_features * top_percent:
        return 'high_fi'
    elif rank < num_features * (mid_percent + top_percent):
        return 'middle_fi'
    else:
        return 'low_fi'


fi['importance_group'] = fi.reset_index()['index'].apply(lambda x: fi_group(x))
fi['feature_group'] = fi['features'].str.extract(r'([a-z]+)_.*')

# handle inconsistency in feature names
fi['feature_group'] = fi['feature_group'].str.replace('users', 'user', regex=False)
fi['feature_group'] = fi['feature_group'].str.replace('uo', 'user', regex=False)

# group based on feature_group(p, up, user) and importance tier(high, middle, low)
groups = fi[['features', 'feature_group', 'importance_group']].groupby(['feature_group', 'importance_group'])[
    'features'].apply(lambda x: x.to_list()).reset_index()
groups['fi_group'] = groups['feature_group'] + '_' + groups['importance_group']
groups[['fi_group', 'features']].to_pickle('data/fi_group_new.pickle')

# output top and mid features as list
groups  = groups[['fi_group', 'features']].set_index('fi_group')

p_high = groups.loc['p_high_fi'].values[0]
p_middle = groups.loc['p_middle_fi'].values[0]

user_high = groups.loc['user_high_fi'].values[0]
user_middle = groups.loc['user_middle_fi'].values[0]

up_high = groups.loc['up_high_fi'].values[0]
up_middle = groups.loc['up_middle_fi'].values[0]


selected_cols = user_high + user_middle + up_high + up_middle + p_high + p_middle
print('{} features selected'.format(len(selected_cols)))

with open('data/selected_top_gain_features_drop_corr.pickle', 'wb') as f:
    pickle.dump(selected_cols, f)


