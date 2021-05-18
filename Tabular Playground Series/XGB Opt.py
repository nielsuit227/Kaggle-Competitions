from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedKFold, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
import joblib


# Model
model = KNeighborsClassifier()
# model = CatBoostClassifier(n_estimators=1000, verbose=0, allow_writing_files=False)

# Data
data = pd.read_csv('Data/train.csv', index_col='id')
# Y = (data['target'] == 'Class_1') * 1
Y = data['target'].apply(lambda x: int(x[6]))
Y.loc[Y == Y.max()] = 0
X = data.drop('target', axis=1)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Parameter distribution
if model.__module__ == 'catboost.core':
    params = {
        'loss_function': ['MultiClass'],
        'learning_rate': uniform(0, 1),
        'l2_leaf_reg': uniform(0, 10),
        'depth': randint(1, 10),
        'min_data_in_leaf': randint(50, 500),
        'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
    }
elif model.__module__ == 'lightgbm.sklearn':
    params = {
        'num_leaves': randint(10, 150),
        'min_child_samples': randint(50, 500),
        'min_child_weight': uniform(0, 1),
        'subsample': uniform(0, 1),
        'colsample_bytree': uniform(0, 1),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1),
    }
elif type(model).__name__ == 'KNeighborsClassifier':
    params = {
        'n_neighbors': randint(5, 500),
        'weights': ['uniform', 'distance'],
        'leaf_size': randint(30, 100),
    }


# Hyperparameter optimization
cv = StratifiedKFold(n_splits=3)
optimizer = HalvingRandomSearchCV(estimator=model,
                                  param_distributions=params,
                                  n_candidates=200,
                                  cv=cv,
                                  verbose=1,
                                  n_jobs=6,
                                  scoring='neg_log_loss',
                                  # resource='n_estimators',
                                  # min_resources=100,
                                  # max_resources=2500,
                                  resource='n_samples',
                                  min_resources=10000,
                                  max_resources=100000
                                  )
optimizer.fit(X, Y)
print(optimizer.best_params_)
joblib.dump(optimizer.best_estimator_, 'Models/%s.joblib' % type(model).__name__)
