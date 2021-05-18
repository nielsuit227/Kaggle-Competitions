from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# Data
data = pd.read_csv('Data/train.csv', index_col='id')
# Y = (data['target'] == 'Class_1') * 1
Y = data['target'].apply(lambda x: int(x[6]))
Y.loc[Y == Y.max()] = 0
X = data.drop('target', axis=1)

# Params
# params = {
#     'verbose': -1,
#     'force_row_wise': True,
#     'colsample_bytree': 0.36520817192963606,
#     'min_child_samples': 339,
#     'min_child_weight': 0.15630682310366817,
#     'num_leaves': 23,
#     'reg_alpha': 0.49834671939652775,
#     'reg_lambda': 0.07794142881381572,
#     'subsample': 0.36197237488107925,
#     'n_estimators': 450}
# params = {'allow_writing_files': False, 'verbose': 0, 'depth': 2, 'grow_policy': 'Lossguide', 'l2_leaf_reg': 9.875651980786989, 'learning_rate': 0.15540580027656792, 'loss_function': 'MultiClass', 'min_data_in_leaf': 236}
params = {'leaf_size': 36, 'n_neighbors': 491, 'weights': 'uniform'}
# Scaling
scaler = StandardScaler()
keys = X.keys()
X = scaler.fit_transform(X)

# Iterate
logLoss = []
splits = 3
cv = StratifiedKFold(n_splits=splits)
for i, (ti, vi) in enumerate(cv.split(X, Y)):
    print('Fold %i / %i' % (i + 1, splits))
    # Split
    Xt, Xv, Yt, Yv = X[ti], X[vi], Y[ti], Y[vi]

    # Train
    model = KNeighborsClassifier(**params)
    # model = CatBoostClassifier(**params)
    # model = LGBMClassifier(**params)
    model.fit(Xt, Yt)

    # Score
    Pv = model.predict_proba(Xv)
    logLoss.append(log_loss(Yv, Pv))
print('Avg. log loss: %.4f \u00B1 %.2f' % (np.mean(logLoss), np.std(logLoss)))

# Make Test Predictions
test = pd.read_csv('Data/test.csv', index_col='id')
Xt = scaler.transform(test[keys])

model = CatBoostClassifier(**params)
model.fit(X, Y)

Prediction = pd.DataFrame(data=model.predict_proba(Xt),
                          columns=['Class_4', 'Class_1', 'Class_2', 'Class_3'],
                          index=np.linspace(100000, 100000 + len(Xt) - 1, len(Xt)).astype('int'))
Prediction.to_csv('Data/Base_submission_v1.csv', index_label='id')

