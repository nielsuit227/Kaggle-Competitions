from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import os


# Data
data = pd.read_csv('Data/train.csv', index_col='id')
Y = data['target'].apply(lambda x: int(x[6]))
Y.loc[Y == Y.max()] = 0
X = data.drop('target', axis=1)

# Scaling
scaler = StandardScaler()
keys = X.keys()
X = scaler.fit_transform(X)

# Params
lgbm_params = {'verbose': -1, 'force_row_wise': True, 'colsample_bytree': 0.36520817192963606,
               'min_child_samples': 339, 'min_child_weight': 0.15630682310366817, 'num_leaves': 23,
               'reg_alpha': 0.49834671939652775, 'reg_lambda': 0.07794142881381572,
               'subsample': 0.36197237488107925, 'n_estimators': 450}
cb_params = {'allow_writing_files': False, 'verbose': 0, 'depth': 2, 'grow_policy': 'Lossguide',
             'l2_leaf_reg': 9.875651980786989, 'learning_rate': 0.15540580027656792, 'loss_function': 'MultiClass',
             'min_data_in_leaf': 236}
knn_params = {'leaf_size': 36, 'n_neighbors': 491, 'weights': 'uniform'}

# Model prep
models = [
    ('linear', RidgeClassifier()),
    ('elastic', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)),
    ('knn', KNeighborsClassifier(**knn_params)),
    # ('svm', SVC()),
    ('rf', RandomForestClassifier()),
    ('lgbm', LGBMClassifier(**lgbm_params)),
    ('cb', CatBoostClassifier(**cb_params)),
]
# final_estimator = LogisticRegression(max_iter=500)
final_estimator = RidgeClassifier()

# Iterate
logLoss = []
splits = 3
cv = StratifiedKFold(n_splits=splits)
for i, (ti, vi) in enumerate(cv.split(X, Y)):
    print('Fold %i / %i' % (i + 1, splits))
    # Split
    Xt, Xv, Yt, Yv = X[ti], X[vi], Y[ti], Y[vi]

    # Train
    model = StackingClassifier(estimators=models, n_jobs=6, final_estimator=final_estimator)
    model.fit(Xt, Yt)

    # Score
    Pv = model.predict_proba(Xv)
    logLoss.append(log_loss(Yv, Pv))
print('Avg. log loss: %.2f \u00B1 %.2f' % (np.mean(logLoss), np.std(logLoss)))

# Produce Prediction Results
test = pd.read_csv('Data/test.csv', index_col='id')
Xt = scaler.transform(test[keys])

model = StackingClassifier(estimators=models, n_jobs=8, final_estimator=final_estimator)
model.fit(X, Y)

Prediction = pd.DataFrame(data=model.predict_proba(Xt),
                          columns=['Class_4', 'Class_1', 'Class_2', 'Class_3'],
                          index=np.linspace(100000, 100000 + len(Xt) - 1, len(Xt)).astype('int'))
version = len([x for x in os.listdir('Data') if 'Stacking' in x])
Prediction.to_csv('Data/Stacking_vi.csv' % version, index_label='id')
