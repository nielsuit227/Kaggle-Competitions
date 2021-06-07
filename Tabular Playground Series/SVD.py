import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix


# Data
data = pd.read_csv('Data/train.csv', index_col='id')
data['target'] = data['target'].apply(lambda x: int(x[6]))
data.loc[data['target'] == data['target'].max(), 'target'] = 0
x = data.drop('target', axis=1)
y = data['target'].values

# Decompose
x = csr_matrix(pd.get_dummies(x, sparse=True))
x = TruncatedSVD().fit_transform(x)

# Train
logLoss = []
cv = StratifiedKFold(n_splits=3)
for i, (t, v) in enumerate(cv.split(x, y)):
    xt, xv, yt, yv = x[t], x[v], y[t], y[v]

    model = LGBMClassifier()
    model.fit(xt, yt)

    logLoss.append(log_loss(yv, model.predict_proba(xv)))
    print(f1_score(yv, model.predict(xv).reshape((-1)), average=None))
print('Avg. Log Loss: %.4f \u00B1 %.4f' % (np.mean(logLoss), np.std(logLoss)))

