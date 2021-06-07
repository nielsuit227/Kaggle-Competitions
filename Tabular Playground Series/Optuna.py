import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


def objective(trial):
    global X, Y
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 250, 2500, step=250),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-1, log=True),
        'random_strength': trial.suggest_int('random_strength', 1, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 1e-1, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 500),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'verbose': 0,
        'loss_function': 'MultiClass',
        'allow_writing_files': False,
    }
    score = []
    cv = StratifiedKFold(n_splits=3)
    for t, v in cv.split(X, Y):
        Xt, Xv, Yt, Yv = X.iloc[t], X.iloc[v], Y.iloc[t], Y.iloc[v]
        model = CatBoostClassifier(**params)
        model.fit(Xt, Yt)
        score.append(log_loss(Yv, model.predict_proba(Xv)))
    return np.mean(score)


# Data
data = pd.read_csv('Data/train.csv', index_col='id')
Y = data['target'].apply(lambda x: int(x[6]))
Y.loc[Y == Y.max()] = 0
X = data.drop('target', axis=1)

# Study
study = optuna.create_study(sampler=optuna.samplers.TPESampler())
study.optimize(objective, show_progress_bar=True)
