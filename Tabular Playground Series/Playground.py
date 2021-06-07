import os
import pandas as pd
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss


data = pd.read_csv('Data/June/train.csv')
x, y = data.drop('target', axis=1).to_numpy(), data['target'].to_numpy()

# TomekLinks (under-sampling)
if not os.path.exists('Data/Tomek.csv'):
    tomek_links = TomekLinks(n_jobs=8)
    x_tl, y_tl = tomek_links.fit_resample(x, y)
    print('Tomek Resampled')
    data = pd.DataFrame(x_tl)
    data['target'] = y_tl
    data.to_csv('Data/Tomek.csv')

# SMOTE (over-sampling)
if not os.path.exists('Data/SMOTE.csv'):
    smote = SMOTE(n_jobs=8)
    x_sm, y_sm = smote.fit_resample(x, y)
    print('SMOTE Resampled')
    data = pd.DataFrame(x_sm)
    data['target'] = y_sm
    data.to_csv('Data/Tomek.csv')

# SMOTETomek (combination)
if not os.path.exists('Data/SMOTE_tomek.csv'):
    smote_tomek = SMOTETomek(n_jobs=8)
    x_st, y_st = smote_tomek.fit_resample(x, y)
    print('SMOTE Tomek Resampled')
    data = pd.DataFrame(x_st)
    data['target'] = y_st
    data.to_csv('Data/SMOTE_tomek.csv')

# Analyse
model_tl = DecisionTreeClassifier(max_depth=5)
model_tl.fit(x_tl, y_tl)
print('Tomek Links: Samples {}, Log Loss: {:.5f}'.format(len(x_tl), log_loss(y, model_tl.predict_proba(x))))

model_sm = DecisionTreeClassifier(max_depth=5)
model_sm.fit(x_sm, y_sm)
print('SMOTE      : Samples {}, Log Loss: {:.5f}'.format(len(x_sm), log_loss(y, model_sm.predict_proba(x))))

model_st = DecisionTreeClassifier(max_depth=5)
model_st.fit(x_st, y_st)
print('SMOTE Tomek: Samples {}, Log Loss: {:.5f}'.format(len(x_st), log_loss(y, model_st.predict_proba(x))))
