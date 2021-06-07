from Amplo import Pipeline
import pandas as pd


data = pd.read_csv('Data/June/train.csv')
data['target'] = data['target'].apply(lambda x: int(x[6]))
data.loc[data['target'] == data['target'].max(), 'target'] = 0
pipeline = Pipeline('target', 
                    project='june', 
                    mode='classification',
                    objective='neg_log_loss',
                    plot_eda=False,
                    grid_search_time_budget=4*3600,
                    process_data=True,
                    stacking=True,
                    max_lags=0,
                    max_diff=0,)
pipeline.fit(data)

# Prediction
data = pd.read_csv('Data/June/test.csv')
prediction = pipeline.predict(data)
submission = pd.read_csv('Data/June/sample_submission.csv')
submission['Class_9'] = prediction[:, 0]
for i in range(8):
    submission['Class_{}'.format(i + 1)] = prediction[:, i + 1]
submission.to_csv('Data/June/Submission_v0.csv', index=False)
