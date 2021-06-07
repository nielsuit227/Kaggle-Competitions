import pandas as pd
from self.AutoML import Pipeline


if __name__ == '__main__':
    # Data
    data = pd.read_csv('Data/train.csv', index_col='id')
    data['target'] = data['target'].apply(lambda x: int(x[6]))
    data.loc[data['target'] == data['target'].max(), 'target'] = 0

    # Automated Machine Learning Pipeline
    pipeline = Pipeline('target',
                        cat_cols=[k for k in data.keys() if k != 'target'],
                        mode='classification',
                        objective='neg_log_loss',
                        plot_eda=False,
                        # process_data=False,
                        validate_result=True,
                        grid_search_iterations=4,
                        normalize=True,
                        max_lags=0,
                        shuffle=True,
                        include_output=False,
                        cv_splits=3,
                        max_diff=0)
    pipeline.fit(data)