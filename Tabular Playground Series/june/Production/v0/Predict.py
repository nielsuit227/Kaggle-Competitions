import pandas as pd
import numpy as np
import struct, re, copy, os


class Predict(object):

    def __init__(self):
        self.version = 'v0.0'

    def predict(self, model, features, data, **args):
        '''
        Prediction function for Amplo AutoML.
        This is in a predefined format:
        - a 'Predict' class, with a 'predict' function taking the arguments:
            model: trained sklearn-like class with the .fit() function
            features: list of strings containing all features fed to the model
            data: the data to predict on
        Note: May depend on additional named arguments within args.
        '''
        ###############
        # Custom Code #
        ###############
            #################
            # Data Cleaning #
            #################
            # Copy vars
            catCols, dateCols, target = [], [], 'target'
            outlierRemoval, missingValues, zScoreThreshold = 'clip', 'interpolate', '4'
    
    # Clean Keys
    new_keys = {}
    for key in data.keys():
        if isinstance(key, int):
            new_keys[key] = 'feature_{}'.format(key)
        else:
            new_keys[key] = re.sub('[^a-zA-Z0-9 \n]', '_', str(key).lower()).replace('__', '_')
    data = data.rename(columns=new_keys)

    def remove_duplicates(data):
        # Remove Duplicates
        data = data.drop_duplicates()
        data = data.loc[:, ~data.columns.duplicated()]

        # Convert Data Types
        for key in dateCols:
            data.loc[:, key] = pd.to_datetime(data[key], errors='coerce', infer_datetime_format=True, utc=True)
        for key in [key for key in data.keys() if key not in dateCols and key not in catCols]:
            data.loc[:, key] = pd.to_numeric(data[key], errors='coerce', downcast='float')
        for key in catCols:
            if key in data.keys():
                dummies = pd.get_dummies(data[key])
                for dummy_key in dummies.keys():
                    dummies = dummies.rename(
                        columns={dummy_key: key + '_' + re.sub('[^a-z0-9]', '_', str(dummy_key).lower())})
                data = data.drop(key, axis=1).join(dummies)

        # Remove Outliers
        if outlierRemoval == 'boxplot':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            for key in q1.keys():
                data.loc[data[key] < q1[key] - 1.5 * (q3[key] - q1[key]), key] = np.nan
                data.loc[data[key] > q3[key] + 1.5 * (q3[key] - q1[key]), key] = np.nan
        elif outlierRemoval == 'z-score':
            z_score = (data - data.mean(skipna=True, numeric_only=True)) \
                     / data.std(skipna=True, numeric_only=True)
            data[z_score > zScoreThreshold] = np.nan
        elif outlierRemoval == 'clip':
            data = data.clip(lower=-1e12, upper=1e12)

        # Remove Missing Values
        data = data.replace([np.inf, -np.inf], np.nan)
        if missingValues == 'remove_rows':
            data = data[data.isna().sum(axis=1) == 0]
        elif missingValues == 'remove_cols':
            data = data.loc[:, data.isna().sum(axis=0) == 0]
        elif missingValues == 'zero':
            data = data.fillna(0)
        elif missingValues == 'interpolate':
            ik = np.setdiff1d(data.keys().to_list(), dateCols)
            data[ik] = data[ik].interpolate(limit_direction='both')
            if data.isna().sum().sum() != 0:
                data = data.fillna(0)
        elif missingValues == 'mean':
            data = data.fillna(data.mean())

        ############
        # Features #
        ############
    def transform(data, features, **args):
        # Split Features
        cross_features = [k for k in features if '__x__' in k or '__d__' in k]
        linear_features = [k for k in features if '__sub__' in k or '__add__' in k]
        trigonometric_features = [k for k in features if 'sin__' in k or 'cos__' in k]
        k_means_features = [k for k in features if 'dist__' in k]
        diff_features = [k for k in features if '__diff__' in k]
        lag_features = [k for k in features if '__lag__' in k]
        original_features = [k for k in features if '__' not in k]

        # Fill missing features for normalization
        required = copy.copy(original_features)
        required += list(itertools.chain.from_iterable([s.split('__')[::2] for s in cross_features]))
        required += list(itertools.chain.from_iterable([s.split('__')[::2] for s in linear_features]))
        required += list(itertools.chain.from_iterable([k.split('__')[1] for k in trigonometric_features]))
        required += list(itertools.chain.from_iterable([s.split('__diff__')[0] for s in diff_features]))
        required += list(itertools.chain.from_iterable([s.split('__lag__')[0] for s in lag_features]))

        # Make sure centers are provided if kMeansFeatures are nonzero
        k_means = None
        if len(k_means_features) != 0:
            if 'k_means' not in args:
                raise ValueError('For K-Means features, the Centers need to be provided.')
            k_means = args['k_means']
            required += [k for k in k_means.keys()]

        # Remove duplicates from required
        required = list(set(required))

        # Impute missing keys
        missing_keys = [k for k in required if k not in data.keys()]
        if len(missing_keys) > 0:
            warnings.warn('Imputing {} keys'.format(len(missing_keys)))
        for k in missing_keys:
            data.loc[:, k] = np.zeros(len(data))

        # Start Output with selected original features
        x = data[original_features]

        # Multiplicative features
        for key in cross_features:
            if '__x__' in key:
                key_a, key_b = key.split('__x__')
                feature = data[key_a] * data[key_b]
                x.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                key_a, key_b = key.split('__d__')
                feature = data[key_a] / data[key_b]
                x.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # Linear features
        for key in linear_features:
            if '__sub__' in key:
                key_a, key_b = key.split('__sub__')
                feature = data[key_a] - data[key_b]
                x.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)
            else:
                key_a, key_b = key.split('__add__')
                feature = data[key_a] + data[key_b]
                x.loc[:, key] = feature.clip(lower=1e-12, upper=1e12).fillna(0)

        # Differentiated features
        for k in diff_features:
            key, diff = k.split('__diff__')
            feature = data[key]
            for i in range(1, int(diff)):
                feature = feature.diff().fillna(0)
            x.loc[:, k] = feature

        # K-Means features
        if len(k_means_features) != 0:
            # Organise data
            centers = k_means.iloc[:-2]
            means = k_means.iloc[-2]
            stds = k_means.iloc[-1]
            temp = copy.deepcopy(data.loc[:, centers.keys()])
            # Normalize
            temp -= means
            temp /= stds
            # Calculate centers
            for key in k_means_features:
                ind = int(key[key.find('dist__') + 6: key.rfind('_')])
                x.loc[:, key] = np.sqrt(np.square(temp - centers.iloc[ind]).sum(axis=1))

        # Lagged features
        for k in lag_features:
            key, lag = k.split('__lag__')
            x.loc[:, k] = data[key].shift(-int(lag), fill_value=0)

        # Trigonometric features
        for k in trigonometric_features:
            func, key = k.split('__')
            x.loc[:, k] = getattr(np, func)(data[key])

        ###########
        # Predict #
        ###########
        mode, normalize = 'classification', False

        # Normalize
        if normalize:
            assert 'scaler' in args.keys(), 'When Normalizing=True, scaler needs to be provided in args'
            X = args['scaler'].transform(X)

        # Predict
        if mode == 'regression':
            if normalize:
                assert 'o_scaler' in args.keys(), 'When Normalizing=True, o_scaler needs to be provided in args'
                predictions = args['OutputScaler'].inverse_transform(model.predict(X))
            else:
                predictions = model.predict(X)
        if mode == 'classification':
            try:
                predictions = model.predict_proba(X)[:, 1]
            except:
                predictions = model.predict(X)

        return predictions
