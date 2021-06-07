# Amplo AutoML Documentation - HistGradientBoostingClassifier v0

## Model Information

SciKits implementation of LightGBM, or **L**ight **G**radient **B**oosting **M**achine, is an iteration on the XGBoost algorithm. Similarly, it uses gradient boosting with decision trees. However, XGBoost tend to be slow for a larger number of samples (>10.000), but with leaf-wise growth instead of depth-wise growth, LightGBM increases training speed significatnly. Performance is often close to XGBoost, sometimes for the better and sometimes for the worse. 

## Model Performance

Model performance is analysed by various metrics. Below you find various metrics and a confusion matrix.
This model has been selected based on the neg_log_loss score.

### Metrics

| Metrcis | Score |
| --- | ---: |
| Avg. Accuracy | 0.36 ± 0.00 |
| F1 Score      | 0.15 ± 0.20 |
| Log Loss      | 1.76 ± 0.00 |

### Confusion Matrix


<table>
    <thead>
        <tr>
            <td> </td>
            <td> </td>
            <td colspan=9 style="text-align:center">True Label</td>
        </tr>
        <tr><td> </td><td> </td><td> 6.0 </td><td> 2.0 </td><td> 8.0 </td><td> 3.0 </td><td> 1.0 </td><td> 5.0 </td><td> 7.0 </td><td> 0.0 </td><td> 4.0 </td></tr>
    </thead>
    <tbody>
        <tr>
<td rowspan=9 style="vertical-align:middle">Prediction</td>
<td>
6.0
</td>
<td>
0.16 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.79 ± 0.05
</td>
<td>
0.02 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
3.51 ± 0.07
</td>
<td>
0.00 ± 0.00
</td>
<td>
6.28 ± 0.03
</td>
</tr>
<tr>

<td>
2.0
</td>
<td>
0.05 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
1.06 ± 0.03
</td>
<td>
0.01 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
1.19 ± 0.03
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.25 ± 0.03
</td>
</tr>
<tr>

<td>
8.0
</td>
<td>
0.15 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
5.87 ± 0.05
</td>
<td>
0.04 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.51 ± 0.05
</td>
<td>
0.00 ± 0.00
</td>
<td>
3.65 ± 0.06
</td>
</tr>
<tr>

<td>
3.0
</td>
<td>
0.10 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.75 ± 0.04
</td>
<td>
0.02 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
1.80 ± 0.02
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.73 ± 0.02
</td>
</tr>
<tr>

<td>
1.0
</td>
<td>
0.03 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.57 ± 0.02
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.63 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
1.11 ± 0.04
</td>
</tr>
<tr>

<td>
5.0
</td>
<td>
0.02 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.29 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.41 ± 0.03
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.82 ± 0.03
</td>
</tr>
<tr>

<td>
7.0
</td>
<td>
0.13 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.24 ± 0.09
</td>
<td>
0.01 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
13.40 ± 0.10
</td>
<td>
0.00 ± 0.00
</td>
<td>
10.12 ± 0.02
</td>
</tr>
<tr>

<td>
0.0
</td>
<td>
0.06 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.79 ± 0.02
</td>
<td>
0.01 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
1.96 ± 0.02
</td>
<td>
0.00 ± 0.00
</td>
<td>
4.57 ± 0.01
</td>
</tr>
<tr>

<td>
4.0
</td>
<td>
0.17 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
1.97 ± 0.03
</td>
<td>
0.03 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
7.18 ± 0.05
</td>
<td>
0.00 ± 0.00
</td>
<td>
16.54 ± 0.05
</td>
</tr>
    </tbody>
</table>
                


## Validation Strategy

All experiments are cross validated. This means that every time a model's performance is evaluated, it's trained on one part of the data, and tested on another. Therefore, the model is always tested against data it has not yet been trained for. This gives the best approximation for real world (out of sample) performance.
The current validation strategy used StratifiedKFold, with 3 splits and with shuffling the data.

## Parameters

The optimized model has the following parameters:

| categorical_features | early_stopping | l2_regularization | learning_rate | loss | max_bins | max_depth | max_iter | max_leaf_nodes | min_samples_leaf|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| None | auto | 8.6755 | 0.1486 | auto | 200 | 8 | 138 | 31 | 261 |

| monotonic_cst | n_iter_no_change | random_state | scoring | tol | validation_fraction | verbose | warm_start|
| --- | --- | --- | --- | --- | --- | --- | --- |
| None | 10 | None | loss | 0.0 | 0.1 | 0 | False |



## Features

### Feature Extraction

Firstly, features that are co-linear (a * x = y), up to 99.0 %, were removed. This resulted in 0 removed features:
 

Subsequently, the features were manipulated and analysed to extract additional information. 
Most combinations are tried (early stopping avoids unpromising features directly). 
The usefulness of a newly extracted features are analysed by a single decision tree. 

| Sort Feature | Quantity | Features |
| --- | ---: | --- |
| Multiplied / Divided Features | 0 |  |
| Added / Subtracted Features   | 0 |  |
| Trigonometric Features        | 0 |  |
| K-Means Features              | 0 |  |
| Lagged Features               | 0 |  |
| Differentiated Features       | 0 |  |

### Feature Selection
Using a Random Forest model, the non-linear Feature Importance is analysed. The Feature Importance is measured
in Mean Decrease in Gini Impurity. 
The Feature Importance is used to create two feature sets, one that contains 95% of all Feature Importance (RFT) and 
one that contains all features that contribute more than 1% to the total Feature Importance (RFI). 

Top 20 features:

<img src="Feature_Importance_HistGradientBoostingClassifier.png" width="400" height="600">

## Data Processing
        
Data cleaning steps: 
1. Removed 0 duplicate columns and 0 duplicate rows.
2. Handled outliers with clip
3. Imputed 0 missing values with interpolate
4. Removed 0 columns with only constant values

## Model Score Board

Not only HistGradientBoostingClassifier has been optimized by the AutoML pipeline. In total, 11 models were trained. 
The following table shows the performance of the top 10 performing models:

| Model | neg_log_loss     | Parameters |
| --- | ---: | --- |
| HistGradientBoostingClassifier | -1.7584 ± 0.0013 | {'categorical_features': None, 'early_stopping': 'auto', 'l2_regularization': 0.0, 'learning_rate': 0.1, 'loss': 'auto', 'max_bins': 255, 'max_depth': None, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 20, 'monotonic_cst': None, 'n_iter_no_change': 10, 'random_state': None, 'scoring': 'loss', 'tol': 1e-07, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False} |
| LGBMClassifier | -1.7614 ± 0.0003 | {'verbosity': -1, 'force_col_wise': True, 'objective': 'multiclass', 'num_classes': 9} |
| CatBoostClassifier | -1.7501 ± 0.0007 | {'verbose': 0, 'allow_writing_files': False, 'n_estimators': 1000} |
| RandomForestClassifier | -2.0170 ± 0.0033 | {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False} |
| HistGradientBoostingClassifier | -1.8542 ± 0.0003 | {'categorical_features': None, 'early_stopping': 'auto', 'l2_regularization': 0.0, 'learning_rate': 0.1, 'loss': 'auto', 'max_bins': 255, 'max_depth': None, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 20, 'monotonic_cst': None, 'n_iter_no_change': 10, 'random_state': None, 'scoring': 'loss', 'tol': 1e-07, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False} |
| LGBMClassifier | -1.8601 ± 0.0002 | {'verbosity': -1, 'force_col_wise': True, 'objective': 'multiclass', 'num_classes': 9} |
| CatBoostClassifier | -1.8517 ± 0.0003 | {'verbose': 0, 'allow_writing_files': False, 'n_estimators': 1000} |
| RandomForestClassifier | -14.1298 ± 0.0968 | {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False} |
| CatBoostClassifier | -1.7474 ± 0.0010 | {'n_estimators': 1512, 'learning_rate': 0.015561726174461438, 'l2_leaf_reg': 5.421254944299484, 'depth': 5, 'min_data_in_leaf': 914, 'grow_policy': 'Lossguide'} |
| HistGradientBoostingClassifier | -1.7562 ± 0.0021 | {'learning_rate': 0.14858839097513943, 'max_iter': 138, 'max_leaf_nodes': 31, 'max_depth': 8, 'min_samples_leaf': 261, 'l2_regularization': 8.675498344925586, 'max_bins': 200} |


