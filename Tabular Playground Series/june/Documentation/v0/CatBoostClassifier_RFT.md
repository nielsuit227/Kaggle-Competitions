# Amplo AutoML Documentation - CatBoostClassifier v0

## Model Information

CatBoost, or **Cat**egorical **Boost**ing,  is an algorithm for gradient boosting on decision trees, with natural implementation for categorical variables. It is similar to XGBoost and LightGBM but differs in implementation of the optimization algorithm. We often see this algorithm performing very well.

## Model Performance

Model performance is analysed by various metrics. Below you find various metrics and a confusion matrix.
This model has been selected based on the neg_log_loss score.

### Metrics

| Metrcis | Score |
| --- | ---: |
| Avg. Accuracy | 0.36 ± 0.00 |
| F1 Score      | 0.14 ± 0.20 |
| Log Loss      | 1.75 ± 0.00 |

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
0.03 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.96 ± 0.01
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
3.40 ± 0.02
</td>
<td>
0.00 ± 0.00
</td>
<td>
6.38 ± 0.02
</td>
</tr>
<tr>

<td>
2.0
</td>
<td>
0.01 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
1.10 ± 0.02
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
1.15 ± 0.03
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.30 ± 0.03
</td>
</tr>
<tr>

<td>
8.0
</td>
<td>
0.03 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
6.10 ± 0.06
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
2.43 ± 0.03
</td>
<td>
0.00 ± 0.00
</td>
<td>
3.66 ± 0.03
</td>
</tr>
<tr>

<td>
3.0
</td>
<td>
0.02 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.90 ± 0.02
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
1.74 ± 0.05
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.74 ± 0.03
</td>
</tr>
<tr>

<td>
1.0
</td>
<td>
0.01 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.61 ± 0.01
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
0.61 ± 0.02
</td>
<td>
0.00 ± 0.00
</td>
<td>
1.13 ± 0.02
</td>
</tr>
<tr>

<td>
5.0
</td>
<td>
0.01 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.31 ± 0.02
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
0.39 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.83 ± 0.02
</td>
</tr>
<tr>

<td>
7.0
</td>
<td>
0.02 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.33 ± 0.02
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
13.27 ± 0.09
</td>
<td>
0.00 ± 0.00
</td>
<td>
10.29 ± 0.08
</td>
</tr>
<tr>

<td>
0.0
</td>
<td>
0.01 ± 0.00
</td>
<td>
0.00 ± 0.00
</td>
<td>
0.86 ± 0.01
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
1.90 ± 0.03
</td>
<td>
0.00 ± 0.00
</td>
<td>
4.62 ± 0.02
</td>
</tr>
<tr>

<td>
4.0
</td>
<td>
0.02 ± 0.01
</td>
<td>
0.00 ± 0.00
</td>
<td>
2.15 ± 0.05
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
6.98 ± 0.02
</td>
<td>
0.00 ± 0.00
</td>
<td>
16.74 ± 0.05
</td>
</tr>
    </tbody>
</table>
                


## Validation Strategy

All experiments are cross validated. This means that every time a model's performance is evaluated, it's trained on one part of the data, and tested on another. Therefore, the model is always tested against data it has not yet been trained for. This gives the best approximation for real world (out of sample) performance.
The current validation strategy used StratifiedKFold, with 3 splits and with shuffling the data.

## Parameters

The optimized model has the following parameters:

| verbose | allow_writing_files | n_estimators | learning_rate | l2_leaf_reg | depth | min_data_in_leaf | grow_policy |

| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | False | 1512 | 0.0156 | 5.4213 | 5 | 914 | Lossguide |



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

<img src="Feature_Importance_CatBoostClassifier.png" width="400" height="600">

## Data Processing
        
Data cleaning steps: 
1. Removed 0 duplicate columns and 0 duplicate rows.
2. Handled outliers with clip
3. Imputed 0 missing values with interpolate
4. Removed 0 columns with only constant values

## Model Score Board

Not only CatBoostClassifier has been optimized by the AutoML pipeline. In total, 9 models were trained. 
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


