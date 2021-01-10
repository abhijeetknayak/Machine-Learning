### Kaggle Course : Intro to ML
* **Scikit-learn library for Machine learning applications**
  * __train_test_split__ (sklearn.model_selection) : Splits train data into training and validation splits
  * Validation : Needed to prove that the model actually works. The model can be overfit or underfit 
    * Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions
    * Underfitting: failing to capture relevant patterns, again leading to less accurate predictions
  * __mean_absolute_error__ (sklearn.metrics) : L1 error
  * __DecisionTreeRegressor__ (sklearn.tree) : Decision tree for data regression
    * The model fit will be determined by the depth of the tree that you choose. If a large depth is used, the model is overfit, whereas if a small depth is used, the model is underfit to the data. Modify **max_leaf_nodes** to change model fitting
  * __RandomForestRegressor__ (sklearn.ensemble) : Random Forest models. Multiple decision trees to prevent under-fitting and over-fitting
  * __Linear Regressor__ (sklearn.linear_model LinearRegression)
  
