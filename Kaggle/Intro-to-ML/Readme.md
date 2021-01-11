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
  * __Imputation__ (sklearn.impute SimpleImputer): To handle missing values
    * Use a strategy to fill in the missing values. This may be 'mean', 'median', 'most_frequent', etc.
    * Imputation is only for non-categorical variables?
  * __Handling Categorical variables__:
    * Label Encoding (sklearn.preprocessing LabelEncoder): Categories are given values
    * One Hot Encoding (sklearn.preprocessing OneHotEncoder): 
  * __Creating Pipelines__ : Helps in organizing your code(Super Useful!!)
    * ColumnTransformer (sklearn.compose): Define Transformers in order 
    * Pipeline (sklearn.pipeline): Define steps in order
#### XGBoost: Extreme Gradient Boosting
* Ensemble method to get the best possible model
  * Start with a naive model and train it to get a loss function
  * Applies gradient descent on the loss function to create a new model, and adds this model to the ensemble. Repeat this process for a number of iterations
  
