# Tractor-Case-Study
How to predict the auction prices of tractors

This project is split into 3 distinct steps, Explatory data analysis, Feature selection, and Grid Search.

# EDA
In this step, I first clean and transform the data so that there are no missing values, and all outliers are properly taken care of. I then create important features based on domain knowledge, etc. knowing cars purchased around the same time tend to cost the same. After getting the data in the way I want, I create a scatter plot to visualize the relationship between continuous variables, and a boxplot to visualize the relationship between categorical variables.

# Feature Selection
In this step, I run 3 forms of feature selection model, Random Forest Regressor, Recursive Feature elimnation with Linear Regression, and Gradient Boosted Regressor. The variables selected end up being very similar although Tree based feature selection finds features that provide the most information gain, and RFE ranks each feature and recursively eliminates the least important feature.

# GridSearch
In this step, we run gridsearch on some of the most common regression models, Lasso, Randomforest, Gradient Boost, and Ada-boost. Gradient Boost ends up being our best algorithm. The error metric measured is mean squared log error.
