# %%
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn. ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# %%#load file
df = pd.read_excel('../Data/MultiDonor_Fluo_IFN.xlsx').drop(['Index'], axis=1)
df = df.dropna()
#Randomly seperate train and test dataset
Train, Test = train_test_split(df, train_size=900)
#keep 14 morph features as X for and reported/measured IDO activity as Y
Xtrain_all = Train.drop(['Donor1', 'level_1', 'Donor2', 'RB_IDO', 'ImageNumber', 'ObjectNumber',],axis=1)
Ytrain_all = Train.RB_IDO
Xtest_all = Test.drop(['Donor1', 'level_1', 'Donor2', 'RB_IDO', 'ImageNumber', 'ObjectNumber',],axis=1)
Ytest_all = Test.RB_IDO
# %%GridSearch for best RFR model
# Create a RFR model
rfr = RandomForestRegressor(random_state=42)
# Define the paramter grid
param_grid = {
    'n_estimators': range(1,200,2),
    'max_depth': range(1, 11, 2),
}
# Perform grid search
grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(Xtrain_all, Ytrain_all)
# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# %%GBR
#Plot permance metrics
results = grid_search.cv_results_
params = results['params']
mean_test_score=results['mean_test_score']
std_test_score=results['std_test_score']
# %% GridSearch for best GBR model
# Define the parameter grid
param_grid = {
    'n_estimators': range(1, 100, 2),
    'max_depth': range(1, 11, 2),
    'learning_rate': [0.01, 0.1, 0.2],
}
# Create a Gradient Boosting Regressor model
gbr = GradientBoostingRegressor(random_state=42)
# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(Xtrain_all, Ytrain_all)
# Get the best model
best_gbr = grid_search.best_estimator_
best_params = grid_search.best_params_
# %% GridSearch for best MLPR model
# Create a MLPR model
mlpr = MLPRegressor()
# Define the paramter grid
param_grid = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['relu', 'tanh','identity', 'logistic'],
    'solver': ['sgd', 'adam'],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [200, 400, 600],
    'alpha': range(1, 600, 50)
}
# Perform frid search
grid_search = GridSearchCV(estimator=mlpr, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(Xtrain_all, Ytrain_all)
# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
# %%GridSearch for best SVR model
# Define the parameter grid
param_grid = {'C': [0.1,1,5,10,20],
              'epsilon': [0.1, 0.3, 0.5, 0.7, 1.0],
              'kernel': ['rbf', 'poly']}
# Create a Gradient Boosting Regressor model
svr=SVR()
# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(Xtrain_all, Ytrain_all)
best_svr = grid_search.best_estimator_
best_params = grid_search.best_params_
# %%GridSearch for best LASSO model
# Define the parameter grid
param_grid = {'alpha': np.linspace(0.01,10, 100)}
# Create a LASSO model
lasso = Lasso()
# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(Xtrain_all, Ytrain_all)
best_lasso = grid_search.best_estimator_
best_params = grid_search.best_params_
# %%GridSearch for best SVR_L model
# Define the parameter grid
param_grid = {'C': [0.1,1,5,10,20],
              'epsilon': [0.1, 0.3, 0.5, 0.7, 1.0],
              'kernel': ['linear']}
# Create a Gradient Boosting Regressor model
svr=SVR()
# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, scoring='r2', cv=5)
grid_search.fit(Xtrain_all, Ytrain_all)
best_svr = grid_search.best_estimator_
best_params = grid_search.best_params_
# %%