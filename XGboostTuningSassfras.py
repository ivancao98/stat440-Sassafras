import pandas as pd
import numpy as np

from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from pyearth import Earth
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold


Xtr = pd.read_csv(r"C:\Users\Ivan Cao\OneDrive\Documents\stat440\project3\project3B\Xtrain.txt", delimiter = ' ', index_col = 'Id')

Xte = pd.read_csv(r"C:\Users\Ivan Cao\OneDrive\Documents\stat440\project3\project3B\Xtest.txt", delimiter = ' ',  index_col = 'Id')

Ytr = pd.read_csv(r"C:\Users\Ivan Cao\OneDrive\Documents\stat440\project3\project3B\Ytrain.txt",  index_col = 'Id')


Xteindex = pd.Series(Xte.index)

Xtr = Xtr.replace('?',0)
Xte = Xte.replace('?',0)
Xtr = Xtr.fillna(0)
Xte = Xte.fillna(0)

Xte = Xte.iloc[:,0:66]
Xtr = Xtr.iloc[:,0:66]
Xtr = Xtr.replace(0, Xtr.median())
Xte = Xte.replace(0, Xte.median())

Xtr['#B17'] = pd.to_numeric(Xtr['#B17'], downcast = "float")
Xte['#B17'] = pd.to_numeric(Xte['#B17'], downcast = "float")



X_train, X_test, y_train, y_test=train_test_split(Xtr, Ytr, test_size=0.3)

def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']

    }
    param_tuning['eval_metric'] = 'mae'

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,                        
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_

xgb_model = XGBRegressor(
        objective = 'reg:squarederror',
        colsample_bytree = 0.5,
        learning_rate = 0.05,
        max_depth = 6,
        min_child_weight = 1,
        n_estimators = 1000,
        subsample = 0.7)

xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)], verbose=False)

y_pred_xgb = xgb_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print("MAE: ", mae_xgb)



yhat = xgb_model.predict(Xte)


t1 = pd.DataFrame(yhat)

tcolumn = pd.DataFrame(Xteindex)



tcolumn['prediction'] = t1

tcolumn.columns = ['Id', 'prediction']

tcolumn.to_csv("C:/Users/Ivan Cao/OneDrive/Documents/stat440/project3/XGBRMAETUNING2.csv", index = False)
