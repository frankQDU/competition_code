import os 
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
import catboost as ctb

from tqdm import tqdm
from scipy.stats import skew,kurtosis
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

def make_score(y_true , y_pred ):
    return mean_squared_error( y_true , y_pred )


def train_model_lgb(X, y, test, params, feature_col, n_splits=10):

    fi = []
    cv_score = []
    test_pred = np.zeros((test.shape[0],))
    train_pred = np.zeros((X.shape[0],))
    skf = KFold(n_splits=n_splits, random_state=2019, shuffle=True)
    
    for index, (train_index, test_index) in enumerate(skf.split(X, y)):

        print(index)
        train_x, test_x = X.iloc[train_index],X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]

        train_set = lgb.Dataset(train_x[feature_col], train_y)
        test_set = lgb.Dataset(test_x[feature_col],test_y)
        lgb_model = lgb.train(params,
                          train_set,
                          valid_sets=[train_set,test_set],
                          early_stopping_rounds=500, 
                          num_boost_round=10000 ,
                          verbose_eval=500)

        y_val = lgb_model.predict(test_x[feature_col])

        train_pred[test_index] = y_val

        print( make_score( test_y , y_val )) 

        cv_score.append( make_score( test_y , y_val) ) 
    
        print(cv_score[index])
    
        test_pred += lgb_model.predict(test[feature_col]) / n_splits
        fi.append(lgb_model.feature_importance(importance_type='gain'))
    return train_pred, test_pred, fi





def train_model_xgb(X, y, test, params, feature_col, n_splits=10):
    fi = []
    cv_score = []
    test_pred = np.zeros((test.shape[0],))
    train_pred = np.zeros((X.shape[0],))
    skf = KFold(n_splits=n_splits, random_state=2019, shuffle=True)
    
    for index, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(index)
        train_x, test_x = X.iloc[train_index],X.iloc[test_index]
        train_y, test_y = y.iloc[train_index], y.iloc[test_index]
        
        train_set = xgb.DMatrix(train_x[feature_col] , train_y)
        test_set = xgb.DMatrix(test_x[feature_col] , test_y)
        
        xgb_model = xgb.train(params,
                          train_set,
                          evals=[(train_set,'train'),(test_set,'test')],
                          early_stopping_rounds=100, 
                          num_boost_round=10000,
                          verbose_eval=1000)

        y_val = xgb_model.predict(test_set)
        train_pred[test_index] = y_val

        print( make_score( test_y , y_val )) 

        cv_score.append( make_score( test_y , y_val) ) 
    
        print(cv_score[index])
    
        test_pred += xgb_model.predict(xgb.DMatrix(test[feature_col])) / n_splits
    
    return train_pred, test_pred, fi

xgb_paras = {'objective': 'reg:squarederror',
 'tree_method': 'gpu_hist',
 'eval_metric': 'rmse',
 'learning_rate': 0.02,
 'alpha': 0.30328974897294075,
 'colsample_bytree': 0.5068082755866445,
 'lambda': 72.2173472522586,
 'max_depth': 9,
 'min_child_weight': 5,
 'subsample': 0.8170133539039669}
lgb_paras = {'objective': 'regression',
 'metric': 'rmse',
 'learning_rate': 0.02,
 'num_threads': -1,
 'early_stopping_rounds': 100,
 'num_boost_round': 10000,
 'bagging_fraction': 0.9978192061670864,
 'bagging_freq': 1,
 'feature_fraction': 0.5234178718477926,
 'max_depth': 7,
 'min_child_weight': 1,
 'num_leaves': 41,
 'reg_alpha': 0.1415592188002883,
 'reg_lambda': 2.2724007900790895}
