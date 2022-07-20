import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datetime import date, datetime
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, cross_val_score, cross_validate, learning_curve
from sklearn import metrics

from lib_preprocessing import DO_PREPROCESSING
from input_values import *
from lib_general_functions import *



class c_Dummy():
  def __init__(self, strategy, X_train, y_train, X_test, y_test, constant=None): #'most_frequent', uniform, stratified
    M = DummyClassifier(strategy=strategy, constant=constant)
    if CV_MODE == 'TS':
      CV = TimeSeriesSplit().split(X_train)
    elif CV_MODE == 'SKF':
      CV = StratifiedKFold().split(X_train,y_train)

    CV_score = cross_val_score(M, X_train, y_train, cv=CV, scoring='accuracy')

    print('   ', strategy + ' TRAIN Accuracy:', round(CV_score.mean(),2) , ' (mean) |  cross_val_score ==> ', CV_score)
    M.fit(X_train, y_train)

    y_pred = M.predict(X_test)
    accuracy_test = round(metrics.accuracy_score(y_test, y_pred),2)
    print('   ', strategy + '  TEST  Accuracy', accuracy_test)

df, X_, Y_, X_train_, X_test_, y_train_, y_test_, X_train_part_, X_valid_, y_train_part_, y_valid_ = DO_PREPROCESSING()
X_SHAPE = X_.shape
Y_ = Y_[col_y_true].tolist()
y_train_ = y_train_[col_y_true].tolist()
y_test_ = y_test_[col_y_true].tolist()

D_mf = c_Dummy('constant', X_train_, y_train_, X_test_, y_test_, constant=1)
print()
D_mf = c_Dummy('most_frequent', X_train_, y_train_, X_test_, y_test_)
print()
D_mf = c_Dummy('prior', X_train_, y_train_, X_test_, y_test_)
print()
D_mf = c_Dummy('stratified', X_train_, y_train_, X_test_, y_test_)
print()
D_mf = c_Dummy('uniform', X_train_, y_train_, X_test_, y_test_)
print()

