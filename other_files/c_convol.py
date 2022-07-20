import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import warnings
warnings.filterwarnings("ignore")

from datetime import date, datetime
import pandas as pd
import numpy as np

from input_nn import *
from lib_preprocessing import DO_PREPROCESSING
from input_values import *
from lib_general_functions import *

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, KFold, StratifiedKFold, cross_val_score, cross_validate, learning_curve
from sklearn import metrics

import tensorflow
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, InputLayer, Embedding, Conv2D, MaxPooling2D, Flatten
from keras import Input

df, X_, Y_, X_train_, X_test_, y_train_, y_test_, X_train_part_, X_valid_, y_train_part_, y_valid_ = DO_PREPROCESSING()
X_SHAPE = X_.shape
# Y = Y[col_y_true].tolist()
# y_train = y_train[col_y_true].tolist()
# y_test = y_test[col_y_true].tolist()



class c_CONV:
  def __init__(self, M):
    self.M = M

  def adjust_predictions(self, y_pred_tr, y_pred_te, treshold):
    treshold_tr = np.mean(y_pred_tr)*0.66 if treshold is None else treshold
    treshold_te = np.mean(y_pred_te)*0.66 if treshold is None else treshold
    y_pred_tr = get_binary_prediction_with_treshold(y_pred_tr, treshold_tr)
    y_pred_te = get_binary_prediction_with_treshold(y_pred_te, treshold_te)
    print('treshold_tr', treshold_tr, '      treshold_te', treshold_te)
    return y_pred_tr, y_pred_te


  def fit_predict_model(self, M, X_train, y_train, X_test, y_test, treshold=None):
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=1, restore_best_weights=True)    #callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.01)
    X_train, y_train, X_test, y_test = self.trasform_shape(X_train, y_train, X_test, y_test)
    M.fit(X_train, y_train, callbacks=[callback], verbose=0)
    y_pred_tr = M.predict(X_train)
    y_pred_te = M.predict(X_test)
    y_pred_tr, y_pred_te = self.adjust_predictions(y_pred_tr, y_pred_te, treshold)
    M_Train_accuracy = round(metrics.accuracy_score(y_train, y_pred_tr),2)
    M_Test_accuracy = round(metrics.accuracy_score(y_test, y_pred_te),2)
    treshold = '' if treshold is None else str(treshold)
    print('  ' + treshold + '   M_Train_accuracy', round(M_Train_accuracy,2), '  |  M_Test_accuracy', round(M_Test_accuracy,2))




M = c_CONV(model)
M.fit_predict_model(model, X_train_, X_test_, y_train_, y_test_)
