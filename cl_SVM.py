import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datetime import date, datetime
import pandas as pd
import numpy as np

from lib_preprocessing import DO_PREPROCESSING
from input_values import *
from lib_general_functions import *

from sklearn.svm import SVC, LinearSVC
from sklearn import metrics


df, X_, Y_, X_train_, X_test_, y_train_, y_test_, X_train_part_, X_valid_, y_train_part_, y_valid_ = DO_PREPROCESSING()
Y_ = Y_[col_y_true].tolist()
y_train_ = y_train_[col_y_true].tolist()
y_test_origin = y_test_.copy()
y_test_ = y_test_[col_y_true].tolist()



class c_SVC:
  def __init__(self, X, Y, X_train, y_train, X_test, y_test):
    self.M = SVC(C=1, max_iter=50000)


  def fit_predict_model(self, M, X_train, y_train, X_test, y_test, label='', print_future_prediction=None):
    # Accuracy of Train & Test
    M.fit(X_train, y_train)
    y_pred_train, y_pred_test = M.predict(X_train), M.predict(X_test)
    print_accuracy_train_test(y_train, y_test, y_pred_train, y_pred_test, label)
    if print_future_prediction:
      df_pred = pd.DataFrame({'time': y_test_origin.index, 'pred': y_pred_test})
      print('df_pred\n', df_pred[-12:])


  def get_best_model(self, N, idx=None, random_state=None):
    random_state=1 if random_state is None else random_state
    file_name = 'results/'+str(ticker_to_predict) + '_' + str(PERIODS_TO_FORECAST) + '_SVC_' + str(N) + '.xlsx'
    best_params = read_from_excel_best_params_Grid_Search(file_name, idx)
    self.best_model = SVC(random_state=random_state, **best_params)
    print('    best_model =>', self.best_model)


  def compute_grid_search(self, X, Y, n_iter):
    grid1 = {'C': [0.1, 1, 10, 100, 1000],
            'gamma': ['auto', 1, 0.1, 0.01, 0.001, 0.0001],
            'degree': [2,3],
            'kernel': ['rbf', 'poly', 'sigmoid'], # Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.    If none is given, 'rbf' will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape
            'coef0': [1, 10]
          }
    grid_poly = {'C': [0.1, 1, 10, 100],
            'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001],
            'degree': [3,4],
            'kernel': ['poly'],
            'coef0': [1, 5, 10]
          }
    grid_rbf = {'C': [0.1, 1, 10, 100],
            'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf'],
          }
    grid_sigmoid = {'C': [0.1, 1, 10, 100],
            'gamma': ['auto', 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['sigmoid'],
            'coef0': [1, 5, 10]
          }

          # POLY ---> SVC(kernel="poly", degree=Degree, coef0=Coef0)
            # degree ==> int, default=3  |  Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
            # coef0 : float, default=0.0   |  Independent term in kernel function.   It is only significant in 'poly' and 'sigmoid'.

          # RBF  ---> SVC(kernel="rbf", gamma=gamma, C=C)

          # BOTH
            # C : float, default=1.0  |   Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
            # gamma  ==> Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    self.df_grid_search, self.best_model = do_Grid_Search(SVC(), grid_sigmoid, X, Y, 'SVC', n_iter)

M = c_SVC(X_, Y_, X_train_, y_train_, X_test_, y_test_)
M.get_best_model(4, idx=0)
M.fit_predict_model(M.best_model, X_train_, y_train_, X_test_, y_test_)
A=9876/0
for i in range(100):
  M.get_best_model(4, idx=i, random_state=71)# QUI Random State NON incide
  M.fit_predict_model(M.best_model, X_train_, y_train_, X_test_, y_test_)
M.compute_grid_search(X_train_, y_train_, 150)
a=987/0
'''
  I buoni sono:
  - SVC (1) [0] <==== POLY
  - SVC (3) [0] <==== RBF
  - SVC (4) [0] <==== SIGMOID
  - 
'''

    