'''
INIZIO
- Cerco il miglior modello (TRAIN) con GridSearch, in cui faccio il CV (5)
- Uso quel modello per la prediction sul TEST => salvo accuracy (train e test)
FINE
'''
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

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

df, X_, Y_, X_train_, X_test_, y_train_, y_test_, X_train_part_, X_valid_, y_train_part_, y_valid_ = DO_PREPROCESSING()
X_SHAPE = X_.shape
Y_ = Y_[col_y_true].tolist()
y_train_ = y_train_[col_y_true].tolist()
y_test_origin = y_test_
y_test_ = y_test_[col_y_true].tolist()


class c_KNN:
  def __init__(self):
    M = KNeighborsClassifier()#n_neighbors=21)
    self.M = M

  def fit_predict_model(self, M, X_train, X_test, y_train, y_test, label=None, print_future_prediction=None):
    M.fit(X_train, y_train)
    y_pred_tr = M.predict(X_train)
    y_pred_te = M.predict(X_test)
    print_accuracy_train_test(y_train, y_test, y_pred_tr, y_pred_te, label)
    if print_future_prediction:
      df_pred = pd.DataFrame({'time': y_test_origin.index, 'pred': y_pred_te})
      print('df_pred\n', df_pred[-12:])


  def compute_grid_search(self, X, Y, n_iter):
    # file_name ==> without format (csv / xlsx)
    M = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    maxN = min(X.shape[0]//3, 50)
    v_neighbors = np.arange(1, maxN)
    param_grid = {'n_neighbors': v_neighbors}
    print('   maxN', maxN, '    param_grid', param_grid, len(v_neighbors))
    self.df_grid_search, self.best_model = do_Grid_Search(M, param_grid, X, Y, 'KNN', n_iter)


  def get_best_model(self, N, idx=None):
    file_name = 'results/'+str(ticker_to_predict) + '_' + str(PERIODS_TO_FORECAST) + '_KNN_' + str(N) + '.xlsx'
    best_params = read_from_excel_best_params_Grid_Search(file_name, idx)
    self.best_model = KNeighborsClassifier(**best_params) # Non ha random state
    print('    best_model =>', self.best_model)


M = c_KNN()
#M.compute_grid_search(X_, Y_, n_iter=50)
M.get_best_model(3, idx=0)
M.fit_predict_model(M.best_model, X_train_, X_test_, y_train_, y_test_)
s=987/0
for i in range(50):
  M.get_best_model(3, idx=i)
  M.fit_predict_model(M.best_model, X_train_, X_test_, y_train_, y_test_, i)
s=987/0

