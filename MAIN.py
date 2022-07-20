import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datetime import date, datetime
import pandas as pd
import numpy as np

from cl_random_forest import *

df, X_, Y_, X_train_, X_test_, y_train_, y_test_, X_train_part_, X_valid_, y_train_part_, y_valid_ = DO_PREPROCESSING()
Y_ = Y_[col_y_true].tolist()
y_train_ = y_train_[col_y_true].tolist()
y_train_part_ = y_train_part_[col_y_true].tolist()
y_valid_ = y_valid_[col_y_true].tolist()
y_test_origin = y_test_.copy()
y_test_ = y_test_[col_y_true].tolist()

M = c_Random_Forest()
M.get_best_model(14, idx=0, random_state=71)
M.fit_predict_model(M.best_model, X_train_, y_train_, X_test_, y_test_) 
