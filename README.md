# Machine Learning to Forecast Financial Market Direction
This work was created by me for the thesis of the master in data science for business and finance, La Statale University of Milan, academic year 2021/2022
# How is structured the work
The aim is to predict the monthly variation of a financial security using various machine learning models.
There is a global configuration file 'input_values.py'. In this file is possible to define:
- (ticker_to_predict) which ticker you want to predict (default = S&P 500). The ticker must be present in 'v_d_tickers'
- (PERIODS_TO_FORECAST) the number of periods you want to predict the direction. For example, if you choose 1, you say you want to predict the next month direction of the ticker; instead, if you choose 12 your aim is to predict wheather the S&P 500 will be above or below the price of today in 12 months.
- which cross-validation method you want to use (Time Series or Stratified K-Fold)
- which indicators you want to include as external regressors
- (V_D_REGRESSORS) which transformations has to be done for each external regressor. The indicators must be declared in 'v_d_macro_indicators', 'v_d_special_indicators', 'v_d_tech_indicators', 'v_lags', 'd_price_ma', 'd_roc'

And there is a common pre-processing phase, done in the file 'lib_preprocessing.py':
- it takes input data from '/data' folder
- it returns a set of dataframes ready to be used, and splitted in train, validation, and test parts.

The models are:
- Random Forest Classifier => file cl_RF.py  ('cl' means 'class')
- KNN => cl_KNN.py
- Support Vector Classifier => cl_SVM.py
- Neural Network Classifier => cl_NN.py, and configurations in input_nn.py

# The logical procedure
To get results follow this scheme
- Choose a Model, for example Random Forest, which seems to be the best for this problem in terms of effectiveness/simplicicy ratio.
- Go to MAIN.py file, include it with 'from cl_RF import *'
- Define a new model 'M = c_Random_Forest()' => on init will be created a 'parameters grid' for 'Grid Search'
- Launch a Randomized Grid Search 'M.compute_grid_search(X_train_, y_train_, 100)', in this example I choose 100 casual iterations. It will save a file in '/results' folder as 'Tid_PF_**RF**_N', where: Tid = ticker_to_predict, PF = PERIODS_TO_FORECAST, RF: Random Forest model, N: the N° of grid search done with Tid+PF+RF
- Load the best model   M.get_best_model(N, idx=0, random_state=71) ==> that will be stored in  self.best_model    <=== note that N is the number of grid search done (explained in previous point)
- Launch the prediction on the test set ==> M.fit_predict_model(M.best_model, X_train_, y_train_, X_test_, y_test_) and get the results in term of accuracy of the best model you have found with the Grid Search.
