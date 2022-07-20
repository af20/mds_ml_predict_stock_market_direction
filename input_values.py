TRAIN_SIZE = 0.8
TRAIN_SIZE_on_VALIDATION = 0.75
CV_MODE = 'SKF' # [TS, SKF]
DO_OVER_SAMPLING = False
N_FOLDS = 5


# ......... INDICATORS ............
# . Transofrmations .
tr_0 = ['ma_7', 'min_max__0_1']
tr_1 = ['roc_12', 'ma_7', 'min_max__0_1']
tr_2 = ['min_max__0_1']

prefix_file_name_indicators = 'data/indicator_'
prefix_file_name_tickers = 'data/ticker_'
ticker_to_predict = 3539
col_y_true = 'y_true'
PERIODS_TO_FORECAST = 12

v_d_tickers = [
  {'id': 3539, 'name': 'S&P 500', 'name_change': 'S&P 500 change'},
  {'id': 3551, 'name': 'Gold', 'name_change': 'Gold change'},
  {'id': 6063, 'name': 'Crude Oil', 'name_change': 'Crude Oil change'}
]

v_d_macro_indicators = [
  {'id': 1, 'name': 'ISM', 'transformations': tr_0},
  {'id': 8, 'name': 'CAPE', 'transformations': tr_0},
  {'id': 43, 'name': 'Sentiment', 'transformations': tr_0},
  {'id': 123, 'name': 'FedFunds', 'transformations': tr_0},
]

v_d_special_indicators = [
  {'id': 1, 'name': 'Mom+Sent', 'transformations': tr_2}
]

v_d_tech_indicators = [
  {'id': 1, 'indicator': 'RSI', 'periods': 14, 'name': 'RSI 14', 'transformations': tr_0},
  {'id': 2, 'indicator': 'RSI', 'periods': 14, 'name': 'RSI 14 ROC', 'transformations': tr_1},
  {'id': 3, 'indicator': 'ATR', 'periods': 14, 'name': 'ATR 14 ROC', 'transformations': tr_1},
]

v_lags = [1]#,12,24,36]
d_price_ma = {'v_ma': [3,24,48], 'transformations': tr_0}

d_roc = {'v_roc': [12,36], 'transformations': tr_0}

V_D_REGRESSORS = [
  {'type': 'macro_indicator', 'id': 1},
  {'type': 'macro_indicator', 'id': 8},
  {'type': 'macro_indicator', 'id': 43},
  {'type': 'macro_indicator', 'id': 123},

  {'type': 'tech_indicator', 'id': 1},
  #{'type': 'tech_indicator', 'id': 2},
  {'type': 'tech_indicator', 'id': 3},

  #{'type': 'special_indicator', id: 1}

  {'type': 'lags'},
  #{'type': 'price_ma'},
  {'type': 'roc'}
]

