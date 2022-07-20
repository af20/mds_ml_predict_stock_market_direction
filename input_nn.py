import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, GRU, BatchNormalization, Reshape, MaxPooling2D, Conv2D, Conv1D, Reshape, MaxPooling1D
from keras import Input

# .............. NEURAL NETWORK MODELS ...................
d_M0 = {
  'id': 0,
  'type': 'normal',
  'v_layers': [
    Dense(12, activation='relu'),
  ]
}

d_M10 = {
  'id': 10,
  'type': 'normal',
  'v_layers': [
    Dense(12, activation='relu'),
    Dropout(0.3),
  ]
}

d_M11 = {
  'id': 11,
  'type': 'normal',
  'v_layers': [
    Dense(12, activation='relu'),
    Dense(6, activation='relu'),
  ]
}

d_M12 = {
  'id': 12,
  'type': 'normal',
  'v_layers': [
    Dense(12, activation='relu'),
    Dropout(0.3),
    Dense(6, activation='relu'),
    Dropout(0.3),
  ]
}



d_M1 = {
  'id': 1,
  'type': 'normal',
  'v_layers': [
    Dense(60, activation="relu"),
    Dense(30, activation="relu"),
    Dense(10, activation="relu"),
  ]
}
'''
  Score for fold 1 => TRAIN: 0.54  |  TEST: 0.43
  Score for fold 2 => TRAIN: 0.49  |  TEST: 0.39
  Score for fold 3 => TRAIN: 0.53  |  TEST: 0.57
  Score for fold 4 => TRAIN: 0.55  |  TEST: 0.5
  Score for fold 5 => TRAIN: 0.57  |  TEST: 0.67
'''

d_M2 = {
  'id': 2,
  'type': 'normal',
  'v_layers': [
    Dense(60, activation="relu"),
    BatchNormalization(),
    Dense(20, activation="relu"),
    BatchNormalization(),
    Dense(10, activation="relu"),
    BatchNormalization(), # TODO prova     Dropout(0.2),
  ]
}

d_M21 = {
  'id': 21,
  'type': 'normal',
  'v_layers': [
    Dense(30, activation="relu"),
    Dropout(0.2), # percentuale di pesi da spegnere, non troppi perchè non ne ho tantissimi
    Dense(10, activation="relu"),
    #Dropout(0.2), # TODO prova
  ]
}# nn va sempre 1

d_M22 = {
  'id': 22,
  'type': 'normal',
  'v_layers': [
    Dense(30, activation="relu"),
    Dropout(0.8), # percentuale di pesi da spegnere, non troppi perchè non ne ho tantissimi
    Dense(10, activation="relu"),
  ]
}# sempre 1




d_M3 = {
  'id': 3,
  'type': 'normal',
  'v_layers': [
    Dense(20, activation="relu", kernel_regularizer='l2'),
    Dense(10, activation="relu", kernel_regularizer='l2')
  ]# sempre 1
}

d_M4 = {
  'id': 4,
  'type': 'special_1',
  'v_layers': [
    LSTM(units=12, return_sequences=True),
    Flatten()
  ]
}

d_M5 = {
  'id': 5,
  'type': 'special_1',
  'v_layers': [
    Dense(100, activation="relu"),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    Dense(20, activation="relu"),
    LSTM(units=10, return_sequences=True),
    Dropout(0.2),
    Flatten(),
    Dense(5, activation="relu"),
    Flatten()
  ]
}

d_M6 = {
  'id': 6,
  'type': 'special_1',
  'v_layers': [
    LSTM(units=300, return_sequences=True),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=10, return_sequences=True),
    Flatten()
  ]
}

d_M7 = {
  'id': 7,
  'type': 'special_1',
  'v_layers': [
    GRU(units=100, return_sequences=True),
    Dropout(0.2),
    GRU(units=50, return_sequences=True),
    Dropout(0.2),
    GRU(units=10, return_sequences=True),
  ]
}



d_M8 = {
  'id': 8,
  'type': 'special_1',
  'v_layers': [
    Conv1D(filters=64, kernel_size=2, activation='relu'), # , input_shape=(n_steps, n_features)
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu')
  ]
}

'''Conv2D(filters=64, kernel_size=7, padding='same', activation='relu'), # if input_shape=[28, 28, 1], make sure you feed the model with data in [n_items,28,28,1]
MaxPooling2D(pool_size=2, strides = 2), # default strides = pool_size
Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
MaxPooling2D(pool_size=2),
Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
MaxPooling2D(pool_size=2),
Flatten(),
Dense(units=128, activation='relu'),
Dense(units=64, activation='relu'),
Dense(units=10, activation='softmax'),'''