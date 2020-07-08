# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/





# open in rl git folder


import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from numpy import loadtxt
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

# to control randomness!
np_seed=123
tf_seed=42

from numpy.random import seed
seed(np_seed)

try:
    from tensorflow.random import set_seed 
    set_seed(tf_seed)
except:
    from tensorflow import set_random_seed # in josephus' machine necessary
    set_random_seed(tf_seed)

def load_data():
    path = "./dlOne"
    col_names = ['Hours'] + ["T" + str(i) for i in range(1, 21)]
    df = pd.read_csv(path, 
                     header = 1, 
                     encoding = "ISO-8859-1", 
                     sep='\t', 
                     skiprows=0).dropna(axis=1).astype(float)
    df.columns = col_names
    df_tem = df.drop(columns = 'Hours')
    return df_tem, df

def load_dataHP():
    path = "./Hpdata"
    #col_names = ['Hours'] + ["T" + str(i) for i in range(1, 21)]
    col_names = ['Hours', 'Tamb', 'ThpOut', 'mLoadFlowRate', 'KJ/hr' , 'cop']
    df_hp = pd.read_csv(path, 
                     header = 1, 
                     encoding = "ISO-8859-1", 
                     sep='\t', 
                     skiprows=0).dropna(axis=1).astype(float)
    df_hp = df_hp.loc[:, df_hp.any()]
    df_hp.columns = col_names
    df_hp = df_hp.drop(columns = 'Hours')
    #df_hp.replace(0,np.nan).dropna(axis=1,how="all")
    #df_tem = df.drop(columns = 'Hours')
    return df_hp
    
def normalize(X):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    scaled_data = scaler.transform(X)
    return scaled_data, scaler

df_T1_20, orig_df = load_data()
df_hp = load_dataHP()


df = pd.concat([df_T1_20, df_hp.loc[:, ["Tamb", "KJ/hr", "cop"]]], axis=1)
df_nrm, scaler = normalize(df)
df_nrm = pd.DataFrame(df_nrm)
df_nrm.columns = df.columns

df_input = df_nrm.loc[:, ["T"+str(i+1) for i in range(0, 20)] + ["Tamb"]]
df_output = df_nrm.loc[:, ["T"+str(i+1) for i in range(0, 20)] + ["KJ/hr", "cop"]]

##############################################
# for LSTM taking k last time-point m values create an ANN
##############################################

k = 3
epochs=600
batch_size=100


###############################################
# take a data frame and generate sample input and output data
###############################################

"""
Let's say data frame has n_rows and n_cols = n_values
n_rows, n_cols = df.shape

"""
def flatten_row_wise(df):
    """Take row by row and attach to one flat single row."""
    return np.ndarray.flatten(np.array(df))

def prepare_df(df):
    n_rows, n_cols = df.shape
    new_rows = np.array([flatten_row_wise(df.iloc[(i-k):i]) for i in range(k, n_rows)])
    new_ys = np.array([row for row in df.iloc[(k):, :].itertuples(index=False)])
    return new_rows, new_ys

X_input, y_input = prepare_df(df_input)
X_output, y_output = prepare_df(df_output)


n_input_features = X_input.shape[1]
n_output_features = y_output.shape[1]

################################################
# train test split only indexes
################################################

def train_test_split_indexes(X, y, test_size=0.2, random_state=42, shuffle=False):
    X_train_indexes, X_test_indexes, y_train_indexes, y_test_indexes = train_test_split(pd.DataFrame(list(range(X.shape[0]))),
                                                                                        pd.DataFrame(list(range(y.shape[0]))),
                                                                                        test_size=test_size,
                                                                                        random_state=random_state,
                                                                                        shuffle=shuffle)
    train_indexes, test_indexes = [x for x in X_train_indexes.iloc[:, 0]]  , [x for x in X_test_indexes.iloc[:, 0]]
    return sorted(train_indexes), sorted(test_indexes)


train_indexes, test_indexes = train_test_split_indexes(X_input, y_input, test_size=0.1, random_state=42, shuffle=True)

X_input_train = X_input[train_indexes, :]
X_input_test  = X_input[test_indexes, :]
y_output_train = y_output[train_indexes, :]
y_output_test  = y_output[test_indexes, :]



def create_model(time_steps, n_input_features, n_output_features):
    model = Sequential()
    model.add(LSTM(3, input_shape = (time_steps, n_input_features // time_steps)))
    model.add(Dense(n_output_features, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

model = create_model(k, n_input_features, n_output_features)
model_fpath="lstmHP.h5"
callbacks_list = [ ModelCheckpoint(filepath=model_fpath,
                                   monitor="val_loss",
                                   save_best_only=True,
                                   mode="min")]


history = model.fit(X_input_train.reshape(X_input_train.shape[0], k, n_input_features // k),
                    y_output_train.reshape(y_output_train.shape[0], n_output_features),
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.3,
                    callbacks = callbacks_list,
                    verbose=1)

print(f"Best val_loss is: {min(history.history['val_loss'])}")




# to plot make data frame out of dict history.history and use .plot() method
pd.DataFrame(history.history).plot()
plt.show()


###########################################
# plot prediction
###########################################
model = load_model(model_fpath) # this loads the best saved model!
yhat=model.predict(X_input_test.reshape(X_input_test.shape[0], k, n_input_features//k))


def unscale(y_values, scaler):
   # it is more complex, because y_values here miss the Tamb column
   # we add artificial Tamb column and remove it again.
   y_values_with_Tamb_fake = np.array([np.concatenate([row[:-2], np.array([1.0]), row[-2:]]) for row in y_values])
   res = scaler.inverse_transform(y_values_with_Tamb_fake)
   return np.array([np.concatenate([row[:-3], row[-2:]]) for row in res])

y_pred_unscaled, y_test_unscaled = unscale(yhat, scaler), unscale(y_output_test, scaler)

def plot(arr_y_pred, arr_y_test, orig_df):
   xdata = orig_df.iloc[4609:, 0]
   df_y_pred = pd.DataFrame(arr_y_pred)
   df_y_test = pd.DataFrame(arr_y_test) # arry_y_pred you took here!
   legends_test =['OrgT' + str(i) for i in range (0, 22) ]
   legends_pred =['PrT' + str(i) for i in range (0, 22) ]
   for i, j in zip(df_y_pred, df_y_test):
       plt.scatter(xdata, df_y_pred.iloc[:, i], label = legends_pred[i])
       plt.scatter(xdata, df_y_test.iloc[:, j], label = legends_test[i])

   plt.legend()
   plt.show()
   return

plt.clf()
plot(y_pred_unscaled, y_test_unscaled, orig_df)















################################################################################

# def create_model(time_steps, n_input_features, n_output_features):
#     model = Sequential()
#     model.add(LSTM(10, input_shape = (time_steps, n_input_features // time_steps)))
#     model.add(Dense(50, activation='relu'))
#     model.add(Dense(n_output_features, activation='linear'))
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
#     return model
# 
# model = create_model(k, n_input_features, n_output_features)
# model_fpath="lstmHP.h5"
# callbacks_list = [ ModelCheckpoint(filepath=model_fpath,
#                                    monitor="val_loss",
#                                    save_best_only=True,
#                                    mode="min")]
# 
# 
# history = model.fit(X_input_train.reshape(X_input_train.shape[0], k, n_input_features // k),
#                     y_output_train.reshape(y_output_train.shape[0], n_output_features),
#                     epochs=epochs,
#                     batch_size=batch_size,
#                     validation_split=0.3,
#                     callbacks = callbacks_list,
#                     verbose=1)
# 
# 
# 
# Train on 3224 samples, validate on 1382 samples
# Epoch 1/600
# 3224/3224 [==============================] - 3s 974us/step - loss: 0.3650 - mean_squared_error: 0.3650 - val_loss: 0.2232 - val_mean_squared_error: 0.2232
# Epoch 2/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.1610 - mean_squared_error: 0.1610 - val_loss: 0.0896 - val_mean_squared_error: 0.0896
# Epoch 3/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0920 - mean_squared_error: 0.0920 - val_loss: 0.0735 - val_mean_squared_error: 0.0735
# Epoch 4/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0731 - mean_squared_error: 0.0731 - val_loss: 0.0505 - val_mean_squared_error: 0.0505
# Epoch 5/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0526 - mean_squared_error: 0.0526 - val_loss: 0.0347 - val_mean_squared_error: 0.0347
# Epoch 6/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0432 - mean_squared_error: 0.0432 - val_loss: 0.0313 - val_mean_squared_error: 0.0313
# Epoch 7/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0405 - mean_squared_error: 0.0405 - val_loss: 0.0312 - val_mean_squared_error: 0.0312
# Epoch 8/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0393 - mean_squared_error: 0.0393 - val_loss: 0.0282 - val_mean_squared_error: 0.0282
# Epoch 9/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0384 - mean_squared_error: 0.0384 - val_loss: 0.0276 - val_mean_squared_error: 0.0276
# Epoch 10/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0373 - mean_squared_error: 0.0373 - val_loss: 0.0267 - val_mean_squared_error: 0.0267
# Epoch 11/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0362 - mean_squared_error: 0.0362 - val_loss: 0.0249 - val_mean_squared_error: 0.0249
# Epoch 12/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0351 - mean_squared_error: 0.0351 - val_loss: 0.0241 - val_mean_squared_error: 0.0241
# Epoch 13/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0338 - mean_squared_error: 0.0338 - val_loss: 0.0238 - val_mean_squared_error: 0.0238
# Epoch 14/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0324 - mean_squared_error: 0.0324 - val_loss: 0.0204 - val_mean_squared_error: 0.0204
# Epoch 15/600
# 3224/3224 [==============================] - 0s 68us/step - loss: 0.0305 - mean_squared_error: 0.0305 - val_loss: 0.0194 - val_mean_squared_error: 0.0194
# Epoch 16/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0294 - mean_squared_error: 0.0294 - val_loss: 0.0164 - val_mean_squared_error: 0.0164
# Epoch 17/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0280 - mean_squared_error: 0.0280 - val_loss: 0.0158 - val_mean_squared_error: 0.0158
# Epoch 18/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0270 - mean_squared_error: 0.0270 - val_loss: 0.0149 - val_mean_squared_error: 0.0149
# Epoch 19/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0264 - mean_squared_error: 0.0264 - val_loss: 0.0139 - val_mean_squared_error: 0.0139
# Epoch 20/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0256 - mean_squared_error: 0.0256 - val_loss: 0.0133 - val_mean_squared_error: 0.0133
# Epoch 21/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0250 - mean_squared_error: 0.0250 - val_loss: 0.0129 - val_mean_squared_error: 0.0129
# Epoch 22/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0240 - mean_squared_error: 0.0240 - val_loss: 0.0130 - val_mean_squared_error: 0.0130
# Epoch 23/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0235 - mean_squared_error: 0.0235 - val_loss: 0.0130 - val_mean_squared_error: 0.0130
# Epoch 24/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0239 - mean_squared_error: 0.0239 - val_loss: 0.0157 - val_mean_squared_error: 0.0157
# Epoch 25/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0229 - mean_squared_error: 0.0229 - val_loss: 0.0140 - val_mean_squared_error: 0.0140
# Epoch 26/600
# 3224/3224 [==============================] - 0s 67us/step - loss: 0.0219 - mean_squared_error: 0.0219 - val_loss: 0.0192 - val_mean_squared_error: 0.0192
# Epoch 27/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0216 - mean_squared_error: 0.0216 - val_loss: 0.0163 - val_mean_squared_error: 0.0163
# Epoch 28/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0208 - mean_squared_error: 0.0208 - val_loss: 0.0154 - val_mean_squared_error: 0.0154
# Epoch 29/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0208 - mean_squared_error: 0.0208 - val_loss: 0.0149 - val_mean_squared_error: 0.0149
# Epoch 30/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0203 - mean_squared_error: 0.0203 - val_loss: 0.0164 - val_mean_squared_error: 0.0164
# Epoch 31/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0200 - mean_squared_error: 0.0200 - val_loss: 0.0176 - val_mean_squared_error: 0.0176
# Epoch 32/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0199 - mean_squared_error: 0.0199 - val_loss: 0.0157 - val_mean_squared_error: 0.0157
# Epoch 33/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0201 - mean_squared_error: 0.0201 - val_loss: 0.0118 - val_mean_squared_error: 0.0118
# Epoch 34/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0201 - mean_squared_error: 0.0201 - val_loss: 0.0163 - val_mean_squared_error: 0.0163
# Epoch 35/600
# 3224/3224 [==============================] - 0s 42us/step - loss: 0.0194 - mean_squared_error: 0.0194 - val_loss: 0.0130 - val_mean_squared_error: 0.0130
# Epoch 36/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0199 - mean_squared_error: 0.0199 - val_loss: 0.0148 - val_mean_squared_error: 0.0148
# Epoch 37/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0194 - mean_squared_error: 0.0194 - val_loss: 0.0122 - val_mean_squared_error: 0.0122
# Epoch 38/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0197 - mean_squared_error: 0.0197 - val_loss: 0.0114 - val_mean_squared_error: 0.0114
# Epoch 39/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0191 - mean_squared_error: 0.0191 - val_loss: 0.0130 - val_mean_squared_error: 0.0130
# Epoch 40/600
# 3224/3224 [==============================] - 0s 55us/step - loss: 0.0195 - mean_squared_error: 0.0195 - val_loss: 0.0110 - val_mean_squared_error: 0.0110
# Epoch 41/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0189 - mean_squared_error: 0.0189 - val_loss: 0.0108 - val_mean_squared_error: 0.0108
# Epoch 42/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0186 - mean_squared_error: 0.0186 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 43/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0188 - mean_squared_error: 0.0188 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 44/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0198 - mean_squared_error: 0.0198 - val_loss: 0.0112 - val_mean_squared_error: 0.0112
# Epoch 45/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0194 - mean_squared_error: 0.0194 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 46/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0189 - mean_squared_error: 0.0189 - val_loss: 0.0111 - val_mean_squared_error: 0.0111
# Epoch 47/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0184 - mean_squared_error: 0.0184 - val_loss: 0.0123 - val_mean_squared_error: 0.0123
# Epoch 48/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0180 - mean_squared_error: 0.0180 - val_loss: 0.0100 - val_mean_squared_error: 0.0100
# Epoch 49/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0182 - mean_squared_error: 0.0182 - val_loss: 0.0108 - val_mean_squared_error: 0.0108
# Epoch 50/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0180 - mean_squared_error: 0.0180 - val_loss: 0.0114 - val_mean_squared_error: 0.0114
# Epoch 51/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0178 - mean_squared_error: 0.0178 - val_loss: 0.0113 - val_mean_squared_error: 0.0113
# Epoch 52/600
# 3224/3224 [==============================] - 0s 71us/step - loss: 0.0177 - mean_squared_error: 0.0177 - val_loss: 0.0111 - val_mean_squared_error: 0.0111
# Epoch 53/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0180 - mean_squared_error: 0.0180 - val_loss: 0.0108 - val_mean_squared_error: 0.0108
# Epoch 54/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0178 - mean_squared_error: 0.0178 - val_loss: 0.0112 - val_mean_squared_error: 0.0112
# Epoch 55/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0178 - mean_squared_error: 0.0178 - val_loss: 0.0115 - val_mean_squared_error: 0.0115
# Epoch 56/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0178 - mean_squared_error: 0.0178 - val_loss: 0.0108 - val_mean_squared_error: 0.0108
# Epoch 57/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0176 - mean_squared_error: 0.0176 - val_loss: 0.0125 - val_mean_squared_error: 0.0125
# Epoch 58/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0173 - mean_squared_error: 0.0173 - val_loss: 0.0124 - val_mean_squared_error: 0.0124
# Epoch 59/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0174 - mean_squared_error: 0.0174 - val_loss: 0.0109 - val_mean_squared_error: 0.0109
# Epoch 60/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0174 - mean_squared_error: 0.0174 - val_loss: 0.0102 - val_mean_squared_error: 0.0102
# Epoch 61/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0172 - mean_squared_error: 0.0172 - val_loss: 0.0107 - val_mean_squared_error: 0.0107
# Epoch 62/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0172 - mean_squared_error: 0.0172 - val_loss: 0.0107 - val_mean_squared_error: 0.0107
# Epoch 63/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0178 - mean_squared_error: 0.0178 - val_loss: 0.0122 - val_mean_squared_error: 0.0122
# Epoch 64/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0174 - mean_squared_error: 0.0174 - val_loss: 0.0116 - val_mean_squared_error: 0.0116
# Epoch 65/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0173 - mean_squared_error: 0.0173 - val_loss: 0.0117 - val_mean_squared_error: 0.0117
# Epoch 66/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0172 - mean_squared_error: 0.0172 - val_loss: 0.0111 - val_mean_squared_error: 0.0111
# Epoch 67/600
# 3224/3224 [==============================] - 0s 56us/step - loss: 0.0171 - mean_squared_error: 0.0171 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 68/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0169 - mean_squared_error: 0.0169 - val_loss: 0.0111 - val_mean_squared_error: 0.0111
# Epoch 69/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0165 - mean_squared_error: 0.0165 - val_loss: 0.0111 - val_mean_squared_error: 0.0111
# Epoch 70/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0170 - mean_squared_error: 0.0170 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 71/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0178 - mean_squared_error: 0.0178 - val_loss: 0.0125 - val_mean_squared_error: 0.0125
# Epoch 72/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0167 - mean_squared_error: 0.0167 - val_loss: 0.0118 - val_mean_squared_error: 0.0118
# Epoch 73/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0170 - mean_squared_error: 0.0170 - val_loss: 0.0099 - val_mean_squared_error: 0.0099
# Epoch 74/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0163 - mean_squared_error: 0.0163 - val_loss: 0.0125 - val_mean_squared_error: 0.0125
# Epoch 75/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0168 - mean_squared_error: 0.0168 - val_loss: 0.0111 - val_mean_squared_error: 0.0111
# Epoch 76/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0165 - mean_squared_error: 0.0165 - val_loss: 0.0140 - val_mean_squared_error: 0.0140
# Epoch 77/600
# 3224/3224 [==============================] - 0s 78us/step - loss: 0.0166 - mean_squared_error: 0.0166 - val_loss: 0.0100 - val_mean_squared_error: 0.0100
# Epoch 78/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0163 - mean_squared_error: 0.0163 - val_loss: 0.0115 - val_mean_squared_error: 0.0115
# Epoch 79/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0166 - mean_squared_error: 0.0166 - val_loss: 0.0109 - val_mean_squared_error: 0.0109
# Epoch 80/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0174 - mean_squared_error: 0.0174 - val_loss: 0.0160 - val_mean_squared_error: 0.0160
# Epoch 81/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0166 - mean_squared_error: 0.0166 - val_loss: 0.0104 - val_mean_squared_error: 0.0104
# Epoch 82/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0163 - mean_squared_error: 0.0163 - val_loss: 0.0102 - val_mean_squared_error: 0.0102
# Epoch 83/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0160 - mean_squared_error: 0.0160 - val_loss: 0.0122 - val_mean_squared_error: 0.0122
# Epoch 84/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0167 - mean_squared_error: 0.0167 - val_loss: 0.0133 - val_mean_squared_error: 0.0133
# Epoch 85/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0169 - mean_squared_error: 0.0169 - val_loss: 0.0118 - val_mean_squared_error: 0.0118
# Epoch 86/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0162 - mean_squared_error: 0.0162 - val_loss: 0.0115 - val_mean_squared_error: 0.0115
# Epoch 87/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0158 - mean_squared_error: 0.0158 - val_loss: 0.0104 - val_mean_squared_error: 0.0104
# Epoch 88/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0159 - mean_squared_error: 0.0159 - val_loss: 0.0126 - val_mean_squared_error: 0.0126
# Epoch 89/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0160 - mean_squared_error: 0.0160 - val_loss: 0.0109 - val_mean_squared_error: 0.0109
# Epoch 90/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0156 - mean_squared_error: 0.0156 - val_loss: 0.0112 - val_mean_squared_error: 0.0112
# Epoch 91/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0155 - mean_squared_error: 0.0155 - val_loss: 0.0113 - val_mean_squared_error: 0.0113
# Epoch 92/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0160 - mean_squared_error: 0.0160 - val_loss: 0.0125 - val_mean_squared_error: 0.0125
# Epoch 93/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0157 - mean_squared_error: 0.0157 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 94/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0164 - mean_squared_error: 0.0164 - val_loss: 0.0111 - val_mean_squared_error: 0.0111
# Epoch 95/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0161 - mean_squared_error: 0.0161 - val_loss: 0.0138 - val_mean_squared_error: 0.0138
# Epoch 96/600
# 3224/3224 [==============================] - 0s 73us/step - loss: 0.0154 - mean_squared_error: 0.0154 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 97/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0155 - mean_squared_error: 0.0155 - val_loss: 0.0114 - val_mean_squared_error: 0.0114
# Epoch 98/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0155 - mean_squared_error: 0.0155 - val_loss: 0.0109 - val_mean_squared_error: 0.0109
# Epoch 99/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0150 - mean_squared_error: 0.0150 - val_loss: 0.0137 - val_mean_squared_error: 0.0137
# Epoch 100/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0151 - mean_squared_error: 0.0151 - val_loss: 0.0102 - val_mean_squared_error: 0.0102
# Epoch 101/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0151 - mean_squared_error: 0.0151 - val_loss: 0.0116 - val_mean_squared_error: 0.0116
# Epoch 102/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0152 - mean_squared_error: 0.0152 - val_loss: 0.0112 - val_mean_squared_error: 0.0112
# Epoch 103/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0152 - mean_squared_error: 0.0152 - val_loss: 0.0103 - val_mean_squared_error: 0.0103
# Epoch 104/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0151 - mean_squared_error: 0.0151 - val_loss: 0.0119 - val_mean_squared_error: 0.0119
# Epoch 105/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0149 - mean_squared_error: 0.0149 - val_loss: 0.0121 - val_mean_squared_error: 0.0121
# Epoch 106/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0149 - mean_squared_error: 0.0149 - val_loss: 0.0118 - val_mean_squared_error: 0.0118
# Epoch 107/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0147 - mean_squared_error: 0.0147 - val_loss: 0.0118 - val_mean_squared_error: 0.0118
# Epoch 108/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0158 - mean_squared_error: 0.0158 - val_loss: 0.0137 - val_mean_squared_error: 0.0137
# Epoch 109/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0155 - mean_squared_error: 0.0155 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 110/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0150 - mean_squared_error: 0.0150 - val_loss: 0.0107 - val_mean_squared_error: 0.0107
# Epoch 111/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0155 - mean_squared_error: 0.0155 - val_loss: 0.0104 - val_mean_squared_error: 0.0104
# Epoch 112/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0148 - mean_squared_error: 0.0148 - val_loss: 0.0114 - val_mean_squared_error: 0.0114
# Epoch 113/600
# 3224/3224 [==============================] - 0s 76us/step - loss: 0.0151 - mean_squared_error: 0.0151 - val_loss: 0.0114 - val_mean_squared_error: 0.0114
# Epoch 114/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0144 - mean_squared_error: 0.0144 - val_loss: 0.0102 - val_mean_squared_error: 0.0102
# Epoch 115/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0147 - mean_squared_error: 0.0147 - val_loss: 0.0093 - val_mean_squared_error: 0.0093
# Epoch 116/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0147 - mean_squared_error: 0.0147 - val_loss: 0.0119 - val_mean_squared_error: 0.0119
# Epoch 117/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0155 - mean_squared_error: 0.0155 - val_loss: 0.0099 - val_mean_squared_error: 0.0099
# Epoch 118/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0147 - mean_squared_error: 0.0147 - val_loss: 0.0133 - val_mean_squared_error: 0.0133
# Epoch 119/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0145 - mean_squared_error: 0.0145 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 120/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0141 - mean_squared_error: 0.0141 - val_loss: 0.0131 - val_mean_squared_error: 0.0131
# Epoch 121/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0143 - mean_squared_error: 0.0143 - val_loss: 0.0116 - val_mean_squared_error: 0.0116
# Epoch 122/600
# 3224/3224 [==============================] - 0s 56us/step - loss: 0.0145 - mean_squared_error: 0.0145 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 123/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0145 - mean_squared_error: 0.0145 - val_loss: 0.0118 - val_mean_squared_error: 0.0118
# Epoch 124/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0143 - mean_squared_error: 0.0143 - val_loss: 0.0113 - val_mean_squared_error: 0.0113
# Epoch 125/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0140 - mean_squared_error: 0.0140 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 126/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0139 - mean_squared_error: 0.0139 - val_loss: 0.0129 - val_mean_squared_error: 0.0129
# Epoch 127/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0143 - mean_squared_error: 0.0143 - val_loss: 0.0113 - val_mean_squared_error: 0.0113
# Epoch 128/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0157 - mean_squared_error: 0.0157 - val_loss: 0.0105 - val_mean_squared_error: 0.0105
# Epoch 129/600
# 3224/3224 [==============================] - 0s 56us/step - loss: 0.0149 - mean_squared_error: 0.0149 - val_loss: 0.0103 - val_mean_squared_error: 0.0103
# Epoch 130/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0142 - mean_squared_error: 0.0142 - val_loss: 0.0124 - val_mean_squared_error: 0.0124
# Epoch 131/600
# 3224/3224 [==============================] - 0s 58us/step - loss: 0.0141 - mean_squared_error: 0.0141 - val_loss: 0.0117 - val_mean_squared_error: 0.0117
# Epoch 132/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0148 - mean_squared_error: 0.0148 - val_loss: 0.0113 - val_mean_squared_error: 0.0113
# Epoch 133/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.0123 - val_mean_squared_error: 0.0123
# Epoch 134/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0152 - mean_squared_error: 0.0152 - val_loss: 0.0104 - val_mean_squared_error: 0.0104
# Epoch 135/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0139 - mean_squared_error: 0.0139 - val_loss: 0.0111 - val_mean_squared_error: 0.0111
# Epoch 136/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0151 - mean_squared_error: 0.0151 - val_loss: 0.0162 - val_mean_squared_error: 0.0162
# Epoch 137/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0148 - mean_squared_error: 0.0148 - val_loss: 0.0117 - val_mean_squared_error: 0.0117
# Epoch 138/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0138 - mean_squared_error: 0.0138 - val_loss: 0.0122 - val_mean_squared_error: 0.0122
# Epoch 139/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0137 - mean_squared_error: 0.0137 - val_loss: 0.0115 - val_mean_squared_error: 0.0115
# Epoch 140/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0135 - mean_squared_error: 0.0135 - val_loss: 0.0115 - val_mean_squared_error: 0.0115
# Epoch 141/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0139 - mean_squared_error: 0.0139 - val_loss: 0.0117 - val_mean_squared_error: 0.0117
# Epoch 142/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0138 - mean_squared_error: 0.0138 - val_loss: 0.0119 - val_mean_squared_error: 0.0119
# Epoch 143/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0134 - mean_squared_error: 0.0134 - val_loss: 0.0103 - val_mean_squared_error: 0.0103
# Epoch 144/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0139 - mean_squared_error: 0.0139 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 145/600
# 3224/3224 [==============================] - 0s 73us/step - loss: 0.0135 - mean_squared_error: 0.0135 - val_loss: 0.0117 - val_mean_squared_error: 0.0117
# Epoch 146/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.0114 - val_mean_squared_error: 0.0114
# Epoch 147/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0134 - mean_squared_error: 0.0134 - val_loss: 0.0101 - val_mean_squared_error: 0.0101
# Epoch 148/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.0134 - val_mean_squared_error: 0.0134
# Epoch 149/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0143 - mean_squared_error: 0.0143 - val_loss: 0.0114 - val_mean_squared_error: 0.0114
# Epoch 150/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.0121 - val_mean_squared_error: 0.0121
# Epoch 151/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0134 - mean_squared_error: 0.0134 - val_loss: 0.0121 - val_mean_squared_error: 0.0121
# Epoch 152/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0131 - mean_squared_error: 0.0131 - val_loss: 0.0115 - val_mean_squared_error: 0.0115
# Epoch 153/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0131 - mean_squared_error: 0.0131 - val_loss: 0.0116 - val_mean_squared_error: 0.0116
# Epoch 154/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0135 - mean_squared_error: 0.0135 - val_loss: 0.0124 - val_mean_squared_error: 0.0124
# Epoch 155/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0129 - mean_squared_error: 0.0129 - val_loss: 0.0111 - val_mean_squared_error: 0.0111
# Epoch 156/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.0106 - val_mean_squared_error: 0.0106
# Epoch 157/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0135 - mean_squared_error: 0.0135 - val_loss: 0.0122 - val_mean_squared_error: 0.0122
# Epoch 158/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0137 - mean_squared_error: 0.0137 - val_loss: 0.0130 - val_mean_squared_error: 0.0130
# Epoch 159/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0136 - mean_squared_error: 0.0136 - val_loss: 0.0129 - val_mean_squared_error: 0.0129
# Epoch 160/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0128 - mean_squared_error: 0.0128 - val_loss: 0.0122 - val_mean_squared_error: 0.0122
# Epoch 161/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0128 - mean_squared_error: 0.0128 - val_loss: 0.0138 - val_mean_squared_error: 0.0138
# Epoch 162/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0128 - mean_squared_error: 0.0128 - val_loss: 0.0125 - val_mean_squared_error: 0.0125
# Epoch 163/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0126 - mean_squared_error: 0.0126 - val_loss: 0.0123 - val_mean_squared_error: 0.0123
# Epoch 164/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0141 - mean_squared_error: 0.0141 - val_loss: 0.0123 - val_mean_squared_error: 0.0123
# Epoch 165/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0156 - mean_squared_error: 0.0156 - val_loss: 0.0130 - val_mean_squared_error: 0.0130
# Epoch 166/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0132 - mean_squared_error: 0.0132 - val_loss: 0.0111 - val_mean_squared_error: 0.0111
# Epoch 167/600
# 3224/3224 [==============================] - 0s 60us/step - loss: 0.0126 - mean_squared_error: 0.0126 - val_loss: 0.0115 - val_mean_squared_error: 0.0115
# Epoch 168/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0133 - mean_squared_error: 0.0133 - val_loss: 0.0103 - val_mean_squared_error: 0.0103
# Epoch 169/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0130 - mean_squared_error: 0.0130 - val_loss: 0.0130 - val_mean_squared_error: 0.0130
# Epoch 170/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0141 - mean_squared_error: 0.0141 - val_loss: 0.0107 - val_mean_squared_error: 0.0107
# Epoch 171/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0134 - mean_squared_error: 0.0134 - val_loss: 0.0110 - val_mean_squared_error: 0.0110
# Epoch 172/600
# 3224/3224 [==============================] - 0s 55us/step - loss: 0.0127 - mean_squared_error: 0.0127 - val_loss: 0.0129 - val_mean_squared_error: 0.0129
# Epoch 173/600
# 3224/3224 [==============================] - 0s 68us/step - loss: 0.0125 - mean_squared_error: 0.0125 - val_loss: 0.0115 - val_mean_squared_error: 0.0115
# Epoch 174/600
# 3224/3224 [==============================] - 0s 57us/step - loss: 0.0127 - mean_squared_error: 0.0127 - val_loss: 0.0114 - val_mean_squared_error: 0.0114
# Epoch 175/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0127 - mean_squared_error: 0.0127 - val_loss: 0.0133 - val_mean_squared_error: 0.0133
# Epoch 176/600
# 3224/3224 [==============================] - 0s 56us/step - loss: 0.0124 - mean_squared_error: 0.0124 - val_loss: 0.0114 - val_mean_squared_error: 0.0114
# Epoch 177/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0123 - mean_squared_error: 0.0123 - val_loss: 0.0114 - val_mean_squared_error: 0.0114
# Epoch 178/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0124 - mean_squared_error: 0.0124 - val_loss: 0.0130 - val_mean_squared_error: 0.0130
# Epoch 179/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0127 - mean_squared_error: 0.0127 - val_loss: 0.0121 - val_mean_squared_error: 0.0121
# Epoch 180/600
# 3224/3224 [==============================] - 0s 56us/step - loss: 0.0134 - mean_squared_error: 0.0134 - val_loss: 0.0137 - val_mean_squared_error: 0.0137
# Epoch 181/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0122 - mean_squared_error: 0.0122 - val_loss: 0.0133 - val_mean_squared_error: 0.0133
# Epoch 182/600
# 3224/3224 [==============================] - 0s 68us/step - loss: 0.0126 - mean_squared_error: 0.0126 - val_loss: 0.0118 - val_mean_squared_error: 0.0118
# Epoch 183/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0120 - mean_squared_error: 0.0120 - val_loss: 0.0138 - val_mean_squared_error: 0.0138
# Epoch 184/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0119 - mean_squared_error: 0.0119 - val_loss: 0.0120 - val_mean_squared_error: 0.0120
# Epoch 185/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0122 - mean_squared_error: 0.0122 - val_loss: 0.0130 - val_mean_squared_error: 0.0130
# Epoch 186/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0120 - mean_squared_error: 0.0120 - val_loss: 0.0131 - val_mean_squared_error: 0.0131
# Epoch 187/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0119 - mean_squared_error: 0.0119 - val_loss: 0.0144 - val_mean_squared_error: 0.0144
# Epoch 188/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0126 - mean_squared_error: 0.0126 - val_loss: 0.0137 - val_mean_squared_error: 0.0137
# Epoch 189/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0118 - mean_squared_error: 0.0118 - val_loss: 0.0128 - val_mean_squared_error: 0.0128
# Epoch 190/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0125 - mean_squared_error: 0.0125 - val_loss: 0.0143 - val_mean_squared_error: 0.0143
# Epoch 191/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0118 - mean_squared_error: 0.0118 - val_loss: 0.0126 - val_mean_squared_error: 0.0126
# Epoch 192/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0119 - mean_squared_error: 0.0119 - val_loss: 0.0128 - val_mean_squared_error: 0.0128
# Epoch 193/600
# 3224/3224 [==============================] - 0s 65us/step - loss: 0.0118 - mean_squared_error: 0.0118 - val_loss: 0.0149 - val_mean_squared_error: 0.0149
# Epoch 194/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0118 - mean_squared_error: 0.0118 - val_loss: 0.0124 - val_mean_squared_error: 0.0124
# Epoch 195/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0125 - mean_squared_error: 0.0125 - val_loss: 0.0127 - val_mean_squared_error: 0.0127
# Epoch 196/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0123 - mean_squared_error: 0.0123 - val_loss: 0.0136 - val_mean_squared_error: 0.0136
# Epoch 197/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0122 - mean_squared_error: 0.0122 - val_loss: 0.0114 - val_mean_squared_error: 0.0114
# Epoch 198/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0118 - mean_squared_error: 0.0118 - val_loss: 0.0131 - val_mean_squared_error: 0.0131
# Epoch 199/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0113 - mean_squared_error: 0.0113 - val_loss: 0.0156 - val_mean_squared_error: 0.0156
# Epoch 200/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0113 - mean_squared_error: 0.0113 - val_loss: 0.0132 - val_mean_squared_error: 0.0132
# Epoch 201/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0117 - mean_squared_error: 0.0117 - val_loss: 0.0148 - val_mean_squared_error: 0.0148
# Epoch 202/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0123 - mean_squared_error: 0.0123 - val_loss: 0.0120 - val_mean_squared_error: 0.0120
# Epoch 203/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0123 - mean_squared_error: 0.0123 - val_loss: 0.0128 - val_mean_squared_error: 0.0128
# Epoch 204/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0114 - mean_squared_error: 0.0114 - val_loss: 0.0143 - val_mean_squared_error: 0.0143
# Epoch 205/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0118 - mean_squared_error: 0.0118 - val_loss: 0.0131 - val_mean_squared_error: 0.0131
# Epoch 206/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0111 - mean_squared_error: 0.0111 - val_loss: 0.0146 - val_mean_squared_error: 0.0146
# Epoch 207/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0110 - mean_squared_error: 0.0110 - val_loss: 0.0115 - val_mean_squared_error: 0.0115
# Epoch 208/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0121 - mean_squared_error: 0.0121 - val_loss: 0.0136 - val_mean_squared_error: 0.0136
# Epoch 209/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0112 - mean_squared_error: 0.0112 - val_loss: 0.0144 - val_mean_squared_error: 0.0144
# Epoch 210/600
# 3224/3224 [==============================] - 0s 62us/step - loss: 0.0116 - mean_squared_error: 0.0116 - val_loss: 0.0130 - val_mean_squared_error: 0.0130
# Epoch 211/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0112 - mean_squared_error: 0.0112 - val_loss: 0.0147 - val_mean_squared_error: 0.0147
# Epoch 212/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0115 - mean_squared_error: 0.0115 - val_loss: 0.0133 - val_mean_squared_error: 0.0133
# Epoch 213/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0114 - mean_squared_error: 0.0114 - val_loss: 0.0123 - val_mean_squared_error: 0.0123
# Epoch 214/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0112 - mean_squared_error: 0.0112 - val_loss: 0.0148 - val_mean_squared_error: 0.0148
# Epoch 215/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0111 - mean_squared_error: 0.0111 - val_loss: 0.0156 - val_mean_squared_error: 0.0156
# Epoch 216/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0112 - mean_squared_error: 0.0112 - val_loss: 0.0148 - val_mean_squared_error: 0.0148
# Epoch 217/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0129 - mean_squared_error: 0.0129 - val_loss: 0.0160 - val_mean_squared_error: 0.0160
# Epoch 218/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0131 - mean_squared_error: 0.0131 - val_loss: 0.0144 - val_mean_squared_error: 0.0144
# Epoch 219/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0113 - mean_squared_error: 0.0113 - val_loss: 0.0122 - val_mean_squared_error: 0.0122
# Epoch 220/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0109 - mean_squared_error: 0.0109 - val_loss: 0.0141 - val_mean_squared_error: 0.0141
# Epoch 221/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0110 - mean_squared_error: 0.0110 - val_loss: 0.0146 - val_mean_squared_error: 0.0146
# Epoch 222/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0108 - mean_squared_error: 0.0108 - val_loss: 0.0161 - val_mean_squared_error: 0.0161
# Epoch 223/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0123 - mean_squared_error: 0.0123 - val_loss: 0.0117 - val_mean_squared_error: 0.0117
# Epoch 224/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0111 - mean_squared_error: 0.0111 - val_loss: 0.0132 - val_mean_squared_error: 0.0132
# Epoch 225/600
# 3224/3224 [==============================] - 0s 58us/step - loss: 0.0113 - mean_squared_error: 0.0113 - val_loss: 0.0147 - val_mean_squared_error: 0.0147
# Epoch 226/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0111 - mean_squared_error: 0.0111 - val_loss: 0.0145 - val_mean_squared_error: 0.0145
# Epoch 227/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0110 - mean_squared_error: 0.0110 - val_loss: 0.0156 - val_mean_squared_error: 0.0156
# Epoch 228/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0115 - mean_squared_error: 0.0115 - val_loss: 0.0128 - val_mean_squared_error: 0.0128
# Epoch 229/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0107 - mean_squared_error: 0.0107 - val_loss: 0.0126 - val_mean_squared_error: 0.0126
# Epoch 230/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0105 - mean_squared_error: 0.0105 - val_loss: 0.0148 - val_mean_squared_error: 0.0148
# Epoch 231/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0107 - mean_squared_error: 0.0107 - val_loss: 0.0148 - val_mean_squared_error: 0.0148
# Epoch 232/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0105 - mean_squared_error: 0.0105 - val_loss: 0.0151 - val_mean_squared_error: 0.0151
# Epoch 233/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0108 - mean_squared_error: 0.0108 - val_loss: 0.0139 - val_mean_squared_error: 0.0139
# Epoch 234/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0107 - mean_squared_error: 0.0107 - val_loss: 0.0168 - val_mean_squared_error: 0.0168
# Epoch 235/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0107 - mean_squared_error: 0.0107 - val_loss: 0.0146 - val_mean_squared_error: 0.0146
# Epoch 236/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0104 - mean_squared_error: 0.0104 - val_loss: 0.0136 - val_mean_squared_error: 0.0136
# Epoch 237/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0105 - mean_squared_error: 0.0105 - val_loss: 0.0151 - val_mean_squared_error: 0.0151
# Epoch 238/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0101 - mean_squared_error: 0.0101 - val_loss: 0.0152 - val_mean_squared_error: 0.0152
# Epoch 239/600
# 3224/3224 [==============================] - 0s 70us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.0146 - val_mean_squared_error: 0.0146
# Epoch 240/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0102 - mean_squared_error: 0.0102 - val_loss: 0.0156 - val_mean_squared_error: 0.0156
# Epoch 241/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0105 - mean_squared_error: 0.0105 - val_loss: 0.0150 - val_mean_squared_error: 0.0150
# Epoch 242/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.0142 - val_mean_squared_error: 0.0142
# Epoch 243/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0100 - mean_squared_error: 0.0100 - val_loss: 0.0140 - val_mean_squared_error: 0.0140
# Epoch 244/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0104 - mean_squared_error: 0.0104 - val_loss: 0.0146 - val_mean_squared_error: 0.0146
# Epoch 245/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.0141 - val_mean_squared_error: 0.0141
# Epoch 246/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.0150 - val_mean_squared_error: 0.0150
# Epoch 247/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.0127 - val_mean_squared_error: 0.0127
# Epoch 248/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0103 - mean_squared_error: 0.0103 - val_loss: 0.0140 - val_mean_squared_error: 0.0140
# Epoch 249/600
# 3224/3224 [==============================] - 0s 68us/step - loss: 0.0101 - mean_squared_error: 0.0101 - val_loss: 0.0154 - val_mean_squared_error: 0.0154
# Epoch 250/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0098 - mean_squared_error: 0.0098 - val_loss: 0.0148 - val_mean_squared_error: 0.0148
# Epoch 251/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0121 - mean_squared_error: 0.0121 - val_loss: 0.0132 - val_mean_squared_error: 0.0132
# Epoch 252/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0101 - mean_squared_error: 0.0101 - val_loss: 0.0157 - val_mean_squared_error: 0.0157
# Epoch 253/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0104 - mean_squared_error: 0.0104 - val_loss: 0.0127 - val_mean_squared_error: 0.0127
# Epoch 254/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0097 - mean_squared_error: 0.0097 - val_loss: 0.0146 - val_mean_squared_error: 0.0146
# Epoch 255/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0105 - mean_squared_error: 0.0105 - val_loss: 0.0142 - val_mean_squared_error: 0.0142
# Epoch 256/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0099 - mean_squared_error: 0.0099 - val_loss: 0.0151 - val_mean_squared_error: 0.0151
# Epoch 257/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0097 - mean_squared_error: 0.0097 - val_loss: 0.0156 - val_mean_squared_error: 0.0156
# Epoch 258/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0108 - mean_squared_error: 0.0108 - val_loss: 0.0129 - val_mean_squared_error: 0.0129
# Epoch 259/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0100 - mean_squared_error: 0.0100 - val_loss: 0.0144 - val_mean_squared_error: 0.0144
# Epoch 260/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0097 - mean_squared_error: 0.0097 - val_loss: 0.0156 - val_mean_squared_error: 0.0156
# Epoch 261/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0097 - mean_squared_error: 0.0097 - val_loss: 0.0165 - val_mean_squared_error: 0.0165
# Epoch 262/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0095 - mean_squared_error: 0.0095 - val_loss: 0.0165 - val_mean_squared_error: 0.0165
# Epoch 263/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0111 - mean_squared_error: 0.0111 - val_loss: 0.0123 - val_mean_squared_error: 0.0123
# Epoch 264/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.0183 - val_mean_squared_error: 0.0183
# Epoch 265/600
# 3224/3224 [==============================] - 0s 80us/step - loss: 0.0099 - mean_squared_error: 0.0099 - val_loss: 0.0137 - val_mean_squared_error: 0.0137
# Epoch 266/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0096 - mean_squared_error: 0.0096 - val_loss: 0.0153 - val_mean_squared_error: 0.0153
# Epoch 267/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0093 - mean_squared_error: 0.0093 - val_loss: 0.0153 - val_mean_squared_error: 0.0153
# Epoch 268/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0093 - mean_squared_error: 0.0093 - val_loss: 0.0158 - val_mean_squared_error: 0.0158
# Epoch 269/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0097 - mean_squared_error: 0.0097 - val_loss: 0.0173 - val_mean_squared_error: 0.0173
# Epoch 270/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.0156 - val_mean_squared_error: 0.0156
# Epoch 271/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.0170 - val_mean_squared_error: 0.0170
# Epoch 272/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0096 - mean_squared_error: 0.0096 - val_loss: 0.0151 - val_mean_squared_error: 0.0151
# Epoch 273/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0099 - mean_squared_error: 0.0099 - val_loss: 0.0168 - val_mean_squared_error: 0.0168
# Epoch 274/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0092 - mean_squared_error: 0.0092 - val_loss: 0.0162 - val_mean_squared_error: 0.0162
# Epoch 275/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0098 - mean_squared_error: 0.0098 - val_loss: 0.0193 - val_mean_squared_error: 0.0193
# Epoch 276/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0097 - mean_squared_error: 0.0097 - val_loss: 0.0165 - val_mean_squared_error: 0.0165
# Epoch 277/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0095 - mean_squared_error: 0.0095 - val_loss: 0.0162 - val_mean_squared_error: 0.0162
# Epoch 278/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0098 - mean_squared_error: 0.0098 - val_loss: 0.0171 - val_mean_squared_error: 0.0171
# Epoch 279/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.0175 - val_mean_squared_error: 0.0175
# Epoch 280/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.0147 - val_mean_squared_error: 0.0147
# Epoch 281/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.0193 - val_mean_squared_error: 0.0193
# Epoch 282/600
# 3224/3224 [==============================] - 0s 69us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.0164 - val_mean_squared_error: 0.0164
# Epoch 283/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0095 - mean_squared_error: 0.0095 - val_loss: 0.0174 - val_mean_squared_error: 0.0174
# Epoch 284/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.0196 - val_mean_squared_error: 0.0196
# Epoch 285/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0090 - mean_squared_error: 0.0090 - val_loss: 0.0217 - val_mean_squared_error: 0.0217
# Epoch 286/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0092 - mean_squared_error: 0.0092 - val_loss: 0.0150 - val_mean_squared_error: 0.0150
# Epoch 287/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0093 - mean_squared_error: 0.0093 - val_loss: 0.0165 - val_mean_squared_error: 0.0165
# Epoch 288/600
# 3224/3224 [==============================] - 0s 64us/step - loss: 0.0100 - mean_squared_error: 0.0100 - val_loss: 0.0158 - val_mean_squared_error: 0.0158
# Epoch 289/600
# 3224/3224 [==============================] - 0s 55us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.0197 - val_mean_squared_error: 0.0197
# Epoch 290/600
# 3224/3224 [==============================] - 0s 65us/step - loss: 0.0101 - mean_squared_error: 0.0101 - val_loss: 0.0177 - val_mean_squared_error: 0.0177
# Epoch 291/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0099 - mean_squared_error: 0.0099 - val_loss: 0.0218 - val_mean_squared_error: 0.0218
# Epoch 292/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.0180 - val_mean_squared_error: 0.0180
# Epoch 293/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0093 - mean_squared_error: 0.0093 - val_loss: 0.0159 - val_mean_squared_error: 0.0159
# Epoch 294/600
# 3224/3224 [==============================] - 0s 68us/step - loss: 0.0101 - mean_squared_error: 0.0101 - val_loss: 0.0150 - val_mean_squared_error: 0.0150
# Epoch 295/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0092 - mean_squared_error: 0.0092 - val_loss: 0.0187 - val_mean_squared_error: 0.0187
# Epoch 296/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0092 - mean_squared_error: 0.0092 - val_loss: 0.0191 - val_mean_squared_error: 0.0191
# Epoch 297/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0097 - mean_squared_error: 0.0097 - val_loss: 0.0175 - val_mean_squared_error: 0.0175
# Epoch 298/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.0211 - val_mean_squared_error: 0.0211
# Epoch 299/600
# 3224/3224 [==============================] - 0s 59us/step - loss: 0.0089 - mean_squared_error: 0.0089 - val_loss: 0.0255 - val_mean_squared_error: 0.0255
# Epoch 300/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0091 - mean_squared_error: 0.0091 - val_loss: 0.0173 - val_mean_squared_error: 0.0173
# Epoch 301/600
# 3224/3224 [==============================] - 0s 59us/step - loss: 0.0089 - mean_squared_error: 0.0089 - val_loss: 0.0183 - val_mean_squared_error: 0.0183
# Epoch 302/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0088 - mean_squared_error: 0.0088 - val_loss: 0.0194 - val_mean_squared_error: 0.0194
# Epoch 303/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0090 - mean_squared_error: 0.0090 - val_loss: 0.0211 - val_mean_squared_error: 0.0211
# Epoch 304/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.0267 - val_mean_squared_error: 0.0267
# Epoch 305/600
# 3224/3224 [==============================] - 0s 60us/step - loss: 0.0088 - mean_squared_error: 0.0088 - val_loss: 0.0196 - val_mean_squared_error: 0.0196
# Epoch 306/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.0193 - val_mean_squared_error: 0.0193
# Epoch 307/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.0204 - val_mean_squared_error: 0.0204
# Epoch 308/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.0252 - val_mean_squared_error: 0.0252
# Epoch 309/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0099 - mean_squared_error: 0.0099 - val_loss: 0.0255 - val_mean_squared_error: 0.0255
# Epoch 310/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0097 - mean_squared_error: 0.0097 - val_loss: 0.0299 - val_mean_squared_error: 0.0299
# Epoch 311/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0090 - mean_squared_error: 0.0090 - val_loss: 0.0183 - val_mean_squared_error: 0.0183
# Epoch 312/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0085 - mean_squared_error: 0.0085 - val_loss: 0.0172 - val_mean_squared_error: 0.0172
# Epoch 313/600
# 3224/3224 [==============================] - 0s 55us/step - loss: 0.0084 - mean_squared_error: 0.0084 - val_loss: 0.0233 - val_mean_squared_error: 0.0233
# Epoch 314/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0084 - mean_squared_error: 0.0084 - val_loss: 0.0239 - val_mean_squared_error: 0.0239
# Epoch 315/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0109 - mean_squared_error: 0.0109 - val_loss: 0.0232 - val_mean_squared_error: 0.0232
# Epoch 316/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0093 - mean_squared_error: 0.0093 - val_loss: 0.0225 - val_mean_squared_error: 0.0225
# Epoch 317/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0088 - mean_squared_error: 0.0088 - val_loss: 0.0256 - val_mean_squared_error: 0.0256
# Epoch 318/600
# 3224/3224 [==============================] - 0s 55us/step - loss: 0.0089 - mean_squared_error: 0.0089 - val_loss: 0.0158 - val_mean_squared_error: 0.0158
# Epoch 319/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0088 - mean_squared_error: 0.0088 - val_loss: 0.0297 - val_mean_squared_error: 0.0297
# Epoch 320/600
# 3224/3224 [==============================] - 0s 57us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.0181 - val_mean_squared_error: 0.0181
# Epoch 321/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0089 - mean_squared_error: 0.0089 - val_loss: 0.0207 - val_mean_squared_error: 0.0207
# Epoch 322/600
# 3224/3224 [==============================] - 0s 55us/step - loss: 0.0088 - mean_squared_error: 0.0088 - val_loss: 0.0156 - val_mean_squared_error: 0.0156
# Epoch 323/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0089 - mean_squared_error: 0.0089 - val_loss: 0.0234 - val_mean_squared_error: 0.0234
# Epoch 324/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0084 - mean_squared_error: 0.0084 - val_loss: 0.0210 - val_mean_squared_error: 0.0210
# Epoch 325/600
# 3224/3224 [==============================] - 0s 59us/step - loss: 0.0085 - mean_squared_error: 0.0085 - val_loss: 0.0346 - val_mean_squared_error: 0.0346
# Epoch 326/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.0189 - val_mean_squared_error: 0.0189
# Epoch 327/600
# 3224/3224 [==============================] - 0s 59us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.0211 - val_mean_squared_error: 0.0211
# Epoch 328/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.0188 - val_mean_squared_error: 0.0188
# Epoch 329/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.0178 - val_mean_squared_error: 0.0178
# Epoch 330/600
# 3224/3224 [==============================] - 0s 59us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.0208 - val_mean_squared_error: 0.0208
# Epoch 331/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0085 - mean_squared_error: 0.0085 - val_loss: 0.0260 - val_mean_squared_error: 0.0260
# Epoch 332/600
# 3224/3224 [==============================] - 0s 59us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.0265 - val_mean_squared_error: 0.0265
# Epoch 333/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0255 - val_mean_squared_error: 0.0255
# Epoch 334/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0246 - val_mean_squared_error: 0.0246
# Epoch 335/600
# 3224/3224 [==============================] - 0s 79us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0184 - val_mean_squared_error: 0.0184
# Epoch 336/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0086 - mean_squared_error: 0.0086 - val_loss: 0.0206 - val_mean_squared_error: 0.0206
# Epoch 337/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0116 - mean_squared_error: 0.0116 - val_loss: 0.0284 - val_mean_squared_error: 0.0284
# Epoch 338/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.0186 - val_mean_squared_error: 0.0186
# Epoch 339/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0085 - mean_squared_error: 0.0085 - val_loss: 0.0205 - val_mean_squared_error: 0.0205
# Epoch 340/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0190 - val_mean_squared_error: 0.0190
# Epoch 341/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0271 - val_mean_squared_error: 0.0271
# Epoch 342/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.0243 - val_mean_squared_error: 0.0243
# Epoch 343/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0084 - mean_squared_error: 0.0084 - val_loss: 0.0324 - val_mean_squared_error: 0.0324
# Epoch 344/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0312 - val_mean_squared_error: 0.0312
# Epoch 345/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0290 - val_mean_squared_error: 0.0290
# Epoch 346/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0079 - mean_squared_error: 0.0079 - val_loss: 0.0220 - val_mean_squared_error: 0.0220
# Epoch 347/600
# 3224/3224 [==============================] - 0s 55us/step - loss: 0.0084 - mean_squared_error: 0.0084 - val_loss: 0.0242 - val_mean_squared_error: 0.0242
# Epoch 348/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0079 - mean_squared_error: 0.0079 - val_loss: 0.0337 - val_mean_squared_error: 0.0337
# Epoch 349/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0249 - val_mean_squared_error: 0.0249
# Epoch 350/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0260 - val_mean_squared_error: 0.0260
# Epoch 351/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.0207 - val_mean_squared_error: 0.0207
# Epoch 352/600
# 3224/3224 [==============================] - 0s 60us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0282 - val_mean_squared_error: 0.0282
# Epoch 353/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0084 - mean_squared_error: 0.0084 - val_loss: 0.0364 - val_mean_squared_error: 0.0364
# Epoch 354/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.0263 - val_mean_squared_error: 0.0263
# Epoch 355/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0293 - val_mean_squared_error: 0.0293
# Epoch 356/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0191 - val_mean_squared_error: 0.0191
# Epoch 357/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0092 - mean_squared_error: 0.0092 - val_loss: 0.0190 - val_mean_squared_error: 0.0190
# Epoch 358/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.0221 - val_mean_squared_error: 0.0221
# Epoch 359/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.0333 - val_mean_squared_error: 0.0333
# Epoch 360/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0079 - mean_squared_error: 0.0079 - val_loss: 0.0204 - val_mean_squared_error: 0.0204
# Epoch 361/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0084 - mean_squared_error: 0.0084 - val_loss: 0.0389 - val_mean_squared_error: 0.0389
# Epoch 362/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.0389 - val_mean_squared_error: 0.0389
# Epoch 363/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.0464 - val_mean_squared_error: 0.0464
# Epoch 364/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0084 - mean_squared_error: 0.0084 - val_loss: 0.0316 - val_mean_squared_error: 0.0316
# Epoch 365/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.0215 - val_mean_squared_error: 0.0215
# Epoch 366/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0079 - mean_squared_error: 0.0079 - val_loss: 0.0295 - val_mean_squared_error: 0.0295
# Epoch 367/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0078 - mean_squared_error: 0.0078 - val_loss: 0.0433 - val_mean_squared_error: 0.0433
# Epoch 368/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.0314 - val_mean_squared_error: 0.0314
# Epoch 369/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0259 - val_mean_squared_error: 0.0259
# Epoch 370/600
# 3224/3224 [==============================] - 0s 72us/step - loss: 0.0079 - mean_squared_error: 0.0079 - val_loss: 0.0437 - val_mean_squared_error: 0.0437
# Epoch 371/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.0187 - val_mean_squared_error: 0.0187
# Epoch 372/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.0293 - val_mean_squared_error: 0.0293
# Epoch 373/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0085 - mean_squared_error: 0.0085 - val_loss: 0.0383 - val_mean_squared_error: 0.0383
# Epoch 374/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.0218 - val_mean_squared_error: 0.0218
# Epoch 375/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0075 - mean_squared_error: 0.0075 - val_loss: 0.0366 - val_mean_squared_error: 0.0366
# Epoch 376/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0078 - mean_squared_error: 0.0078 - val_loss: 0.0269 - val_mean_squared_error: 0.0269
# Epoch 377/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0333 - val_mean_squared_error: 0.0333
# Epoch 378/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0329 - val_mean_squared_error: 0.0329
# Epoch 379/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.0367 - val_mean_squared_error: 0.0367
# Epoch 380/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0079 - mean_squared_error: 0.0079 - val_loss: 0.0301 - val_mean_squared_error: 0.0301
# Epoch 381/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.0278 - val_mean_squared_error: 0.0278
# Epoch 382/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0307 - val_mean_squared_error: 0.0307
# Epoch 383/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0319 - val_mean_squared_error: 0.0319
# Epoch 384/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.0423 - val_mean_squared_error: 0.0423
# Epoch 385/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0396 - val_mean_squared_error: 0.0396
# Epoch 386/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.0264 - val_mean_squared_error: 0.0264
# Epoch 387/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0285 - val_mean_squared_error: 0.0285
# Epoch 388/600
# 3224/3224 [==============================] - 0s 78us/step - loss: 0.0078 - mean_squared_error: 0.0078 - val_loss: 0.0355 - val_mean_squared_error: 0.0355
# Epoch 389/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.0290 - val_mean_squared_error: 0.0290
# Epoch 390/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0308 - val_mean_squared_error: 0.0308
# Epoch 391/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0083 - mean_squared_error: 0.0083 - val_loss: 0.0355 - val_mean_squared_error: 0.0355
# Epoch 392/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0078 - mean_squared_error: 0.0078 - val_loss: 0.0273 - val_mean_squared_error: 0.0273
# Epoch 393/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0078 - mean_squared_error: 0.0078 - val_loss: 0.0369 - val_mean_squared_error: 0.0369
# Epoch 394/600
# 3224/3224 [==============================] - 0s 57us/step - loss: 0.0075 - mean_squared_error: 0.0075 - val_loss: 0.0327 - val_mean_squared_error: 0.0327
# Epoch 395/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0075 - mean_squared_error: 0.0075 - val_loss: 0.0414 - val_mean_squared_error: 0.0414
# Epoch 396/600
# 3224/3224 [==============================] - 0s 56us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0308 - val_mean_squared_error: 0.0308
# Epoch 397/600
# 3224/3224 [==============================] - 0s 58us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0241 - val_mean_squared_error: 0.0241
# Epoch 398/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0289 - val_mean_squared_error: 0.0289
# Epoch 399/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0075 - mean_squared_error: 0.0075 - val_loss: 0.0337 - val_mean_squared_error: 0.0337
# Epoch 400/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0356 - val_mean_squared_error: 0.0356
# Epoch 401/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0249 - val_mean_squared_error: 0.0249
# Epoch 402/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0498 - val_mean_squared_error: 0.0498
# Epoch 403/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0081 - mean_squared_error: 0.0081 - val_loss: 0.0222 - val_mean_squared_error: 0.0222
# Epoch 404/600
# 3224/3224 [==============================] - 0s 61us/step - loss: 0.0094 - mean_squared_error: 0.0094 - val_loss: 0.0340 - val_mean_squared_error: 0.0340
# Epoch 405/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0075 - mean_squared_error: 0.0075 - val_loss: 0.0381 - val_mean_squared_error: 0.0381
# Epoch 406/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0082 - mean_squared_error: 0.0082 - val_loss: 0.0324 - val_mean_squared_error: 0.0324
# Epoch 407/600
# 3224/3224 [==============================] - 0s 84us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0399 - val_mean_squared_error: 0.0399
# Epoch 408/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0078 - mean_squared_error: 0.0078 - val_loss: 0.0368 - val_mean_squared_error: 0.0368
# Epoch 409/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0434 - val_mean_squared_error: 0.0434
# Epoch 410/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.0353 - val_mean_squared_error: 0.0353
# Epoch 411/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0078 - mean_squared_error: 0.0078 - val_loss: 0.0383 - val_mean_squared_error: 0.0383
# Epoch 412/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0071 - mean_squared_error: 0.0071 - val_loss: 0.0593 - val_mean_squared_error: 0.0593
# Epoch 413/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0286 - val_mean_squared_error: 0.0286
# Epoch 414/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0450 - val_mean_squared_error: 0.0450
# Epoch 415/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0256 - val_mean_squared_error: 0.0256
# Epoch 416/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0075 - mean_squared_error: 0.0075 - val_loss: 0.0310 - val_mean_squared_error: 0.0310
# Epoch 417/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0468 - val_mean_squared_error: 0.0468
# Epoch 418/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.0403 - val_mean_squared_error: 0.0403
# Epoch 419/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.0314 - val_mean_squared_error: 0.0314
# Epoch 420/600
# 3224/3224 [==============================] - 0s 55us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0418 - val_mean_squared_error: 0.0418
# Epoch 421/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0078 - mean_squared_error: 0.0078 - val_loss: 0.0535 - val_mean_squared_error: 0.0535
# Epoch 422/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0416 - val_mean_squared_error: 0.0416
# Epoch 423/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0075 - mean_squared_error: 0.0075 - val_loss: 0.0344 - val_mean_squared_error: 0.0344
# Epoch 424/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0071 - mean_squared_error: 0.0071 - val_loss: 0.0263 - val_mean_squared_error: 0.0263
# Epoch 425/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.0423 - val_mean_squared_error: 0.0423
# Epoch 426/600
# 3224/3224 [==============================] - 0s 59us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0329 - val_mean_squared_error: 0.0329
# Epoch 427/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0255 - val_mean_squared_error: 0.0255
# Epoch 428/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0216 - val_mean_squared_error: 0.0216
# Epoch 429/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0517 - val_mean_squared_error: 0.0517
# Epoch 430/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0075 - mean_squared_error: 0.0075 - val_loss: 0.0362 - val_mean_squared_error: 0.0362
# Epoch 431/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0087 - mean_squared_error: 0.0087 - val_loss: 0.0384 - val_mean_squared_error: 0.0384
# Epoch 432/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0334 - val_mean_squared_error: 0.0334
# Epoch 433/600
# 3224/3224 [==============================] - 0s 56us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0388 - val_mean_squared_error: 0.0388
# Epoch 434/600
# 3224/3224 [==============================] - 0s 55us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0447 - val_mean_squared_error: 0.0447
# Epoch 435/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0374 - val_mean_squared_error: 0.0374
# Epoch 436/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0563 - val_mean_squared_error: 0.0563
# Epoch 437/600
# 3224/3224 [==============================] - 0s 66us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0359 - val_mean_squared_error: 0.0359
# Epoch 438/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0474 - val_mean_squared_error: 0.0474
# Epoch 439/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0568 - val_mean_squared_error: 0.0568
# Epoch 440/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.0335 - val_mean_squared_error: 0.0335
# Epoch 441/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0071 - mean_squared_error: 0.0071 - val_loss: 0.0308 - val_mean_squared_error: 0.0308
# Epoch 442/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0323 - val_mean_squared_error: 0.0323
# Epoch 443/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0317 - val_mean_squared_error: 0.0317
# Epoch 444/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.0427 - val_mean_squared_error: 0.0427
# Epoch 445/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0398 - val_mean_squared_error: 0.0398
# Epoch 446/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.0510 - val_mean_squared_error: 0.0510
# Epoch 447/600
# 3224/3224 [==============================] - 0s 63us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0304 - val_mean_squared_error: 0.0304
# Epoch 448/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0071 - mean_squared_error: 0.0071 - val_loss: 0.0353 - val_mean_squared_error: 0.0353
# Epoch 449/600
# 3224/3224 [==============================] - 0s 59us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0450 - val_mean_squared_error: 0.0450
# Epoch 450/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0519 - val_mean_squared_error: 0.0519
# Epoch 451/600
# 3224/3224 [==============================] - 0s 58us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0290 - val_mean_squared_error: 0.0290
# Epoch 452/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0696 - val_mean_squared_error: 0.0696
# Epoch 453/600
# 3224/3224 [==============================] - 0s 57us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.0343 - val_mean_squared_error: 0.0343
# Epoch 454/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.0409 - val_mean_squared_error: 0.0409
# Epoch 455/600
# 3224/3224 [==============================] - 0s 69us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0555 - val_mean_squared_error: 0.0555
# Epoch 456/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.0438 - val_mean_squared_error: 0.0438
# Epoch 457/600
# 3224/3224 [==============================] - 0s 57us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0431 - val_mean_squared_error: 0.0431
# Epoch 458/600
# 3224/3224 [==============================] - 0s 53us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0602 - val_mean_squared_error: 0.0602
# Epoch 459/600
# 3224/3224 [==============================] - 0s 61us/step - loss: 0.0078 - mean_squared_error: 0.0078 - val_loss: 0.0231 - val_mean_squared_error: 0.0231
# Epoch 460/600
# 3224/3224 [==============================] - 0s 88us/step - loss: 0.0071 - mean_squared_error: 0.0071 - val_loss: 0.0346 - val_mean_squared_error: 0.0346
# Epoch 461/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.0329 - val_mean_squared_error: 0.0329
# Epoch 462/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.0409 - val_mean_squared_error: 0.0409
# Epoch 463/600
# 3224/3224 [==============================] - 0s 61us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0303 - val_mean_squared_error: 0.0303
# Epoch 464/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0397 - val_mean_squared_error: 0.0397
# Epoch 465/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0476 - val_mean_squared_error: 0.0476
# Epoch 466/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0523 - val_mean_squared_error: 0.0523
# Epoch 467/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0071 - mean_squared_error: 0.0071 - val_loss: 0.0421 - val_mean_squared_error: 0.0421
# Epoch 468/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0077 - mean_squared_error: 0.0077 - val_loss: 0.0335 - val_mean_squared_error: 0.0335
# Epoch 469/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0067 - mean_squared_error: 0.0067 - val_loss: 0.0601 - val_mean_squared_error: 0.0601
# Epoch 470/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0430 - val_mean_squared_error: 0.0430
# Epoch 471/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0489 - val_mean_squared_error: 0.0489
# Epoch 472/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0298 - val_mean_squared_error: 0.0298
# Epoch 473/600
# 3224/3224 [==============================] - 0s 56us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0505 - val_mean_squared_error: 0.0505
# Epoch 474/600
# 3224/3224 [==============================] - 0s 50us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0318 - val_mean_squared_error: 0.0318
# Epoch 475/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0067 - mean_squared_error: 0.0067 - val_loss: 0.0299 - val_mean_squared_error: 0.0299
# Epoch 476/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0357 - val_mean_squared_error: 0.0357
# Epoch 477/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0495 - val_mean_squared_error: 0.0495
# Epoch 478/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.0570 - val_mean_squared_error: 0.0570
# Epoch 479/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0311 - val_mean_squared_error: 0.0311
# Epoch 480/600
# 3224/3224 [==============================] - 0s 64us/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.0496 - val_mean_squared_error: 0.0496
# Epoch 481/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0457 - val_mean_squared_error: 0.0457
# Epoch 482/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0379 - val_mean_squared_error: 0.0379
# Epoch 483/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0076 - mean_squared_error: 0.0076 - val_loss: 0.0321 - val_mean_squared_error: 0.0321
# Epoch 484/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0071 - mean_squared_error: 0.0071 - val_loss: 0.0458 - val_mean_squared_error: 0.0458
# Epoch 485/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0088 - mean_squared_error: 0.0088 - val_loss: 0.0377 - val_mean_squared_error: 0.0377
# Epoch 486/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0463 - val_mean_squared_error: 0.0463
# Epoch 487/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0401 - val_mean_squared_error: 0.0401
# Epoch 488/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0067 - mean_squared_error: 0.0067 - val_loss: 0.0303 - val_mean_squared_error: 0.0303
# Epoch 489/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0080 - mean_squared_error: 0.0080 - val_loss: 0.0231 - val_mean_squared_error: 0.0231
# Epoch 490/600
# 3224/3224 [==============================] - 0s 63us/step - loss: 0.0067 - mean_squared_error: 0.0067 - val_loss: 0.0532 - val_mean_squared_error: 0.0532
# Epoch 491/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0414 - val_mean_squared_error: 0.0414
# Epoch 492/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0313 - val_mean_squared_error: 0.0313
# Epoch 493/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0502 - val_mean_squared_error: 0.0502
# Epoch 494/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0466 - val_mean_squared_error: 0.0466
# Epoch 495/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0431 - val_mean_squared_error: 0.0431
# Epoch 496/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0382 - val_mean_squared_error: 0.0382
# Epoch 497/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0407 - val_mean_squared_error: 0.0407
# Epoch 498/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0071 - mean_squared_error: 0.0071 - val_loss: 0.0330 - val_mean_squared_error: 0.0330
# Epoch 499/600
# 3224/3224 [==============================] - 0s 54us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0410 - val_mean_squared_error: 0.0410
# Epoch 500/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0396 - val_mean_squared_error: 0.0396
# Epoch 501/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0459 - val_mean_squared_error: 0.0459
# Epoch 502/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0346 - val_mean_squared_error: 0.0346
# Epoch 503/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0067 - mean_squared_error: 0.0067 - val_loss: 0.0311 - val_mean_squared_error: 0.0311
# Epoch 504/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.0446 - val_mean_squared_error: 0.0446
# Epoch 505/600
# 3224/3224 [==============================] - 0s 66us/step - loss: 0.0067 - mean_squared_error: 0.0067 - val_loss: 0.0284 - val_mean_squared_error: 0.0284
# Epoch 506/600
# 3224/3224 [==============================] - 0s 57us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0347 - val_mean_squared_error: 0.0347
# Epoch 507/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0306 - val_mean_squared_error: 0.0306
# Epoch 508/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0074 - mean_squared_error: 0.0074 - val_loss: 0.0374 - val_mean_squared_error: 0.0374
# Epoch 509/600
# 3224/3224 [==============================] - 0s 60us/step - loss: 0.0089 - mean_squared_error: 0.0089 - val_loss: 0.0299 - val_mean_squared_error: 0.0299
# Epoch 510/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0407 - val_mean_squared_error: 0.0407
# Epoch 511/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0400 - val_mean_squared_error: 0.0400
# Epoch 512/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0502 - val_mean_squared_error: 0.0502
# Epoch 513/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0324 - val_mean_squared_error: 0.0324
# Epoch 514/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0516 - val_mean_squared_error: 0.0516
# Epoch 515/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0418 - val_mean_squared_error: 0.0418
# Epoch 516/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0071 - mean_squared_error: 0.0071 - val_loss: 0.0436 - val_mean_squared_error: 0.0436
# Epoch 517/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0408 - val_mean_squared_error: 0.0408
# Epoch 518/600
# 3224/3224 [==============================] - 0s 52us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0260 - val_mean_squared_error: 0.0260
# Epoch 519/600
# 3224/3224 [==============================] - 0s 56us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0385 - val_mean_squared_error: 0.0385
# Epoch 520/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0391 - val_mean_squared_error: 0.0391
# Epoch 521/600
# 3224/3224 [==============================] - 0s 57us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0361 - val_mean_squared_error: 0.0361
# Epoch 522/600
# 3224/3224 [==============================] - 0s 65us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0507 - val_mean_squared_error: 0.0507
# Epoch 523/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0415 - val_mean_squared_error: 0.0415
# Epoch 524/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0393 - val_mean_squared_error: 0.0393
# Epoch 525/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0302 - val_mean_squared_error: 0.0302
# Epoch 526/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0551 - val_mean_squared_error: 0.0551
# Epoch 527/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.0320 - val_mean_squared_error: 0.0320
# Epoch 528/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0326 - val_mean_squared_error: 0.0326
# Epoch 529/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0067 - mean_squared_error: 0.0067 - val_loss: 0.0555 - val_mean_squared_error: 0.0555
# Epoch 530/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0468 - val_mean_squared_error: 0.0468
# Epoch 531/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0372 - val_mean_squared_error: 0.0372
# Epoch 532/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0367 - val_mean_squared_error: 0.0367
# Epoch 533/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.0423 - val_mean_squared_error: 0.0423
# Epoch 534/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0463 - val_mean_squared_error: 0.0463
# Epoch 535/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0073 - mean_squared_error: 0.0073 - val_loss: 0.0238 - val_mean_squared_error: 0.0238
# Epoch 536/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.0604 - val_mean_squared_error: 0.0604
# Epoch 537/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0385 - val_mean_squared_error: 0.0385
# Epoch 538/600
# 3224/3224 [==============================] - 0s 72us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0429 - val_mean_squared_error: 0.0429
# Epoch 539/600
# 3224/3224 [==============================] - 0s 59us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0315 - val_mean_squared_error: 0.0315
# Epoch 540/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0491 - val_mean_squared_error: 0.0491
# Epoch 541/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0059 - mean_squared_error: 0.0059 - val_loss: 0.0424 - val_mean_squared_error: 0.0424
# Epoch 542/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0310 - val_mean_squared_error: 0.0310
# Epoch 543/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0520 - val_mean_squared_error: 0.0520
# Epoch 544/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0318 - val_mean_squared_error: 0.0318
# Epoch 545/600
# 3224/3224 [==============================] - 0s 49us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0492 - val_mean_squared_error: 0.0492
# Epoch 546/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0506 - val_mean_squared_error: 0.0506
# Epoch 547/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0292 - val_mean_squared_error: 0.0292
# Epoch 548/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0070 - mean_squared_error: 0.0070 - val_loss: 0.0278 - val_mean_squared_error: 0.0278
# Epoch 549/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0470 - val_mean_squared_error: 0.0470
# Epoch 550/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0060 - mean_squared_error: 0.0060 - val_loss: 0.0406 - val_mean_squared_error: 0.0406
# Epoch 551/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0599 - val_mean_squared_error: 0.0599
# Epoch 552/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0540 - val_mean_squared_error: 0.0540
# Epoch 553/600
# 3224/3224 [==============================] - 0s 67us/step - loss: 0.0061 - mean_squared_error: 0.0061 - val_loss: 0.0410 - val_mean_squared_error: 0.0410
# Epoch 554/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0288 - val_mean_squared_error: 0.0288
# Epoch 555/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0465 - val_mean_squared_error: 0.0465
# Epoch 556/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0432 - val_mean_squared_error: 0.0432
# Epoch 557/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0406 - val_mean_squared_error: 0.0406
# Epoch 558/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0398 - val_mean_squared_error: 0.0398
# Epoch 559/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0417 - val_mean_squared_error: 0.0417
# Epoch 560/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0406 - val_mean_squared_error: 0.0406
# Epoch 561/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0060 - mean_squared_error: 0.0060 - val_loss: 0.0392 - val_mean_squared_error: 0.0392
# Epoch 562/600
# 3224/3224 [==============================] - 0s 51us/step - loss: 0.0061 - mean_squared_error: 0.0061 - val_loss: 0.0238 - val_mean_squared_error: 0.0238
# Epoch 563/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0406 - val_mean_squared_error: 0.0406
# Epoch 564/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0264 - val_mean_squared_error: 0.0264
# Epoch 565/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0064 - mean_squared_error: 0.0064 - val_loss: 0.0347 - val_mean_squared_error: 0.0347
# Epoch 566/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0068 - mean_squared_error: 0.0068 - val_loss: 0.0354 - val_mean_squared_error: 0.0354
# Epoch 567/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0305 - val_mean_squared_error: 0.0305
# Epoch 568/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0061 - mean_squared_error: 0.0061 - val_loss: 0.0428 - val_mean_squared_error: 0.0428
# Epoch 569/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0414 - val_mean_squared_error: 0.0414
# Epoch 570/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0060 - mean_squared_error: 0.0060 - val_loss: 0.0354 - val_mean_squared_error: 0.0354
# Epoch 571/600
# 3224/3224 [==============================] - 0s 61us/step - loss: 0.0072 - mean_squared_error: 0.0072 - val_loss: 0.0254 - val_mean_squared_error: 0.0254
# Epoch 572/600
# 3224/3224 [==============================] - 0s 42us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0475 - val_mean_squared_error: 0.0475
# Epoch 573/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0058 - mean_squared_error: 0.0058 - val_loss: 0.0385 - val_mean_squared_error: 0.0385
# Epoch 574/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0289 - val_mean_squared_error: 0.0289
# Epoch 575/600
# 3224/3224 [==============================] - 0s 47us/step - loss: 0.0061 - mean_squared_error: 0.0061 - val_loss: 0.0337 - val_mean_squared_error: 0.0337
# Epoch 576/600
# 3224/3224 [==============================] - 0s 42us/step - loss: 0.0075 - mean_squared_error: 0.0075 - val_loss: 0.0394 - val_mean_squared_error: 0.0394
# Epoch 577/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0061 - mean_squared_error: 0.0061 - val_loss: 0.0429 - val_mean_squared_error: 0.0429
# Epoch 578/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0060 - mean_squared_error: 0.0060 - val_loss: 0.0443 - val_mean_squared_error: 0.0443
# Epoch 579/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.0375 - val_mean_squared_error: 0.0375
# Epoch 580/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0059 - mean_squared_error: 0.0059 - val_loss: 0.0333 - val_mean_squared_error: 0.0333
# Epoch 581/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0367 - val_mean_squared_error: 0.0367
# Epoch 582/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0387 - val_mean_squared_error: 0.0387
# Epoch 583/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0058 - mean_squared_error: 0.0058 - val_loss: 0.0478 - val_mean_squared_error: 0.0478
# Epoch 584/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0058 - mean_squared_error: 0.0058 - val_loss: 0.0368 - val_mean_squared_error: 0.0368
# Epoch 585/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0349 - val_mean_squared_error: 0.0349
# Epoch 586/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0069 - mean_squared_error: 0.0069 - val_loss: 0.0297 - val_mean_squared_error: 0.0297
# Epoch 587/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0059 - mean_squared_error: 0.0059 - val_loss: 0.0346 - val_mean_squared_error: 0.0346
# Epoch 588/600
# 3224/3224 [==============================] - 0s 58us/step - loss: 0.0058 - mean_squared_error: 0.0058 - val_loss: 0.0380 - val_mean_squared_error: 0.0380
# Epoch 589/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0058 - mean_squared_error: 0.0058 - val_loss: 0.0398 - val_mean_squared_error: 0.0398
# Epoch 590/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0408 - val_mean_squared_error: 0.0408
# Epoch 591/600
# 3224/3224 [==============================] - 0s 48us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0297 - val_mean_squared_error: 0.0297
# Epoch 592/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0060 - mean_squared_error: 0.0060 - val_loss: 0.0418 - val_mean_squared_error: 0.0418
# Epoch 593/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0065 - mean_squared_error: 0.0065 - val_loss: 0.0344 - val_mean_squared_error: 0.0344
# Epoch 594/600
# 3224/3224 [==============================] - 0s 45us/step - loss: 0.0067 - mean_squared_error: 0.0067 - val_loss: 0.0506 - val_mean_squared_error: 0.0506
# Epoch 595/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0062 - mean_squared_error: 0.0062 - val_loss: 0.0374 - val_mean_squared_error: 0.0374
# Epoch 596/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0058 - mean_squared_error: 0.0058 - val_loss: 0.0479 - val_mean_squared_error: 0.0479
# Epoch 597/600
# 3224/3224 [==============================] - 0s 44us/step - loss: 0.0061 - mean_squared_error: 0.0061 - val_loss: 0.0336 - val_mean_squared_error: 0.0336
# Epoch 598/600
# 3224/3224 [==============================] - 0s 43us/step - loss: 0.0066 - mean_squared_error: 0.0066 - val_loss: 0.0490 - val_mean_squared_error: 0.0490
# Epoch 599/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0063 - mean_squared_error: 0.0063 - val_loss: 0.0378 - val_mean_squared_error: 0.0378
# Epoch 600/600
# 3224/3224 [==============================] - 0s 46us/step - loss: 0.0057 - mean_squared_error: 0.0057 - val_loss: 0.0420 - val_mean_squared_error: 0.0420
# 
# 
# 
# 
# 
# 
# 







