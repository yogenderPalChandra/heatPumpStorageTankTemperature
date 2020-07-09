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
from numpy.random import seed
seed(123)
#from tensorflow import set_random_seed
#set_random_seed(42)
import tensorflow
tensorflow.random.set_seed(42)

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
    col_names = ['Hours', 'Tamb', 'ThpOut', 'mLoadFlowRate', 'KJ/hr' , 'cop']
    df_hp = pd.read_csv(path, 
                     header = 1, 
                     encoding = "ISO-8859-1", 
                     sep='\t', 
                     skiprows=0).dropna(axis=1).astype(float)
    df_hp = df_hp.loc[:, df_hp.any()]
    df_hp.columns = col_names
    df_hp = df_hp.drop(columns = 'Hours')
    return df_hp

#df1 = pd.merge(df_hp[['Tamb']], how="left")

##df1 = pd.concat([df_hp[['Tamb']], df], axis =1)
##
##df2 = pd.concat([df_hp[['KJ/hr', 'cop']], df], axis = 1)
##df3 = pd.concat([df_hp[['Tamb','KJ/hr', 'cop']], df], axis =1)
    
def normalize(X):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    scaled_data = scaler.transform(X)
    return scaled_data, scaler



df, orig_df = load_data()
df_hp = load_dataHP()
##df_nrm, scaler = normalize(df)
##df_nrm = pd.DataFrame(df_nrm)
##dfhp_nrm, scaler = normalize(df_hp)
##dfhp_nrm = pd.DataFrame(dfhp_nrm)
##df3, df3scaler = normalize(df3)


#df3 = pd.DataFrame(df3)

df1 = pd.concat([df_hp.iloc[:, 0], df], axis =1)

def normalize(X):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    scaled_data = scaler.transform(X)
    return scaled_data, scaler

df1_nrm, scaler_df1 = normalize(df1)


df2 = pd.concat([df_hp.iloc[:, 1:3], df], axis =1)

df2_nrm, scaler_df2 = normalize(df2)

##############################################
# for LSTM taking k last time-point m values create an ANN
##############################################

k = 3
n_features = 21

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
    new_ys = np.array([row for row in df.iloc[(k):, :].iterrows()])
    # idxs = [x[0] for x in new_ys]
    # new_ys = [x[1] for x in new_ys]
    return new_rows, new_ys

X, y = prepare_df(df_nrm)
idxs = [x[0] for x in y]
y = np.array([np.array(x[1]) for x in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

Xdf1, ydf1 = prepare_df(df1)

Xdf2, ydf2 = prepare_df(df2)

#takingX values from df1 and taking y values of df2 

y_df2 = np.array([np.array(x[1]) for x in ydf2])

X_train, X_test, y_train, y_test = train_test_split(Xdf1, y_df2, test_size=0.2, random_state=42, shuffle=False)

#y = np.array([np.array(x[1]) for x in ydf1])

'''
def create_model(time_steps, n_features):
    model = Sequential()
    model.add(LSTM(10, input_shape = (time_steps, n_features)))
    model.add(Dense(20, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

model = create_model(k, n_features)
model_fpath="lstm.h5"
callbacks_list = [ ModelCheckpoint(filepath=model_fpath,
                                   monitor="val_loss",
                                   save_best_only=True,
                                   mode="min")]

history = model.fit(X_train.reshape(X_train.shape[0], k, n_features),
                    y_train.reshape(y_train.shape[0], n_features),
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.3,
                    callbacks = callbacks_list,
                    verbose=1)

'''

<<<<<<< HEAD
def create_model(time_steps, n_features):
    model = Sequential()
    model.add(LSTM(10, input_shape = (time_steps, n_features)))
    model.add(Dense(22, activation='linear'))
=======





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




def create_model(time_steps, n_input_features, n_output_features):
    model = Sequential()
    model.add(LSTM(3, input_shape = (time_steps, n_input_features // time_steps)))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_output_features, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_output_features, activation='linear'))
>>>>>>> bc50570bf8d14c5287e5901e3e90e6099797ca5e
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

model = create_model(k, n_features)
model_fpath="lstmHPHP.h5"
callbacks_list = [ ModelCheckpoint(filepath=model_fpath,
                                   monitor="val_loss",
                                   save_best_only=True,
                                   mode="min")]


history = model.fit(X_train.reshape(X_train.shape[0], k, n_features),
                    y_train.reshape(y_train.shape[0], 22),
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.3,
                    callbacks = callbacks_list,
                    shuffle=True,
                    verbose=1)

<<<<<<< HEAD
=======
print(f"Best val_loss is: {min(history.history['val_loss'])}")
# Best val_loss is: 0.011279674120215017




>>>>>>> bc50570bf8d14c5287e5901e3e90e6099797ca5e
# to plot make data frame out of dict history.history and use .plot() method
pd.DataFrame(history.history).plot()
plt.show()


###########################################
# plot prediction
###########################################
model = load_model(model_fpath) # this loads the best saved model!
yhat=model.predict(X_test.reshape(X_test.shape[0], k, n_features))

plt.scatter(y_test, yhat)
plt.show()


def unscale(y_values, scaler):
   return scaler.inverse_transform(y_values)

y_pred_unscaled, y_test_unscaled = unscale(yhat, scaler_df2), unscale(y_test, scaler_df2)

def plot(arr_y_pred, arr_y_test, orig_df):
   xdata = orig_df.iloc[4609:, 0]
   df_y_pred = pd.DataFrame(arr_y_pred)
   df_y_test = pd.DataFrame(arr_y_test) # arry_y_pred you took here!
   legends_test =['OrgT' + str(i) for i in range (0, 22) ]
   legends_pred =['PrT' + str(i) for i in range (0, 22) ]
   for i, j in zip(df_y_pred, df_y_test):
       plt.plot(xdata, df_y_pred.iloc[:, i], label = legends_pred[i])
       plt.plot(xdata, df_y_test.iloc[:, j], label = legends_test[i])

   plt.legend()
   plt.show()
   return

plt.clf()
plot(y_pred_unscaled, y_test_unscaled, orig_df)

print(f"Best val_loss is: {min(history.history['val_loss'])}")





















