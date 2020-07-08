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

seed_num=42
try:
    from tensorflow import set_random_seed # in josephus' computer
    set_random_seed(seed_num)
except:
    from tensorflow.random import set_seed # in yogenders' computer
    set_seed(seed_num)

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
    #df_hp.replace(0,np.nan).dropna(axis=1,how="all")
    #df_tem = df.drop(columns = 'Hours')
    return df_hp

def normalize(X):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    scaled_data = scaler.transform(X)
    return scaled_data, scaler

df, orig_df = load_data()
df_nrm, scaler = normalize(df)
df_nrm = pd.DataFrame(df_nrm)
df_hp = load_dataHP()
dfhp_nrm, scaler = normalize(df_hp)
dfhp_nrm = pd.DataFrame(dfhp_nrm)
dfhp_nrm.columns = df_hp.columns

all(df_hp.loc[:, 'Hours'] == orig_df.loc[:, 'Hours']) # True
# thus the two dfs can be merged

##############################################
# for LSTM taking k last time-point m values create an ANN
##############################################

k = 3
n_features = 20

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


# to plot make data frame out of dict history.history and use .plot() method
pd.DataFrame(history.history).plot()
plt.show()


###########################################
# plot prediction
###########################################
model = load_model(model_fpath) # this loads the best saved model!
yhat=model.predict(X_test.reshape(X_test.shape[0], k, n_features))


def unscale(y_values, scaler):
   return scaler.inverse_transform(y_values)

y_pred_unscaled, y_test_unscaled = unscale(yhat, scaler), unscale(y_test, scaler)

def plot(arr_y_pred, arr_y_test, orig_df):
   xdata = orig_df.iloc[4609:, 0]
   df_y_pred = pd.DataFrame(arr_y_pred)
   df_y_test = pd.DataFrame(arr_y_test) # arry_y_pred you took here!
   legends_test =['OrgT' + str(i) for i in range (1, 21) ]
   legends_pred =['PrT' + str(i) for i in range (1, 21) ]
   for i, j in zip(df_y_pred, df_y_test):
       plt.plot(xdata, df_y_pred.iloc[:, i], label = legends_pred[i])
       plt.plot(xdata, df_y_test.iloc[:, j], label = legends_test[i])

   plt.legend()
   plt.show()
   return

plt.clf()
plot(y_pred_unscaled, y_test_unscaled, orig_df)

print(f"Best val_loss is: {min(history.history['val_loss'])}")





















