
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

def load_data(fpath="./dlOne", col_names = ['Hours'] + ["T" + str(i) for i in range(1, 21)]):
    df = pd.read_csv(fpath, 
                     header = 1, 
                     encoding = "ISO-8859-1", 
                     sep='\t', 
                     skiprows=0).dropna(axis=1).astype(float)
    df.columns = col_names
    return df.drop(columns = 'Hours'), df

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



def normalize(X):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    scaled_data = scaler.transform(X)
    return scaled_data, scaler

"""
Let's say data frame has n_rows and n_cols = n_values
n_rows, n_cols = df.shape

"""

def flatten_row_wise(df):
    """Take row by row and attach to one flat single row."""
    return np.ndarray.flatten(np.array(df))

def prepare_df(df, k):
    n_rows, n_cols = df.shape
    new_rows = np.array([flatten_row_wise(df.iloc[(i-k):i]) for i in range(k, n_rows)])
    new_ys = np.array([row for row in df.iloc[(k):, :].itertuples(index=False)])
    return new_rows, new_ys

"""
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
X, y = prepare_df(df)
looks correct!

last row is last y
X doesn't contain last row and begins from first row!
"""


def create_ann_30_model(n_input_features, n_output_features):
    model = Sequential()
    model.add(Dense(30, input_shape=(n_input_features,), activation='relu'))
    model.add(Dense(n_output_features, activation='linear'))
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_squared_error"])
    return model

def create_lstm_10_20_model(time_steps, n_features):
    model = Sequential()
    model.add(LSTM(10, input_shape = (time_steps, n_features)))
    model.add(Dense(20, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

def get_callbacks(moedl_fpath):
    return [ ModelCheckpoint(filepath=model_fpath,
                                   monitor="val_loss",
                                   save_best_only=True,
                                   mode="min")]

def unscale(y_pred, y_test, scaler):
    y_pred_orig = scaler.inverse_transform(y_pred)
    y_test_orig = scaler.inverse_transform(y_test)
    return y_pred_orig, y_test_orig

def unscale_1(y_values, scaler):
   # it is more complex, because y_values here miss the Tamb column
   # we add artificial Tamb column and remove it again.
   y_values_with_Tamb_fake = np.array([np.concatenate([row[:-2], np.array([1.0]), row[-2:]]) for row in y_values])
   res = scaler.inverse_transform(y_values_with_Tamb_fake)
   return np.array([np.concatenate([row[:-3], row[-2:]]) for row in res])

def unscale_2(y_values, scaler):
   return scaler.inverse_transform(y_values)

def train_test_split_indexes(X, y, test_size=0.2, random_state=42, shuffle=False):
    X_train_indexes, X_test_indexes, y_train_indexes, y_test_indexes = train_test_split(pd.DataFrame(list(range(X.shape[0]))),
                                                                                        pd.DataFrame(list(range(y.shape[0]))),
                                                                                        test_size=test_size,
                                                                                        random_state=random_state,
                                                                                        shuffle=shuffle)
    train_indexes, test_indexes = [x for x in X_train_indexes.iloc[:, 0]]  , [x for x in X_test_indexes.iloc[:, 0]]
    return sorted(train_indexes), sorted(test_indexes)

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

# def plot(arr_y_pred, arr_y_test, orig_df):
#    xdata = orig_df.iloc[4609:, 0]
#    df_y_pred = pd.DataFrame(arr_y_pred)
#    df_y_test = pd.DataFrame(arr_y_test) # arry_y_pred you took here!
#    legends_test =['OrgT' + str(i) for i in range (1, 21) ]
#    legends_pred =['PrT' + str(i) for i in range (1, 21) ]
#    for i, j in zip(df_y_pred, df_y_test):
#        plt.plot(xdata, df_y_pred.iloc[:, i], label = legends_pred[i])
#        plt.plot(xdata, df_y_test.iloc[:, j], label = legends_test[i])
# 
#    plt.legend()
#    plt.show()
#    return


def minimal_val_loss(history):
    min_val_loss=min(history.history['val_loss'])
    print(f"Best val_loss is: {min_val_loss}")
    return min_val_loss

def plot_losses(history):
    pd.DataFrame(history.history).plot()
    plt.show()

def prediction_vs_truth_plot(model, X_test, y_test, scaler, unscale=unscale, out_fpath=None):
    yhat=model.predict(X_test)
    y_pred_unscaled, y_test_unscaled = unscale(yhat, y_test, scaler)
    plt.scatter(y_test_unscaled, y_pred_unscaled)
    if out_fpath is not None:
        plt.savefig(out_fpath)
    plt.show()

