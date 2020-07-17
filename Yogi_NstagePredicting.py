import pandas as pd
import numpy as np
import os
from utility import load_data, load_dataHP, normalize, train_test_split_indexes, splitter, load_data_big
from utility import create_Q_t_model, get_callbacks

from utility import plotOutlier
import matplotlib.pyplot as plt





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


import pandas as pd
import numpy as np
import os
from utility import load_data, load_dataHP, normalize, train_test_split_indexes, splitter, flatten_row_wise, prepare_df_Q_t
from utility import create_Q_t_model, get_callbacks


df_t1_20, orig_df = load_data()
df_hp = load_dataHP()
df_new = load_data_big()


'''
df = pd.concat([df_t1_20, df_hp.loc[:, ["Tamb", "KJ/hr"]]], axis=1)
df_nrm, scaler = normalize(df)
df_nrm = pd.DataFrame(df_nrm)
df_nrm.columns = df.columns



df_t1_20, orig_df = load_data()
df_hp = load_dataHP().iloc[5000:11000, :] # here is the problem!! KJ/hr and Tamb are bad!!
# they are all 0.0!!
'''


df = pd.concat([df_t1_20, df_hp.loc[:, ["Tamb", "KJ/hr"]]], axis=1)
df_nrm, scaler = normalize(df)
df_nrm = pd.DataFrame(df_nrm)
df_nrm.columns = df.columns


X = df_nrm.iloc[:, :]
y = pd.DataFrame(df_nrm.iloc[:, :-2])

#X, y = prepare_df_Q_t(X, y, k)

n_output_features = y.shape[1]
n_input_features = X.shape[1]

X = pd.DataFrame(X)
#X.columns = df.columns
y = pd.DataFrame(y)
#y.columns = df.columns



k = 3
epochs = 500
batch_size = 2000
n_features = 22


#train_indexes, test_indexes = train_test_split_indexes(X, y, test_size=0.2, random_state=42, shuffle=True)

#X_train, X_test, y_train, y_test = splitter(X, y, train_indexes, test_indexes)

def flatten_row_wise(df):
    """Take row by row and attach to one flat single row."""
    return np.ndarray.flatten(np.array(df))

def prepare_df(df):
    n_rows, n_cols = df.shape
    new_rows = np.array([flatten_row_wise(df.iloc[(i-k):i]) for i in range(k, n_rows)])
    new_ys = np.array([row for row in df.iloc[(k):, :].itertuples(index=False)])
    # idxs = [x[0] for x in new_ys]
    # new_ys = [x[1] for x in new_ys]
    return new_rows, new_ys


Xdf1, ydf1 = prepare_df(X)

Xdf2, ydf2 = prepare_df(y)


X_train, X_test, y_train, y_test = train_test_split(Xdf1, ydf2, test_size=0.2, random_state=42, shuffle=False)
#X_train, X_test, y_train, y_test = train_test_split(Xdf1, y_df1, test_size=0.2, random_state=42, shuffle=False)


def create_model(time_steps, n_features):
    model = Sequential()
    model.add(LSTM(20, input_shape = (time_steps, n_features)))
    model.add(Dense(20, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

model = create_model(k, n_features)
model_fpath="./20ToutPlus20TTambQhInput.h5"
callbacks_list = [ ModelCheckpoint(filepath=model_fpath,
                                   monitor="val_loss",
                                   save_best_only=True,
                                   mode="min")]


history = model.fit(X_train.reshape(X_train.shape[0], k, n_features),
                    y_train.reshape(y_train.shape[0], 20),
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.3,
                    callbacks = callbacks_list,
                    verbose=1)


model.save(model_fpath)

from keras.models import load_model
model = load_model(model_fpath)

plt.scatter(y_test, yhat)
plt.show()


def prepare_df_X_test(model, arr, k):
    model = load_model('20ToutPlus20TTambQhInput.h5')
    n_rows, n_cols = arr.shape
    
    Tamb_Qh = [[i[-2:] for i in j ] for j in X_test.reshape(X_test.shape[0],k, n_features)]

    pred=[np.array(xi) for xi in arr[1].reshape( k, n_features)]
    pred_20T=[x[:-2] for x in flatten_row_wise(arr[1]).reshape(k,n_features)]
    #print ('length of pred=',len(pred),pred)
    for i in range(k, n_rows):

        pred_shaped = flatten_row_wise(pred[i-k:i]).reshape(1, k, n_features )
        yhat=model.predict(pred_shaped)
        pred_20T.append(np.squeeze(yhat,axis=(0,)))
        yhat_faked = np.concatenate([yhat, np.expand_dims(Tamb_Qh[i-2][-1], axis=0)], axis=1)

        pred.append(np.squeeze(yhat_faked, axis=(0,)))
        print ('looping nex loop', i)
        
    return pred, np.array(pred_20T)

yhat_pred, yhat_20T =prepare_df_X_test(model, X_test, k)


def unscale(y_values, scaler):
   return scaler.inverse_transform(y_values)

def unscale_1(df, y_values, scaler):
   # it is more complex, because y_values here miss the Tamb column
   # we add artificial Tamb column and remove it again.
   #y_values_with_Tamb_fake = np.array([np.concatenate([row[:-2], np.array([1.0]), row[-2:]]) for row in y_values])
   ydf2_Tamb_Qheat = np.array(df.iloc[0:len(y_values), -2:])
   arr_fake=np.array(np.concatenate([y_values, ydf2_Tamb_Qheat], axis = 1))
   #res = scaler.inverse_transform(y_values_with_Tamb_fake)
   #return np.array([np.concatenate([row[:-3], row[-2:]]) for row in res])
   unscaled = unscale(arr_fake, scaler)
   unscaled=unscaled[:, :-2]
   return unscaled

y_pred_unscaled, y_test_unscaled = unscale_1(df, yhat_20T, scaler), unscale_1(df, y_test, scaler)


def plot(arr_y_pred, arr_y_test, orig_df):
   xdata = orig_df.iloc[len(orig_df)-len(X_test):, 0]
   df_y_pred = pd.DataFrame(arr_y_pred)
   df_y_test = pd.DataFrame(arr_y_test) # arry_y_pred you took here!
   legends_test =['OrgT' + str(i) for i in range (1, 23) ]
   legends_pred =['PrT' + str(i) for i in range (1, 23) ]
   for i, j in zip(df_y_pred, df_y_test):
       plt.plot(xdata, df_y_pred.iloc[:, i], label = legends_pred[i])
       #plt.plot(xdata, df_y_test.iloc[:, j], label = legends_test[i])

   plt.legend()
   plt.show()
   return

plt.clf()
plot(y_pred_unscaled, y_test_unscaled, orig_df)
#plot(yhat, y_test, orig_df)
