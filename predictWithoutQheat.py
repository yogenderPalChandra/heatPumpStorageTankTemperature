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


df_t1_20, orig_df = load_data(fpath='./dleteItTankambientloss')
df_hp = load_dataHP('deleteItHpData')
#df_new = load_data_big()


#df_t1_20_Amb_loss, orig_df_Amb_loss = load_data_tankLoss()
#df_hp_Amb_loss = load_dataHP_tankLoss()

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


X = df_nrm.iloc[:, :-1]
y = pd.DataFrame(pd.concat([df_nrm.iloc[:, :-2], df_nrm.iloc[:, -1]], axis = 1))

#X, y = prepare_df_Q_t(X, y, k)

n_output_features = y.shape[1]
n_input_features = X.shape[1]

X = pd.DataFrame(X)
#X.columns = df.columns
y = pd.DataFrame(y)
#y.columns = df.columns



k = 3
epochs = 2000
batch_size = 3000
n_features = X.shape[1]


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


#X_train, X_test, y_train, y_test = train_test_split(Xdf1, ydf2, test_size=0.2, random_state=42, shuffle=False)
#X_train, X_test, y_train, y_test = train_test_split(Xdf1, y_df1, test_size=0.2, random_state=42, shuffle=False)

train_indexes, test_indexes = train_test_split_indexes(Xdf1, ydf2, test_size=0.2, random_state=42, shuffle=True)

X_train, X_test, y_train, y_test = splitter(pd.DataFrame(Xdf1), pd.DataFrame(ydf2), train_indexes, test_indexes)

X_train = np.array(X_train)

X_test = np.array(X_test)

y_train = np.array(y_train)

y_test = np.array(y_test)

def create_model(time_steps, n_features):
    model = Sequential()
    model.add(LSTM(20, input_shape = (time_steps, n_features)))
    model.add(Dense(y_train.shape[1], activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

model = create_model(k, n_features)
model_fpath="./Tank_heat_loss_In20TPlusTambOut20TQh"
callbacks_list = [ ModelCheckpoint(filepath=model_fpath,
                                   monitor="val_loss",
                                   save_best_only=True,
                                   mode="min")]


history = model.fit(X_train.reshape(X_train.shape[0], k, n_features),
                    y_train.reshape(y_train.shape[0], y_train.shape[1]),
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.3,
                    callbacks = callbacks_list,
                    verbose=1)


model.save(model_fpath)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


from keras.models import load_model
model = load_model(model_fpath)


def prepare_df_X_test(model, arr, y_test, k):
    #model = load_model('20ToutPlus20TTambQhInput.h5')
    """
    1. Did not! work! I have feeling that This algo dosent work because - Q has alot of zeros and
    Q is used in the X, hence its not calculating weights properly. So intutively - the only
    solution remained is to predict the Q (i.e. put Q in y). In other words, try to predict Q
    by 20T and Tamb as X and 20 T and Q in y. this way Q would be in Y and its just mapping
    for the y and Q has not to be in the X. this is the solution if this algo has to work
    at all. 
    2. this algo did not work as well. So predicting from the previous predicted rows actually dosent work.

    3. Should eliminate Q overall. and just predict 20T Temperatures form 20T and Tamb mayeb then this algo works?

    
    """
    n_rows, n_cols = arr.shape
    
    Tamb = [[i[-1] for i in j ] for j in X_test.reshape(X_test.shape[0],k, n_features)] #sanity check done!
    TQ= [row for row in np.array(Tamb).flatten()[3:]] #sanity check done
    #TQ_pairwise = [[np.array([TQ[l],TQ[m]])] for l, m in zip(range(0, len(TQ), 2),range(1, len(TQ),2))] #sanity check done
    

    pred=[np.array(xi) for xi in arr[0].reshape( k, n_features)] #it is definately arr[0], otherwise Q & Tamb are incorrect
    #in TQ and TQ_pairwise
    #pred_20T_Q=[x[:-1] for x in flatten_row_wise(arr[0]).reshape(k,n_features)] #sanity check done
    pred_20T_Q = [np.append(x, y) for x, y in zip(arr[0].reshape(k, n_features)[:, :-1], y_test[0:3, -1])]
    #print ('length of pred=',len(pred),pred)
    for i in range(k, n_rows):

        pred_shaped = flatten_row_wise(pred[i-k:i]).reshape(1, k, n_features ) #sanity check done
        yhat=model.predict(pred_shaped)
        pred_20T_Q.append(np.squeeze(yhat,axis=(0,)))                             #sanity check done
        #yhat_faked = np.concatenate([np.expand_dims(yhat.flatten()[:-1], axis=0),np.expand_dims(TQ[i-3], axis=0)], axis=1) #sanity check = done (2X)
        yhat_faked = np.append(yhat.flatten()[:-1], TQ[i-3]) #sanity check = done (2X)
    
        #pred.append(np.squeeze(yhat_faked, axis=(0,)))
        pred.append(yhat_faked)#sanity check done
        print ('looping next loop', i)
        
    return pred, np.array(pred_20T_Q)

yhat_pred_Tamb, yhat_20T_Q =prepare_df_X_test(model, X_test, y_test, k)

pred = [np.random.randint(5, size=(2, 4)), np.random.randint(3, size=(2, 4)), np.random.randint(7, size=(2, 4))]
pred=[np.array(xi) for xi in X_train[1].reshape( k, n_features)]
pred.append(np.random.randint(2, size=(22,)))
pred_shaped = flatten_row_wise(pred[0:3]).reshape(1, k, n_features)
yhat=model.predict(pred_shaped)
Tamb_Qh = [[i[-2:] for i in j ] for j in X_test.reshape(X_test.shape[0],k, n_features)]
yhat_faked = np.concatenate([yhat, np.expand_dims(Tamb_Qh[1][-1], axis=0)], axis=1)
pred.append(yhat_faked)
pred.append(np.squeeze(yhat_faked, axis=(0,))


def unscale(y_values, scaler):
   return scaler.inverse_transform(y_values)

def unscale_1(df, y_values, scaler):
   # it is more complex, because y_values here miss the Tamb column
   # we add artificial Tamb column and remove it again.
   #y_values_with_Tamb_fake = np.array([np.concatenate([row[:-2], np.array([1.0]), row[-2:]]) for row in y_values])
   ydf2_Qheat = np.array(df.iloc[0:len(y_values), -1])
   ydf2_Tamb = np.array(df.iloc[0:len(y_values), -2])
   #arr_fake=np.array(np.concatenate([pred_values[:, :-1], np.expand_dims(ydf2_Tamb, axis=1), np.expand_dims(pred_values[:, -1], axis = 1)], axis = 1))
   arr_fake=np.array(np.concatenate([y_values[:, :-1], np.expand_dims(ydf2_Tamb, axis=1), np.expand_dims(y_values[:, -1], axis = 1)], axis = 1))
   #res = scaler.inverse_transform(y_values_with_Tamb_fake)
   #return np.array([np.concatenate([row[:-3], row[-2:]]) for row in res])
   unscaled = unscale(arr_fake, scaler)
   #unscaled_y = unscale(y_test_fake, scaler)
   col2slice = [i for i in range(0, 20)] + [21]
   unscaled=unscaled[:, col2slice]
   #y_test_unscaled = unscaled_y[:, :-1]
   return unscaled

y_pred_unscaled, y_test_unscaled = unscale_1(df,yhat_20T_Q, scaler), unscale_1(df,y_test, scaler)


y_pred_unscaled_tem = y_pred_unscaled[:, 3]

y_test_unscaled_tem =  y_test_unscaled[:, 3]

plt.scatter(y_test_unscaled_tem, y_pred_unscaled_tem)
plt.show()



df1 = pd.DataFrame(yhat_20T_Q[:, :-1])
df2 = pd.DataFrame(yhat_20T_Q)
df3 = pd.DataFrame(ydf2_Tamb)
df4 = pd.DataFrame(yhat_20T_Q[:, -1])

df5 = pd.DataFrame(y_test_fake)
arr_fake=pd.DataFrame(np.array(np.concatenate([yhat_20T_Q[:, :-1], np.expand_dims(ydf2_Tamb, axis=1), np.expand_dims(yhat_20T_Q[:, -1], axis = 1)], axis = 1))


def plot(arr_y_pred, arr_y_test, orig_df):
   xdata = orig_df.iloc[len(orig_df)-len(X_test):, 0]
   df_y_pred = pd.DataFrame(arr_y_pred)
   df_y_test = pd.DataFrame(arr_y_test) # arry_y_pred you took here!
   legends_test =['OrgT' + str(i) for i in range (1, 23) ]
   legends_pred =['PrT' + str(i) for i in range (1, 23) ]
   for k in [df_y_pred ,df_y_test ]:
       plt.figure()
       for i in range (0,20):
           #plt.plot(xdata, df_y_pred.iloc[:, i], label = legends_pred[i])
           plt.plot(xdata, k.iloc[:, i], label = legends_test[i])

   plt.legend()
   plt.show()
   return

plt.clf()
plot(y_pred_unscaled, y_test_unscaled, orig_df)
#plot(yhat, y_test, orig_df)


def plot(arr_y_pred, arr_y_test, orig_df):
   xdata = orig_df.iloc[len(orig_df)-len(X_test):, 0]
   df_y_pred = pd.DataFrame(arr_y_pred)
   df_y_test = pd.DataFrame(arr_y_test) # arry_y_pred you took here!
   legends_test =['OrgT' + str(i) for i in range (1, 23) ]
   legends_pred =['PrT' + str(i) for i in range (1, 23) ]
   for i, j in zip(df_y_pred, df_y_test):
       #plt.plot(xdata, df_y_pred.iloc[:, i], label = legends_pred[i])
       plt.plot(xdata, df_y_test.iloc[:, j], label = legends_test[i])

   plt.legend()
   plt.show()
   return

plt.clf()
plot(y_pred_unscaled, y_test_unscaled, orig_df)
#plot(yhat, y_test, orig_df)



df_missing =  df.isna()
df_missing_sum = df_missing.sum()
df_missing_per = df_missing_sum/len(df)

df.fillna(0).count() / len(df)

df.isna().mean().round(4) * 100

(df == 0).count()

df[['T1']].apply( lambda s : s.value_counts().get(key=0,default=0), axis=1).sum() /len(df[['T1']])

#df.apply( lambda s : s.value_counts().get(key=0,default=0), axis=1).sum() /len(df)



'''
def plot(arr_y_pred, arr_y_test, orig_df):
   xdata = orig_df.iloc[len(orig_df)-len(X_test):, 0]
   df_y_p1red = pd.DataFrame(arr_y_pred)
   df_y_test = pd.DataFrame(arr_y_test) # arry_y_pred you took here!
   legends_test =['OrgT' + str(i) for i in range (1, 23) ]
   legends_pred =['PrT' + str(i) for i in range (1, 23) ]
   for i in range (0,20):
       for k in [df_y_pred ,df_y_test ]:
           plt.figure()
           #plt.plot(xdata, df_y_pred.iloc[:, i], label = legends_pred[i])
           plt.plot(xdata, k.iloc[:, i], label = legends_test[i])

   plt.legend()
   plt.show()
   return

plt.clf()
plot(y_pred_unscaled, y_test_unscaled, orig_df)
#plot(yhat, y_test, orig_df)
'''
