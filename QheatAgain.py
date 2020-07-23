###############################################
# T(t) -> Q(t)
###############################################

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

df_t1_20, orig_df = load_data()
df_hp = load_dataHP()
df_new = load_data_big()

df = pd.concat([df_t1_20, df_hp.loc[:, ["Tamb", "KJ/hr"]]], axis=1)
df_nrm, scaler = normalize(df)
df_nrm = pd.DataFrame(df_nrm)
df_nrm.columns = df.columns

X = df_nrm.iloc[:, :]
y = pd.DataFrame(df_nrm.iloc[:, :-2])

train_indexes, test_indexes = train_test_split_indexes(X, y, test_size=0.2, random_state=42, shuffle=True)

X_train, X_test, y_train, y_test = splitter(X, y, train_indexes, test_indexes)

"""
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

def create_Q_t_model(X, y):
    n_input_features = X.shape[1]
    n_output_features = y.shape[1]
    model = Sequential()
    model.add(Dense(3, input_shape=(n_input_features,), activation='relu'))
    model.add(Dense(n_output_features, activation='linear'))
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_squared_error"])
    return model
"""

# to control randomness!
numpy_seed=123
tf_seed=42

from numpy.random import seed
seed(numpy_seed)
try:
    from tensorflow import set_random_seed
    set_random_seed(tf_seed)
except:
    from tensorflow.random import set_seed
    set_seed(tf_seed)




epochs = 60 # already at 60 val_loss > loss
batch_size = 300

model = create_Q_t_model(X, y)
model_fpath = "./Q_t_model.h5"
callbacks_list = get_callbacks(model_fpath)
history = model.fit(X_train,
                    y_train,
                    epochs = epochs,
                    batch_size = batch_size,
                    shuffle=True,
                    callbacks = callbacks_list,
                    validation_split=0.3)
model.save('modelBigDatajosephusQheat')

# In [60]: minimal_val_loss(history)                                                                                                                                                   
# Best val_loss is: 0.12757985863319704
# Out[60]: 0.12757985863319704


#########################
##Seeing outliers in data 
#########################

def plotOutlier(df_orig, df_hp):
    import seaborn as sns; sns.set()
    import matplotlib.pyplot as plt
    df_snsScatterPlot = pd.concat([df_orig, df_hp.loc[:, ["Tamb", "KJ/hr"]]], axis=1)
    ax = sns.scatterplot(x="Hours", y="KJ/hr", data =df_snsScatterPlot)
    plt.show()
    return 


plottingOutliers = plotOutlier(orig_df, df_hp)


import seaborn as sns
import matplotlib.pyplot as plt

from utility import plot_losses, minimal_val_loss

minimal_val_loss(history)
plot_losses(history) # saved as Q_t_losses.png

from keras.models import load_model
model = load_model('modelBigDatajosephusQheat')
from utility import unscale, yogi_prediction_vs_truth_plot_Q_t
yogi_prediction_vs_truth_plot_Q_t(model, X_test, y_test, scaler, unscale)
prediction_vs_truth_plot_Q_t(model, X_test, y_test, scaler, unscale)
# saved as Q_t_prediction_vs_truth.svg

# prediction is very bad!!

################
##prediction against hours
################



def plot(model, X_test, y_test, scaler,out_fpath=None):
    import matplotlib.pyplot as plt
    from utility import unscale, prediction_vs_truth_plot_Q_t
    yhat=model.predict(X_test)
    yhat_df = pd.DataFrame(yhat)
    yhat_df.index = X_test.index
    yhat_df = pd.concat([yhat_df, X_test.iloc[:, -2:]], axis=1) # why suddenly NaN?
    ytest_df = pd.concat([y_test, X_test.iloc[:, -2:]], axis=1)
    y_pred_unscaled, y_test_unscaled = unscale(yhat_df, ytest_df, scaler)

    timelen = len(df)-len(X_test)
    xdata = orig_df.iloc[timelen:, 0]
    df_y_pred_unscaled=pd.DataFrame(y_pred_unscaled)
    df_y_test_unscaled=pd.DataFrame(y_test_unscaled)
    #df_y_pred = pd.DataFrame(arr_y_pred)
    #df_y_test = pd.DataFrame(arr_y_test) # arry_y_pred you took here!
    #legends_test =['OrgT' + str(i) for i in range (1, 23) ]
    #legends_pred =['PrT' + str(i) for i in range (1, 23) ]
    plt.plot(xdata, df_y_pred_unscaled.iloc[:, 0:20], marker='o', ms=0.5 )
    #plt.plot(xdata, df_y_test_unscaled.iloc[:, 0:20])

    plt.legend()
    plt.show()
    return

plt.clf()
#plot(y_pred_unscaled, y_test_unscaled, orig_df)
plot (model, X_test, y_test, scaler)




'''
def load_data(fpath="./dlOne", col_names = ['Hours'] + ["T" + str(i) for i in range(1, 21)]):
    df = pd.read_csv(fpath, 
                     header = 1, 
                     encoding = "ISO-8859-1", 
                     sep='\t', 
                     skiprows=0).dropna(axis=1).astype(float)
    df.columns = col_names
    return df.drop(columns = 'Hours'), df

df_t1_20, orig_df = load_data()


df_snsScatterPlot = pd.concat([orig_df, df_hp.loc[:, ["Tamb", "KJ/hr"]]], axis=1)
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
ax = sns.scatterplot(x="Hours", y="KJ/hr", data =df_snsScatterPlot)
plt.show()

###
'''

###############
#yogi function for 20T+Tamb+Qheat input, 20T out, but X_test only first row. 
###############

def flatten_row_wise(df):
    """Take row by row and attach to one flat single row."""
    return np.ndarray.flatten(np.array(df))

def prepare_df_yogi(df, k):
    n_rows, n_cols = df.shape
    new_rows = np.array([flatten_row_wise(df.iloc[(i-k):i]) for i in range(k, n_rows)])
    new_ys = np.array([row for row in df.iloc[(k):, :].itertuples(index=False)])
    return new_rows

def prepare_df_X_test(model, df1, k):
    model = load_model('modelBigDatajosephusQheat')
    n_rows, n_cols = df1.shape
    print (df1.shape)
    #X_text = np.array([flatten_row_wise(df.iloc[i]) for i in range(k)])
    #new_rows = np.array([flatten_row_wise(df.iloc[0:i]) for i in range(k, n_rows)])
    #new_rows = np.array([flatten_row_wise(df.iloc[(i-k):i]) for i in range(k, n_rows)])
    pred = [np.array([flatten_row_wise(df1.iloc[i]) for i in range(k)])]
    print (pred)
    for i in range(k, n_rows):
        #X_values =  np.array([flatten_row_wise(df.iloc[(i-k):i])])
        
        yhat=model.predict(pred[i-k:i])
        #predd.append(pd.concat([X_test, yhat],  ignore_index=True))
        pred.append(yhat)
        
    return nX, ny

prepare_df_X_test(model, X_test, 3)
    

def create_lstm_10_20_model(time_steps, n_features):
    model = Sequential()
    model.add(LSTM(10, input_shape = (time_steps, n_features)))
    model.add(Dense(20, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

# to control randomness!
numpy_seed=123
tf_seed=42

from numpy.random import seed
seed(numpy_seed)
try:
    from tensorflow import set_random_seed
    set_random_seed(tf_seed)
except:
    from tensorflow.random import set_seed
    set_seed(tf_seed)


from utility import create_Q_t_model_1, get_callbacks

epochs = 60 # already at 60 val_loss > loss
batch_size = 300

model = create_Q_t_model(X, y)
model_fpath = "./Q_t_model_1.h5"
callbacks_list = get_callbacks(model_fpath)
history = model.fit(X_train,
                    y_train,a8a7047
                    epochs = epochs,
                    batch_size = batch_size,
                    shuffle=True,
                    callbacks = callbacks_list,
                    validation_split=0.5)
model.save(model_fpath)

# overfits already quite early! 

# it seems Q_t needs some more steps T1_20(t-k) in advance?













###############################################
# T(t-k) + Q (t-k) + Tamb(t-k) -> 20T(t) && only (20(T-1))+Q ->20T only using one layer of 20T and all
#the layers of Q gives rest of the 20T values
###############################################

k = 3
epochs = 100
batch_size = 300


import pandas as pd
import numpy as np
import os
from utility import load_data, load_dataHP, normalize, train_test_split_indexes, splitter, flatten_row_wise, prepare_df_Q_t
from utility import create_Q_t_model, get_callbacks


import pandas as pd
import numpy as np
import os
from utility import load_data, load_dataHP, normalize, train_test_split_indexes, splitter, load_data_big
from utility import create_Q_t_model, get_callbacks

from utility import plotOutlier
import matplotlib.pyplot as plt

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



#epochs = 60 # already at 60 val_loss > loss
#batch_size = 300


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

yhat=model.predict(X_test.reshape(X_test.shape[0], k, n_features))

plt.scatter(y_test, yhat)
plt.show()



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


y_pred_unscaled, y_test_unscaled = unscale_1(df, yhat, scaler), unscale_1(df, y_test, scaler)



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
plot(yhat, y_test, orig_df)



####
##predicting with single row:
####


def flatten_row_wise(df):
    """Take row by row and attach to one flat single row."""
    return np.ndarray.flatten(np.array(df))

def prepare_df_yogi(df, k):
    n_rows, n_cols = df.shape
    new_rows = np.array([flatten_row_wise(df.iloc[(i-k):i]) for i in range(k, n_rows)])
    new_ys = np.array([row for row in df.iloc[(k):, :].itertuples(index=False)])
    return new_rows

def prepare_df_X_test(model, arr, k):
    model = load_model('20ToutPlus20TTambQhInput.h5')
    n_rows, n_cols = arr.shape
    print (arr.shape)
    #X_text = np.array([flatten_row_wise(df.iloc[i]) for i in range(k)])
    #new_rows = np.array([flatten_row_wise(df.iloc[0:i]) for i in range(k, n_rows)])
    #new_rows = np.array([flatten_row_wise(df.iloc[(i-k):i]) for i in range(k, n_rows)])
    #pred = [np.array([flatten_row_wise(df1.iloc[i]) for i in range(k)])]
    
    Tamb_Qh = [[i[-2:] for i in j ] for j in X_test.reshape(X_test.shape[0],k, n_features)]
    #pred = [np.array([flatten_row_wise(arr[1])]).reshape(1, k, n_features)]
    #pred = np.array([flatten_row_wise(arr[1])]).reshape(1, k, n_features)
    pred = flatten_row_wise(arr[1]) #66    
    print (pred)
    for i in range(k, n_rows):
        #X_values =  np.array([flatten_row_wise(df.iloc[(i-k):i])])
        
        yhat=model.predict(pred[i-k:i])
        yhat_faked = np.concatenate([yhat, np.expand_dims(Tamb_Qh[i-2][-1], axis=0)], axis=1)
        #predd.append(pd.concat([X_test, yhat],  ignore_index=True))
        #pred.append(yhat_faked)
        np.append(pred, yhat_faked, axis =1)
        
    return pred

yhat_pred = prepare_df_X_test(model, X_test, k)

def prepare_df_X_test(model, arr, k):
    model = load_model('20ToutPlus20TTambQhInput.h5')
    n_rows, n_cols = arr.shape
    
    Tamb_Qh = [[i[-2:] for i in j ] for j in X_test.reshape(X_test.shape[0],k, n_features)]

    pred=[np.array(xi) for xi in arr[0].reshape( k, n_features)]
    pred_20T=[x[:-2] for x in flatten_row_wise(arr[0]).reshape(k,n_features)]
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


##def list2array(arr):
##    return np.array(arr)
##yaht_20T_arr = list2array(arr)

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
       #plt.plot(xdata, df_y_pred.iloc[:, i], label = legends_pred[i])
       plt.plot(xdata, df_y_test.iloc[:, j], label = legends_test[i])

   plt.legend()
   plt.show()
   return

plt.clf()
plot(y_pred_unscaled, y_test_unscaled, orig_df)
#plot(yhat, y_test, orig_df)


'''

pred = [np.random.randint(5, size=(2, 4)), np.random.randint(3, size=(2, 4)), np.random.randint(7, size=(2, 4))]
pred=[np.array(xi) for xi in X_train[1].reshape( k, n_features)]
pred.append(np.random.randint(2, size=(22,)))
pred_shaped = flatten_row_wise(pred[0:3]).reshape(1, k, n_features)
yhat=model.predict(pred_shaped)
Tamb_Qh = [[i[-2:] for i in j ] for j in X_test.reshape(X_test.shape[0],k, n_features)]
yhat_faked = np.concatenate([yhat, np.expand_dims(Tamb_Qh[1][-1], axis=0)], axis=1)
pred.append(yhat_faked)
pred.append(np.squeeze(yhat_faked, axis=(0,))
for i in range(5):
    pred.append(i)

for i in range(k, n_rows):
    pred_shaped = [flatten_row_wise(pred[i-k:i]).reshape(1, k, n_features )]
print (pred_shaped)

    
for i in range(22, 88, 22):
    print (np.array(pred[i-22:i]))
n = np.random.randint(10, 50, size=(10, 3, 2))

m =[[np.array(i) for i in j]for j in n ] 
q =[[j[i+1][-1]] for i, j in enumerate(m)]
q =[[m[i-2][-1]] for i in range(k, len(m))]
Tamb_Qh[i-2][-1]
q =[Tamb_Qh[i-2][-1] for i in range(k, n_rows)]
for i in range(k, len(m)):
    print (len(m[i]))

for i in q:
    print (np.expand_dims(i, axis=0).shape)
'''
'''

#Tester functions
Tamb_Qh = [[i[-2:] for i in j ]for j in X_test.reshape(X_test.shape[0],k, n_features)]
Tamb_Qh = [[i.loc[-2:] for i in j ]for j in X_test.reshape(X_test.shape[0],k, n_features)]

for i in Tamb_Qh :
    print (len(i))

for i in Tamb_Qh:
    #print ([j[-1].isin([-1]).any() for j in i])
    print ([j[-1] for j in i])


df.where(condition).count()

print('Non -1 number found:')

#To check there are no only -1 values of Qheat in Tamb_Qh object
for i in Tamb_Qh:
    print('Non -1 number found:')
    i.eq(-1).sum()
    #print(i.iloc[:, -1].where(i != -1).count())


'''
def unscale(y_values, scaler):
   return scaler.inverse_transform(y_values)
'''



'''
model = create_lstm_10_20_model(time_steps, n_features)
model_fpath = "./Q_t_model_1.h5"
callbacks_list = get_callbacks(model_fpath)
history = model.fit(X_train,
                    y_train,a8a7047
                    epochs = epochs,
                    batch_size = batch_size,
                    shuffle=True,
                    callbacks = callbacks_list,
                    validation_split=0.5)
model.save(model_fpath)
'''


