###############################################
# T(t) -> Q(t)
###############################################

import pandas as pd
import numpy as np
import os
from utility import load_data, load_dataHP, normalize, train_test_split_indexes, splitter, load_data_big
from utility import create_Q_t_model, get_callbacks

from utility import plotOutlier


df_t1_20, orig_df = load_data()
df_hp = load_dataHP()
df_new = load_data_big()

df = pd.concat([df_t1_20, df_hp.loc[:, ["Tamb", "KJ/hr"]]], axis=1)
df_nrm, scaler = normalize(df)
df_nrm = pd.DataFrame(df_nrm)
df_nrm.columns = df.columns

X = df_nrm.iloc[:, :-1]
y = pd.DataFrame(df_nrm.iloc[:, -1])

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

'''
def plotOutlier(df_orig, df_hp):
    import seaborn as sns; sns.set()
    import matplotlib.pyplot as plt
    df_snsScatterPlot = pd.concat([df_orig, df_hp.loc[:, ["Tamb", "KJ/hr"]]], axis=1)
    ax = sns.scatterplot(x="Hours", y="KJ/hr", data =df_snsScatterPlot)
    plt.show()
    return
'''

plottingOutliers = plotOutlier(orig_df, df_hp)


import seaborn as sns
import matplotlib.pyplot as plt

from utility import plot_losses, minimal_val_loss

minimal_val_loss(history)
plot_losses(history) # saved as Q_t_losses.png

model = load_model('modelBigDatajosephusQheat')
from utility import unscale, prediction_vs_truth_plot_Q_t
prediction_vs_truth_plot_Q_t(model, X_test, y_test, scaler, unscale)
# saved as Q_t_prediction_vs_truth.svg

# prediction is very bad!!

################
##prediction against hours
################



def plot(model, X_test, y_test, scaler,out_fpath=None):
    from utility import unscale, prediction_vs_truth_plot_Q_t
    yhat=model.predict(X_test)
    yhat_df = pd.DataFrame({"KJ/hr": [x[0] for x in yhat]})
    yhat_df.index = X_test.index
    yhat_df = pd.concat([X_test, yhat_df], axis=1) # why suddenly NaN?
    ytest_df = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)
    y_pred_unscaled, y_test_unscaled = unscale(yhat_df, ytest_df, scaler)

    timelen = len(df)-len(X_test)
    xdata = orig_df.iloc[timelen:, 0]
    df_y_pred_unscaled=pd.DataFrame(y_pred_unscaled)
    df_y_test_unscaled=pd.DataFrame(y_test_unscaled)
    #df_y_pred = pd.DataFrame(arr_y_pred)
    #df_y_test = pd.DataFrame(arr_y_test) # arry_y_pred you took here!
    #legends_test =['OrgT' + str(i) for i in range (1, 23) ]
    #legends_pred =['PrT' + str(i) for i in range (1, 23) ]
    #plt.plot(xdata, df_y_pred_unscaled.iloc[:, -1])
    plt.plot(xdata, df_y_test_unscaled.iloc[:, -1])

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
# T(t-k) -> Q(t)
###############################################

k = 3
epochs = 100
batch_size = 300


import pandas as pd
import numpy as np
import os
from utility import load_data, load_dataHP, normalize, train_test_split_indexes, splitter, flatten_row_wise, prepare_df_Q_t
from utility import create_Q_t_model, get_callbacks

df_t1_20, orig_df = load_data()
df_hp = load_dataHP().iloc[5000:11000, :] # here is the problem!! KJ/hr and Tamb are bad!!
# they are all 0.0!!

df = pd.concat([df_t1_20, df_hp.loc[:, ["Tamb", "KJ/hr"]]], axis=1)
df_nrm, scaler = normalize(df)
df_nrm = pd.DataFrame(df_nrm)
df_nrm.columns = df.columns

X = df_nrm.iloc[:, :-1]
y = pd.DataFrame(df_nrm.iloc[:, -1])

X, y = prepare_df_Q_t(X, y, k)

n_output_features = y.shape[1]
n_input_features = X.shape[1]



train_indexes, test_indexes = train_test_split_indexes(X, y, test_size=0.2, random_state=42, shuffle=True)

X_train, X_test, y_train, y_test = splitter(X, y, train_indexes, test_indexes)


