# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

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
numpy_seed=123
tf_seed=42

from numpy.random import seed
seed(numpy_seed)
try:
    from tensorflow import set_random_seed
    set_random_seed(tf_seed)
except:
    import tensorflow.random import set_seed
    set_seed(tf_seed)

df, orig_df = load_data()
df_nrm, scaler = normalize(df)
df_nrm = pd.DataFrame(df_nrm)


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


X, y = prepare_df(df_nrm)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

model = create_lstm_10_20_model(k, n_features)
model_fpath="lstm.h5"

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

y_pred_unscaled, y_test_unscaled = unscale_2(yhat, scaler), unscale_2(y_test, scaler)

plt.clf()
plot(y_pred_unscaled, y_test_unscaled, orig_df)

print(f"Best val_loss is: {min(history.history['val_loss'])}")





















