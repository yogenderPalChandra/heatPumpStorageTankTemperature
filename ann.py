import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt
from keras.models import load_model


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

def normalize(X):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    scaled_data = scaler.transform(X)
    return scaled_data, scaler



df, orig_df = load_data()
df_nrm, scaler = normalize(df)
df_nrm = pd.DataFrame(df_nrm)

##############################################
# for ANN taking k last time-point m values create an ANN
##############################################

k = 3
n_values = 20

n_input = k * n_values

epochs=300
batch_size=3

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

"""
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
X, y = prepare_df(df)
looks correct!

last row is last y
X doesn't contain last row and begins from first row!
"""

X, y = prepare_df(df_nrm)
idxs = [x[0] for x in y]
y = np.array([np.array(x[1]) for x in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

def create_model(n_values):
    model = Sequential()
    model.add(Dense(30, input_shape=(n_input,), activation='relu'))
    model.add(Dense(n_values, activation='linear'))
    model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["mean_squared_error"])
    return model

model = create_model(n_values)

history = model.fit(X_train,
                    y_train,
                    epochs = epochs,
                    batch_size = batch_size,
                    shuffle=True,
                    validation_data = (X_test, y_test))

model.save("model.h5")
print ("model saved")





epochs = 1200
batch_size = 200
model = create_model(n_values)

history = model.fit(X_train,
                    y_train,
                    epochs = epochs,
                    batch_size = batch_size,
                    shuffle=True,
                    validation_data = (X_test, y_test))





model = load_model('loss_0016_val_loss_0018_model')


yhat=model.predict(X_test)

plt.scatter(y_test, yhat)
plt.show()

def unscale(y_pred, y_test, scaler):
    
    y_pred_orig = scaler.inverse_transform(yhat)
    y_test_orig = scaler.inverse_transform(y_test)

    return y_pred_orig, y_test_orig

y_pred_unscale, y_test_unscaled = unscale(yhat, y_test, scaler)






