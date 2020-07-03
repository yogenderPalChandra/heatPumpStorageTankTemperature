import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import matplotlib.pyplot as plt

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

epochs=1800
batch_size=300

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


yhat=model.predict(X_test)

plt.scatter(y_test, yhat)
plt.show()

# def unscale(y_pred, y_test, scaler):
#     
#     y_pred_orig = scaler.inverse_transform(yhat)
#     y_test_orig = scaler.inverse_transform(y_test)
# 
#     return y_pred_orig, y_test_orig
# # well done! but better reusable would be a function which takes just each one.


def unscale(y_values, scaler):
    return scaler.inverse_transform(y_values)

y_pred_unscaled, y_test_unscaled = unscale(yhat, scaler), unscale(y_test, scaler)

plt.scatter(y_test_unscaled, y_pred_unscaled)
plt.show()

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


plot(y_pred_unscaled, y_test_unscaled, orig_df)
# the order of y_test (and thus y_pred) is random! That is why the plot looks shitty.
# but otherwiese well done with the function!

# to bring them into a meaningful order - consult orig_df and df and df_norm
# the last two we need to com from normed state (df_norm)
# to not-normed (df) and then to get info about 'Hours' column (orig_df).
# so we would need a function to order it using these three data frames!

# but at the end you need only df_norm because it was already ordered for
# Hours so the order information of Hours is already in the row-indexes of
# df_norm.

def get_indexes(y_test, df_nrm):
    # find out which of the rows in df_norm are exact the same like y_test!
    # For this, the best would be to create a dict where the key are the 
    # the 20 values (in tuple!) and value are index of the row
    dct = {tuple(row): idx for idx, row in df_nrm.iterrows()} # one of few cases where iterrows() output (idx, row) is good!
    # note you can't use a list as key in dict - but by changing them to tuple, it works!
    return [dct[tuple(row)] for row in y_test]

# so, now you can get for each of the y_test rows and index.

y_test_indexes = get_indexes(y_test, df_nrm)

# now, you can use the order in y_test_indexes and the current indexes of y_test
# to also retransform y_pred in the same way.
# you can also use a dict because it is kind of a mapping of the row orders!

def correct_order(test_array, actual_indexes):
    dct = {x: i for i, x in enumerate(actual_indexes)}
    return np.array([test_array[dct[x]] for x in actual_indexes])

y_test_corr, y_pred_corr = correct_order(y_test_unscaled, y_test_indexes), correct_order(y_pred_unscaled, y_test_indexes)

# now, with corrected order, we can plot
plot(y_pred_corr, y_test_corr, orig_df)
# but even then it looks shitty, because test are just not-coherent random picks

# ah but we could make it better by using index (which is kind of Hour scale)
# as x-axis measure!

def plot(arr_y_pred, arr_y_test, df_nrm):
    y_test_indexes = get_indexes(arr_y_test, df_nrm)
    y_test_unscaled, y_pred_unscaled = unscale(arr_y_test, scaler), unscale(arr_y_pred, scaler)
    y_test_corr, y_pred_corr = correct_order(y_test_unscaled, y_test_indexes), correct_order(y_pred_unscaled, y_test_indexes)
    df_y_pred = pd.DataFrame(y_pred_corr)
    df_y_test = pd.DataFrame(y_test_corr)
    legends_test =['OrgT' + str(i) for i in range (1, 21) ]
    legends_pred =['PrT' + str(i) for i in range (1, 21) ] 
    for i, j in zip(df_y_pred, df_y_test):
        plt.plot(y_test_indexes, df_y_pred.iloc[:, i], label = legends_pred[i])
        plt.plot(y_test_indexes, df_y_test.iloc[:, j], label = legends_test[i])

    plt.legend()
    plt.show()
    return

plot(yhat, y_test, df_nrm) # this looks much more ordered, however, the lineplot connecting all dots is not good.



def plot(arr_y_pred, arr_y_test, df_nrm):
    y_test_indexes = get_indexes(arr_y_test, df_nrm)
    y_test_unscaled, y_pred_unscaled = unscale(arr_y_test, scaler), unscale(arr_y_pred, scaler)
    y_test_corr, y_pred_corr = correct_order(y_test_unscaled, y_test_indexes), correct_order(y_pred_unscaled, y_test_indexes)
    df_y_pred = pd.DataFrame(y_pred_corr)
    df_y_test = pd.DataFrame(y_test_corr)
    legends_test =['OrgT' + str(i) for i in range (1, 21) ]
    legends_pred =['PrT' + str(i) for i in range (1, 21) ] 
    for i, j in zip(df_y_pred, df_y_test):
        plt.scatter(y_test_indexes, df_y_pred.iloc[:, i], label = legends_pred[i])
        plt.scatter(y_test_indexes, df_y_test.iloc[:, j], label = legends_test[i])

    plt.legend()
    plt.show()
    return

plot(yhat, y_test, df_nrm) # that now looks right! Now make point size smaller

def plot(arr_y_pred, arr_y_test, df_nrm):
    y_test_indexes = get_indexes(arr_y_test, df_nrm)
    y_test_unscaled, y_pred_unscaled = unscale(arr_y_test, scaler), unscale(arr_y_pred, scaler)
    y_test_corr, y_pred_corr = correct_order(y_test_unscaled, y_test_indexes), correct_order(y_pred_unscaled, y_test_indexes)
    df_y_pred = pd.DataFrame(y_pred_corr)
    df_y_test = pd.DataFrame(y_test_corr)
    legends_test =['OrgT' + str(i) for i in range (1, 21) ]
    legends_pred =['PrT' + str(i) for i in range (1, 21) ] 
    for i, j in zip(df_y_pred, df_y_test):
        plt.scatter(y_test_indexes, df_y_pred.iloc[:, i], label = legends_pred[i], marker='.', s=0.2)
        plt.scatter(y_test_indexes, df_y_test.iloc[:, j], label = legends_test[i], marker='.', s=0.2)

    plt.legend()
    plt.show()
    return
    
plot(yhat, y_test, df_nrm) # looks nicer # through smaller point size, one can see better how prediction and actual value are close!


# one could plot each of them alone:

def plot_y_test(arr_y, df_nrm, scaler):
    y_indexes = get_indexes(arr_y, df_nrm)
    y_unscaled = unscale(arr_y, scaler)
    y_corr = correct_order(y_unscaled, y_indexes)
    df_y = pd.DataFrame(y_corr)
    legends =['T' + str(i) for i in range (1, 21) ]
    for i in df_y:
        plt.scatter(y_indexes, df_y.iloc[:, i], label = legends[i], marker='.', s=0.2)
    plt.legend()
    plt.show()
    return

plot_y_test(y_test, df_nrm, scaler)

# we can't use same function to plot yhat, because yhat deviates from df_nrm values (y_test).
# so for calculation of y_indexes, we need y_test!

def plot_y_pred(arr_y_pred, arr_y_test, df_nrm, scaler):
    y_indexes = get_indexes(arr_y_test, df_nrm)
    y_unscaled = unscale(arr_y_pred, scaler)
    y_corr = correct_order(y_unscaled, y_indexes)
    df_y = pd.DataFrame(y_corr)
    legends =['T' + str(i) for i in range (1, 21) ]
    for i in df_y:
        plt.scatter(y_indexes, df_y.iloc[:, i], label = legends[i], marker='.', s=0.2)
    plt.legend()
    plt.show()
    return

plot_y_pred(yhat, y_test, df_nrm, scaler)

def plotOriginaldf(df):
    xdata = df.iloc[:, 0]
    for i in range(1, 20):
        plt.plot(xdata, df.iloc[:, i])
    plt.show()

plotOriginaldf(orig_df)

# now write the last plot functions with title, x-axis-labels and y-axis-labels














def plot_y_test(arr_y, df_nrm, scaler, title, xlabel, ylabel):
    y_indexes = get_indexes(arr_y, df_nrm)
    y_unscaled = unscale(arr_y, scaler)
    y_corr = correct_order(y_unscaled, y_indexes)
    df_y = pd.DataFrame(y_corr)
    legends =['T' + str(i) for i in range (1, 21) ]
    for i in df_y:
        plt.scatter(y_indexes, df_y.iloc[:, i], label = legends[i], marker='.', s=0.2)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    return



# we can't use same function to plot yhat, because yhat deviates from df_nrm values (y_test).
# so for calculation of y_indexes, we need y_test!

def plot_y_pred(arr_y_pred, arr_y_test, df_nrm, scaler, title, xlabel, ylabel):
    y_indexes = get_indexes(arr_y_test, df_nrm)
    y_unscaled = unscale(arr_y_pred, scaler)
    y_corr = correct_order(y_unscaled, y_indexes)
    df_y = pd.DataFrame(y_corr)
    legends =['T' + str(i) for i in range (1, 21) ]
    for i in df_y:
        plt.scatter(y_indexes, df_y.iloc[:, i], label = legends[i], marker='.', s=0.2)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    return

def plotOriginaldf(df, title, xlabel, ylabel):
    xdata = df.iloc[:, 0]
    for i in range(1, 20):
        plt.plot(xdata, df.iloc[:, i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


plot_y_test(y_test, df_nrm, scaler, title="y_test", xlabel="timescale", ylabel="Temp")
plot_y_pred(yhat, y_test, df_nrm, scaler, title="y_pred", xlabel="timescale", ylabel="Temp")
plotOriginaldf(orig_df, title="orig_df", xlabel="timescale", ylabel="Temp")



















