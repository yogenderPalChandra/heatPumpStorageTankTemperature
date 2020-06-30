# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/





# open in rl git folder


import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

path = "./dlOne"
path_1 = "./dlone2"

col_names = ['Hours'] + ["T" + str(i) for i in range(1, 21)]

df = pd.read_csv(path, 
                 header = 1, 
                 encoding = "ISO-8859-1", 
                 sep='\t', 
                 skiprows=0,
                 columns=col_names).dropna(axis="columns").astype(float)
df_tem = df.drop(columns = 'Hours')

#plotting the hist to see seasonality
from matplotlib import pyplot
df_tem.hist()
pyplot.show()


"""
heatEnergy is kj/hr given out by hp.
to convert it to kWh: convert it to kW by dividing it with 3600
multiply it with time step: 5 sec/3600. and add all

heatEnergy = heatToLoad.sum()
heatEnergy =(1/21600)*heatEnergy
heatEnergy =(5/12960000)*heatEnergy
heatEnergy =(1/12960000)*heatEnergy
hotWaterConsumption = df.iloc[:, -2]
hotWaterConsumptionTotal = (1/6)*hotWaterConsumption.sum()

exclude the last comumn of nan
"""


def series_to_supervised(df_tem, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(df_tem) is list else df_tem.shape[1]
	df = df_tem
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('T%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('T%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('T%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


df_difference = df.diff().dropna(axis=0)
data = series_to_supervised(df_difference)

# # To check whats the data in lag+forward dat look like
# data[['T1(t-1)', 'T1(t)']]

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(data)
scaled_data = scaler.transform(data)

X, y = scaled_data[:, 0:20], scaled_data[:, 20:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def create_model(batch_size, time_steps, n_features):
    model = Sequential()
    model.add(LSTM(32, input_shape = (time_steps, n_features)))
    model.add(Dense(20, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model

epochs=20
batch_size = 1
time_steps = 1
n_features = X_train.shape[1]
n_samples = X_train.shape[0]
model = create_model(batch_size, time_steps, n_features)

history = model.fit(X_train.reshape(n_samples, time_steps, n_features),
                    y_train.reshape(n_samples, n_features),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2)

# to plot make data frame out of dict history.history and use .plot() method
pd.DataFrame(history.history).plot()
pyplot.show()







