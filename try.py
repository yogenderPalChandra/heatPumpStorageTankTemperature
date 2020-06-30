# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
path = "/home/yogender/Desktop/KaggleHousePricePrediction/dlTrnsys/dltrnsys/dlOne"
path2 = "/home/yogender/Desktop/KaggleHousePricePrediction/dlTrnsys/dltrnsys/dlone2"
df = pd.read_csv(path, header = 1, encoding = "ISO-8859-1", sep='\t', skiprows=0)
df2 = pd.read_csv(path2, header = 1, encoding = "ISO-8859-1", sep='\t', skiprows=0) 
##heatToLoad = df.iloc[:, -5]
##heatToLoad = pd.to_numeric(heatToLoad)


#heatEnergy is kj/hr given out by hp.
#to convert it to kWh: convert it to kW by dividing it with 3600
#multiply it with time step: 5 sec/3600. and add all

##heatEnergy = heatToLoad.sum()
##
##heatEnergy =(1/21600)*heatEnergy
##
##heatEnergy =(5/12960000)*heatEnergy
##
##heatEnergy =(1/12960000)*heatEnergy
##hotWaterConsumption = df.iloc[:, -2]
##
##hotWaterConsumptionTotal = (1/6)*hotWaterConsumption.sum()


#exclude the last comumn of nan

df = df.dropna(axis='columns')
df.astype(float)

df.dtypes  

#df = df.iloc[:, 0:-1]

#df_numeric = pd.to_numeric(df)

df.columns = ['Hours', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20']

df_tem = df.drop(columns = 'Hours')

#plotting the hist to see seasonality

from matplotlib import pyplot
df_tem.hist()
pyplot.show()

# create a differenced series

def difference(df):
    return df.diff()

df_difference = difference(df_tem)

df_difference = df_difference.dropna()

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

data = series_to_supervised(df_difference)

df_tem = df.drop(columns = 'Hours')


#To check whats the data in lag+forward dat look like
data[['T1(t-1)', 'T1(t)']]


#creating a list of list of input/output payer
def df2lol(df):
    lol = []
    for i in range(1, 20):
        x = [df[['T{Y}(t-1)'.format(Y=i)]], df[['T{X}(t)'.format(X = i)]]]
        #print (x)
        lol.append(x)
    return lol

lol_df = df2lol(data)



scaler = MinMaxScaler(feature_range=(-1, 1))

scaler = scaler.fit(data)

scaled_data = scaler.transform(data)

#X, y = data.iloc[:, 0:20], data.iloc[:, 20:]

X, y = scaled_data[:, 0:20], scaled_data[:, 20:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def func2reshape(x_train, x_test, yy_train, yy_test):
    X_train_reshaped = x_train.reshape(len(x_train), 20, 1)
    X_test_reshaped = x_test.reshape(len(x_test), 20, 1)
    y_train_reshaped = yy_train.reshape(len(yy_train), 20, 1)
    y_test_reshaped = yy_test.reshape(len(yy_test), 20, 1)
    return X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped

X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped=func2reshape(X_train, X_test, y_train, y_test)


def func2reshape(x_train, x_test, yy_train, yy_test):
    X_train_reshaped = x_train.reshape(1,len(x_train), 20)
    X_test_reshaped = x_test.reshape(1, len(x_test), 20)
    y_train_reshaped = yy_train.reshape(1,len(yy_train), 20)
    y_test_reshaped = yy_test.reshape(1, len(yy_test), 20)
    return X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped

X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped=func2reshape(X_train, X_test, y_train, y_test)

##def scale(train, test):
##	# fit scaler
##	scaler = MinMaxScaler(feature_range=(-1, 1))
##	scaler = scaler.fit(train)


def fit_lstm(X_train, y_train):
    model = Sequential()
    model.add(LSTM(32, input_shape =(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(20))
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer='adam')
    history=model.fit(X_train, y_train, epochs=20, batch_size = 1, verbose=2)
    #scores = model.evaluate(X, y, verbose=0)
    #print("Model Accuracy: %.2f%%" % (scores[1]*100))
    print(history.history.keys())
    print(history.history['val_loss'])
    return model

lstm_model = fit_lstm(X_train_reshaped, y_train_reshaped)



def fit_lstm (l_df):
    model = sequential()
    model.add(LSTM(32), input_shape =(5760, 1))
    
    for i in l_dg:
        X = i[0]
        #.valued creates the underlying numpy array of the pandas df
        #so that .reshape can be used
        X = X.values
        y = i[1]
        y =y.values
        X = X.rehaspe(1, X.shape[0], 1)
        

             
        
        
