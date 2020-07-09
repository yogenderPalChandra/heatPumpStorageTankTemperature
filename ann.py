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

from utility import load_data, normalize, flatten_row_wise, prepare_df, create_ann_30_model, unscale, prediction_vs_truth_plot
df, orig_df = load_data()
df_nrm, scaler = normalize(df)
df_nrm = pd.DataFrame(df_nrm)

##############################################
# for ANN taking k last time-point m values create an ANN
##############################################

k = 3

n_output_features = df_nrm.shape[1]
n_input_features = k * n_output_features

epochs=300
batch_size=100

###############################################
# take a data frame and generate sample input and output data
###############################################

X, y = prepare_df(df_nrm, k)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

model = create_ann_30_model(n_input_features, n_output_features)

history = model.fit(X_train,
                    y_train,
                    epochs = epochs,
                    batch_size = batch_size,
                    shuffle=True,
                    validation_data = (X_test, y_test))



epochs = 1200
batch_size = 200
model = create_ann_30_model(n_input_features, n_output_features)

history = model.fit(X_train,
                    y_train,
                    epochs = epochs,
                    batch_size = batch_size,
                    shuffle=True,
                    validation_data = (X_test, y_test))
model.save("model.h5")




model = load_model('model.h5')

prediction_vs_truth_plot(model, X_test, y_test, unscale=unscale, scaler=scaler)








