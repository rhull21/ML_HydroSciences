# CREATED 10/28/2020 11:00 am
# MOVED to (below) at 10/28/2020 11:00 am
# /Users/roberthull/OneDrive/Workspace/Work/03_UA/04_Research/01_github/ParFlow_Hull_Git_b/04_ML 
# Script to predict streamflow at Verde River using LSTM approaches


# Resources: 
# 2020. Youchan Hu, Stream-Flow Forecasting of Small Rivers Based on LSTM
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://keras.io/api/layers/recurrent_layers/lstm/
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

# %%
# Questions
# # 1) How do we use large datasets without computer processing being 
# # impossible
# # 2) Better predictive variable than precip accum
# # 3) Scaling data
# # 4) Multiple Variable


# %% modules
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import json
import urllib.request as req
import urllib
import datetime
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math


# %% functions
def extractmasonet(base_url, args):
    """takes
       1) a string of 'base_url' 
       2) a dictionary of api arguments
       3) a string of 'token'
       specific to the metamot API

       See more about metamot:
       https://developers.synopticdata.com/mesonet/explorer/
       https://developers.synopticdata.com/about/station-variables/
       https://developers.synopticdata.com/mesonet/v2/getting-started/

       returns a dictionary
       containing the response of a JSON 'query'
       """

    # concat api arguments (careful with commas)
    apiString = urllib.parse.urlencode(args)
    apiString = apiString.replace('%2C', ',')

    # concat the API string to the base_url to create full_URL
    fullUrl = base_url + '?' + apiString
    print('full url =', fullUrl, '\n')

    # process data (use url to query data)
    # return as dictionary
    response = req.urlopen(fullUrl)
    responseDict = json.loads(response.read())

    return responseDict

# %%
# 0. Define Problem and initialize variables

# # 0.a Problem: Forecast Streamflow n hours into the future using data
# # #  from m hours into the past via LSTM

# # 0.b initialize varibles
# # # X var[=all-Q], Y var[=Q]
X_var = "Q1"
Y_var = "Q"
# # # m[=12] (number hrs past), n[=5] (number hrs future)
m = 12
n = 6 
# # # y[=7] (number train), z[=3] (number test)
y_n = 7
z_n = 3
# # # nodes[=64] (number hidden layers)
# # # optimizer[=Adam] (solver for model)
# # # batch_size[=72] (size of run, smaller = slower)
# # # epochs[=30] (times running model, higher = longer)
nodes = 64
optimizer = 'adam'
batch_size = 72
epochs = 10

# %%
# 1. Prepare Data
# # 1.a Assemble flow, precip, and temperature data (hourly)
# # # 1.a.1 ln flow (hourly) from Verde River USGS (data)
# # # # 1.a.1.a url
site = '09506000'
start = '1997-01-01'
end = '2020-12-31'
url = "https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on" \
      "&format=rdb&site_no="+site+ \
      "&period=&begin_date="+start+"&end_date="+end

# # # # 1.a.1.b read in data
data = pd.read_table(url, sep='\t', skiprows=32,
                     names=['agency_cd', 'site_no',
                            'datetime', 'tz', 'flow', 'code'],
                     parse_dates=['datetime'],
                     index_col='datetime'
                     )

# # # # 1.a.1.c re-instantiate data with just the flow
data = data[['flow']]
# # # # 1.a.1.d set tz as UTC-7 (MST)
data.index = data.index.tz_localize(tz="US/Arizona")
# # # 1.a.1.e resample hourly
data = data.resample('H').mean()

# %% 
# # 1.b Assemble temp, precip (hourly) from mesowest station QV...
# # # 1.b.1 mesowest api args
mytoken = 'demotoken'
base_url = "https://api.synopticdata.com/v2/stations/timeseries"
stationname = 'QVDA3'
args = {
        'start': '199701010000',
        'end': '202012310000',
        'obtimezone': 'local',
        'vars': 'air_temp,precip_accum',
        'stids': stationname,
        'units': 'temp|F,precip|mm',
        'token': mytoken}

# # # 1.b.2 extract dictionary of data response
responseDict = extractmasonet(base_url, args)
# # # 1.b.3 assemble items into a dataframe (df) 
for key, value in responseDict["STATION"][0]['OBSERVATIONS'].items():
    # creates a list of value related to key
    if (key == 'date_time'):
        # create index
        df = pd.DataFrame({key: pd.to_datetime(value)})
    else:
        # concat df
        df = pd.concat([df, pd.DataFrame({key: value})], axis=1)
# # # 1.b.4 set index of df
df = df.set_index('date_time')
# # # 1.b.5 resample hourly
df = df.resample('H').mean()
# # # 1.b.6 reset timezone 
df.index = df.index.tz_convert(tz="US/Arizona")
# # # 1.b.7 calculate hourly precip
df[['precip_accum_set_1_2']] = df[['precip_accum_set_1']].shift(1)
df['precip_accum_diff'] = df['precip_accum_set_1'] \
                             - df['precip_accum_set_1_2']

# %% 
# # 1.c merge datasets together and clean (datadf) 
# # # 1.c.1 join df to data to make new dataframe (datadf)
datadf = data.join(df[['air_temp_set_1','precip_accum_diff']], rsuffix="_"+stationname)
# # # #  1.c.1.a rename column names
names = ['Q', 'T', 'P']
datadf = datadf.rename(columns={'flow': names[0], 'air_temp_set_1': names[1],
                       'precip_accum_diff': names[2]})
# # # 1.c.2 summarize new data set
print(datadf.head())
print(datadf.describe())

# # # 1.c.3 subset data to avoid null values and wonky precip data
datadf_sub = datadf.loc["2010-01-01":"2020-12-31"]
# # # 1.c.4 replace NaN with zero for precipitation
datadf_sub['P'] = datadf_sub['P'].fillna(0)
# # # 1.c.5 replace < 0 with zero for precipitation
datadf_sub['P'].where((datadf_sub['P']>=0),other=0,inplace=True)
# # # 1.c.6 backfill nan data for temperature and discharge
datadf_sub['Q'].fillna(method='bfill',inplace=True)
datadf_sub['T'].fillna(method='bfill',inplace=True)

# %%
# # # 1.c.x show pretty pictures (long run time)
# # box and whisker plots
# fig1, ax1 = plt.subplots()
# ax1.boxplot(datadf_sub['Q'].iloc[0:100], showfliers=False)
# fig1.show()

# # histograms
# fig2, ax2 = plt.subplots()
# ax2.hist(datadf_sub['Q'])
# fig2.show()

# # scatter plot matrix
# scatter_matrix(datadf_sub)
# plt.show()

# %%
# # 2.a normalize training data [0 to 1]
# # # 2.a.1 set random seed
np.random.seed(7)

# # # 2.a.2 involves MinMaxScaler
# # # more info at: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
# # # # 2.a.2.a definie min max scaler
scaler = MinMaxScaler()

# # # # 2.a.2.b transform training data
# # # # # 2.a.2.b.1 names of predictive variables
array_columns = ['Q'] # list(datadf_sub.columns)
array = datadf_sub['Q'].values

# # # # # 2.a.2.b.2 careful with dimensions of transform
array = scaler.fit_transform(array.reshape(-1,1))

# # # # 2.b.2 for x_train (predictives) use y(t-m) to y(t-1), where y = Q, P, and T
# X = array[12:-6, array_columns.index(X_var)]
# Y = array[12:-6, array_columns.index(Y_var)]


# # # # # (all precip, flow, and temperature in the last m hours)
# # # # 2.b.3 for y_train, use Q(t+n) (predicted)
# # # # 2.b.4 remove null rows (only at beginning and end)

# # # the below code is for multiple input variables
# # X = datadf_sub.iloc[12:-6, (datadf_sub.columns != 'Q') & \
# #                     (datadf_sub.columns != 'P') & \
# #                     (datadf_sub.columns != 'T')].values


# # %%
# # 2. Pre-Processing -> select into test and train periods
# # # 2.b create y(t+X) from streamflow (Q), precip (P), and temperature (T)
# # # # 2.b.1 lag so that y(t-m) to Q(t+n), where y = Q, P, and T 
# for y in names:
#     print(y)
#     for i in range(m):
#         # auto-create new title [title{i}]
#         title1 = y+"{0}".format(i+1)
#         # lag data by 'i' weeks
#         datadf_sub[title1] = datadf_sub[y].shift(i+1)
#     for j in range(n):
#         # auto-create new title [title{-j}]
#         title2 = y+"{0}".format(-(j+1))
#         # push forward data by 'j' weeks
#         datadf_sub[title2] = datadf_sub[y].shift(-(j+1))

# %% 
# # 2.c divide into train/test sets ratio of y/z
train_size = int(X.shape[0]*(y_n/(y_n + z_n)))
test_size = X.shape[0]-train_size
X_train, X_test = X[0:train_size], X[train_size:X.shape[0]] # for multiple X[0:train_size,:], X[train_size:X.shape[0],:]
Y_train, Y_test = Y[0:train_size], Y[train_size:X.shape[0]] # for mulltiple Y[0:train_size,:], Y[train_size:X.shape[0],:]
print(train_size, test_size)

# # the below code doesn't work because with sequential data order is important
# X_train, X_validation, Y_train, Y_validation = \
#     train_test_split(X.values, Y.values, test_size=(z_n/(y_n + z_n)), \
#     random_state=1)

# # 2.d reshape input to be [samples, time steps, features]
# # # look into why we need time steps and features (1) 
X_train = np.reshape(X_train, (X_train.shape[0], 1, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 1, 1))
print(X_train.shape, X_test.shape)

# # 2.e reshape outputs to be [samples, features]
Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))
print(Y_train.shape, Y_test.shape)

# %%
# 3. Evaluate Algorithms

# # 3.a Model training (build different models)

# # # 3.a.1 Use KERAS library for LSTM
# # # more reference: https://keras.io/api/layers/recurrent_layers/lstm/
# # # more reference: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

# # # # 3.a.1.a create and fit the LSTM network
# # # # # LSTM arguments (nodes, optimizer, batch size, epochs)
model = Sequential()
# nodes, but input_shape?
model.add(LSTM(nodes, input_shape=(1, 1)))
# this step?
model.add(Dense(1))
# LSTM arguments (nodes, optimizer, batch size, epochs)
model.compile(loss='mean_squared_error', optimizer=optimizer)
# verbose?
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2)

# %% 
# # # 3.a.2 Use SVR and MLP for Comparison
# # # # Empty for now! 

# # # 3.a.3 Evaluate algorithms on training data
# # # # Add RMSE, MAE, R^2 (from Youchan 2020)
# make predictions
trainPredict = model.predict(X_train)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
Y_train = scaler.inverse_transform([Y_train])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(Y_train[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

# # 3.b Test Model (test/validate models)
# # # 3.b.1 Use SVR and MLP for Comparison
# # # # Empty for Now

# # # 3.b.322 Evaluate algorithms on training data
# # # # Add RMSE, MAE, R^2 (from Youchan 2020)
# make predictions
testPredict = model.predict(X_test)
# invert predictions
testPredict = scaler.inverse_transform(testPredict)
Y_test = scaler.inverse_transform([Y_test])
# calculate root mean squared error
testScore = math.sqrt(mean_squared_error(Y_test[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# %%
# 4. Compare Results (fit on training and test data)
# # 4.a tables of fits
# # 4.b time series comparing models to reality

# %%
# 5. Make predictions 
# # blank for now! 

# %%
# 6. Extended experiments
# # 6.a vary number and types of predictive variables
# # 6.b vary how far can we predict into the future (n)
# # 6.c vary how much 'encoding' (data before) (m)



# %% random experiments with lstm
inputs = tf.random.normal([32, 10, 8])
lstm = tf.keras.layers.LSTM(4)
output = lstm(inputs)
# print(output.shape)
# # (32, 4)
# lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
# whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
# # print(whole_seq_output.shape)
# (32, 10, 4)
# # print(final_memory_state.shape)
# (32, 4)
# # print(final_carry_state.shape)
# (32, 4)
# %%
