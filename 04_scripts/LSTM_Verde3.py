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
# # 5) Messing around with different inputs:
    # Time Sample (Hr, Da, Week, etc..)
    # number of nodes
    # epochs, batch_size, optimizer, etc...
# # 6) Turn into multiple functions
# # 7) Compare with other 'traditional' ML methods
    # Linear Regression, SVR and MLP


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
import time


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

def create_dataset(dataset, look_back=1):
    """convert an array of values 
        into a dataset matrix
        """

    dataX = []
    dataY = [] 
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# %%
# function for regressing log-stream flow (updated 11022020)


def lstm_run(dfra, v_name, nds, epc, b_s, y_in=7, z_in=3, look_b=1, opt='adam'):
    """
    Multi step process for running a LSTM model

    dfra = pandas dataframe containing data to be used
    v_name = the predictive variable (by default, a single variable)
    y_in = proportion of the train (out of 10), default=7
    z_in = proportion of the test (out of 10), default=3
    look_b = for autoregression, number of times to look back, default=1
    nds = number of nodes
    opt = optimizer, default='adam'
    epc = number of epochs
    b_s = batch size, the number of samples to use in model fitting

    returns the rmse of testScore and trainScore, and the history of the run
    """
    # # 2.a normalize data [0 to 1]
    # # # 2.a.1 set random seed
    np.random.seed(7)
    # # # 2.a.2 involves MinMaxScaler
    # # # more info at: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    # # # # 2.a.2.a definie min max scaler
    scaler = MinMaxScaler()
    # # # # 2.a.2.b transform training data
    # # # # # 2.a.2.b.1 names of predictive variables
    array = dfra[v_name].values
    # # # # # 2.a.2.b.2 careful with dimensions of transform
    dataset = scaler.fit_transform(array.reshape(-1,1))

    # # 2.b split into test and training set 
    # # the following modified from: 
    # # https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

    # # # 2.b.1 split into train and test sets
    train_size = int(len(dataset) * (y_in/(y_in + z_in)))
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # # # 2.b.2 reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_b)
    testX, testY = create_dataset(test, look_b)
    # # # 2.b.3 reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # 3. create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(nds, input_shape=(1, look_b)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=opt)

    # 4a. train model and evaluate against training
    history = model.fit(trainX, trainY, epochs=epc, batch_size=b_s, verbose=2)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # 4b. make predictions
    trainPredict = model.predict(trainX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))

    # 5. make predicitons and evaluate against testing 
    # make predictions
    testPredict = model.predict(testX)
    # invert predictions
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # 6. evaluate output and show pretty pictures
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    fig, ax = plt.subplots()
    ax.plot(datadf_sub.index.values, scaler.inverse_transform(dataset))
    ax.plot(datadf_sub.index.values, trainPredictPlot)
    ax.plot(datadf_sub.index.values, testPredictPlot)
    ax.set_yscale('log')
    ax.set_xlabel('date')
    ax.set_ylabel('flow, cfs')
    ax.set_title('Real, Train (orange), and Test (blue) data')
    plt.show()

    # plot zoom train
    fig, ax = plt.subplots()
    ax.plot(datadf_sub.index.values, scaler.inverse_transform(dataset), linewidth=3)
    ax.plot(datadf_sub.index.values, trainPredictPlot, alpha=0.7, color='orange')
    ax.set_yscale('log')
    ax.set_xlabel('date')
    ax.set_ylabel('flow, cfs')
    ax.set_xlim(datadf_sub.index.values[look_back], datadf_sub.index.values[len(trainPredict)+look_back])
    ax.set_title('Real and Train (orange) data')
    plt.show()

    # plot zoom test
    fig, ax = plt.subplots()
    ax.plot(datadf_sub.index.values, scaler.inverse_transform(dataset), linewidth=3)
    ax.plot(datadf_sub.index.values, testPredictPlot, alpha=0.7, color='green')
    ax.set_yscale('log')
    ax.set_xlabel('date')
    ax.set_ylabel('flow, cfs')
    ax.set_xlim(datadf_sub.index.values[len(trainPredict)+(look_back*2)+1],datadf_sub.index.values[len(dataset)-1])
    ax.set_title('Real and Test (green) data')
    plt.show()

    return trainScore, testScore, history

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
# 10. Simple Run: Define Problem and initialize variables

# # 10.a Problem: Forecast Streamflow n hours into the future using data
# # #  from m hours into the past via LSTM

# # 10.b initialize varibles
# # # X var[=all-Q], Y var[=Q]
X_var = "Q1"
Y_var = "Q"
var_name = "Q"
# # # how far to lag via look_back
look_back = 1
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

out1, out2, out3 = lstm_run(datadf_sub, var_name, nodes, epochs, batch_size)

# %%
# 11. Complicated Run: Utilizing different 
# # nodes, batch sizes, epochs
# # for different single variable regression
# # look_back from 1 to 12

var_name = "Q"
dfin = datadf_sub
nodes = 64
optimizer = 'adam'
batch_size = 72
epochs = 10

titleList = ['var_name', 'nodes', 'optimizer', 'epochs', 'batch_size', 'look_back']
inputList = []
trainScoreList = [] 
testScoreList = []
historyList = []
timeList = []

for look_back in range(1,12):
    inputList.append([var_name, nodes, optimizer, epochs, batch_size, look_back])
    start_time = time.time()
    trainSc, testSc, hist = lstm_run(dfin, var_name, nodes, epochs, batch_size, look_b=look_back)
    trainScoreList.append(trainSc)
    testScoreList.append(testSc)
    historyList.append(hist)
    timeList.append(time.time() - start_time)


# Additional Tasks: 
    # 1) do multi-pronged for-loop varying look_back, as well as nodes, optimizer, epochs, etc..
    # 2) return array results to show 'health' of the response
    # 3) Compare with SVR and MLP results



# %%
