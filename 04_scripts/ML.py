# Experimenting with Python ML techniques
# Initiated 10272020


# Outline
# (_1_) A generic tutorial for python and ML @ https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# (_2_) A tutorial on k-fold-cross-validation
# (_3_) From 2020. Youchan Hu, Stream-Flow Forecasting of Small Rivers Based on LSTM


# (_1_)
# %%
# 2.1 import libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# %%
# 2.2 load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# %%
# 3.1 dimensions
print(dataset.shape)
# 3.2 peek
print(dataset.head(20))
# 3.3 summarize
print(dataset.describe())
# 3.4 class distribution
print(dataset.groupby('class').size())


# %%
# 4. Data Visualization
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
# histograms
dataset.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# %%
# 5. Evaluate some algoirthms

# 5.1 Separate out a validation dataset.
# # This selects the training and validation randomly 
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# 5.2 Set-up the test harness to use 10-fold cross validation.
# # Split dataset into 10 parts, train on 9 and test on 1
# # # read more about 'k-fold cross validation'

# 5.3 Build multiple different models to predict species from flower measurements
# # 6 different algorithms
# # Logistic Regression (LR)
# # Linear Discriminant Analysis (LDA)
# # K-Nearest Neighbors (KNN).
# # Classification and Regression Trees (CART).
# # Gaussian Naive Bayes (NB).
# # Support Vector Machines (SVM).
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    # how we stratify our samples
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# 5.4 Select the best model.
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
# %%
# 6. Make Predictions
# 6.1 Make Predictions
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# 6.2 Evaluate Predictions
# Evaluate predictions
print(accuracy_score(Y_validation, predictions)) # accuracy
print(confusion_matrix(Y_validation, predictions)) # indication of errors
print(classification_report(Y_validation, predictions)) # breakdown of class by precision, f1 score and support show excellent results

# (_2_)
# %%
# data
from numpy import array
from sklearn.model_selection import KFold
# data sample
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# prepare cross validation
# # create an instance that splits a dataset into 3 folds, shuffles prior to the split, and uses a value of 1 for the pseudorandom number generator.
kfold = KFold(3, True, 1)
# # split method to return each group of train and test sets
# # # enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (train, test))


# (_3_)
# %%
# CREATED 10/28/2020 11:00 am
# MOVED to (below) at 10/28/2020 11:00 am
# /Users/roberthull/OneDrive/Workspace/Work/03_UA/04_Research/01_github/ParFlow_Hull_Git_b/04_ML 

# Resources: 
# 2020. Youchan Hu, Stream-Flow Forecasting of Small Rivers Based on LSTM
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# 0. Define Problem
# # Forecast Streamflow n hours into the future using data
# # #  from m hours into the past via LSTM
# # initialize varibles
# # # m[=12] (number hrs past), n[=5] (number hrs future) 
# # # y[=7] (number train), z[=3] (number test)
# # # nodes[=64] (number hidden layers)
# # # optimizer[=Adam] (solver for model)
# # # batch_size[=72] (size of run, smaller = slower)
# # # epochs[=30] (times running model, higher = longer)

# 1. Prepare Data
# # Assemble flow, precip, and temperature data (hourly)
# # # flow (hourly) from Verde River USGS
# # # temp, precip (hourly) from mesowest station QV...
# # # merge datasets together


# 2. Pre-Processing -> select into test and train periods
# # 2.a create y(t+X) from streamflow (Q), precip (P), and temperature (T)
# # # 2.a.1 lag so that y(t-m) to Q(t+n), where y = Q, P, and T
# # # 2.a.2 for x_set (predictives) use y(t-m) to y(t-1), where y = Q, P, and T
# # # # (all precip, flow, and temperature in the last m hours)
# # # 2.a.3 for y_set, use Q(t+n) (predicted)
# # # 2.a.4 remove null rows (think about this)

# # 2.b divide into train/test sets ratio of y/z
# # # 2.b.1 train_test_split(X, y, test_size=0.20, random_state=1)

# # 2.c normalize data if it has different dimensions?
# # # 2.c.1 invlves MinMaxScaler??

# 3. Evaluate Algorithms
# # 3.a Model training (build different models)
# # # 3.a.1 Use KERAS library for LSTM
# # # # LSTM arguments (nodes, optimizer, batch size, epochs)

# # # 3.a.2 Use SVR and MLP for Comparison

# # # 3.a.3 Evaluate algorithms on training data
# # # # RMSE, MAE, R^2 (from Youchan 2020)
# # # # Accuracy Score, Confusion Matrix, Classification Report (from tutorial)

# # 3.b Test Model (test/validate models)
# # # 3.b.1 Use KERAS library for LSTM
# # # # LSTM arguments (nodes, optimizer, batch size, epochs)

# # # 3.b.2 Use SVR and MLP for Comparison

# # # 3.b.3 Evaluate algorithms on training data
# # # # RMSE, MAE, R^2 (from Youchan 2020)
# # # # Accuracy Score, Confusion Matr

# 4. Compare Results (fit on training and test data)
# # 4.a tables of fits
# # 4.b time series comparing models to reality

# 5. Make predictions 
# # blank for now! 

# 6. Extended experiments
# # 6.a vary number and types of predictive variables
# # 6.b vary how far can we predict into the future (n)
# # 6.c vary how much 'encoding' (data before) (m)

#%% 
# demo of minmaxscaler
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
print(scaler.fit(data))
# MinMaxScaler()
print(scaler.data_max_)
# [ 1. 18.]
print(scaler.transform(data))
# [[0.   0.  ]
# [0.25 0.25]
# [0.5  0.5 ]
# [1.   1.  ]]
print(scaler.transform([[2, 2]]))
# [[1.5 0. ]]
# %%
