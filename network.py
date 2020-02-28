import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, LSTM
from Nikkei import dataset
import variable


np.random.seed(7)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * variable.data_ratio)
test_size = len(dataset) - train_size
train = dataset[0:train_size, :]
test = dataset[train_size:len(dataset), :]


# convert an array of values into a dataset matrix
def create_dataset(dataset, maxlen):
    dataX, dataY = [], []
    for i in range(len(dataset)-maxlen-1):
        a = dataset[i:(i+maxlen), 0]
        dataX.append(a)
        dataY.append(dataset[i + maxlen, 0])
    return np.array(dataX), np.array(dataY)


# reshape into X=t and Y=t+maxlen
maxlen = variable.maxlen
trainX, trainY = create_dataset(train, maxlen)
testX, testY = create_dataset(test, maxlen)

print(trainX[:10, :])
print(trainY[:10])

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX[:10, :])

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, maxlen)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate RMSE( root mean squared error)
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[maxlen:len(trainPredict)+maxlen, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(maxlen*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), color="g", label="Actual")
plt.plot(trainPredictPlot, color="b", label="Train_Prediction")
plt.plot(testPredictPlot, color="m", label="Test_Prediction")

plt.legend()
plt.show()
