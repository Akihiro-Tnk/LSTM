import pandas

dataframe = pandas.read_csv('TimeSeries_DataPoints.csv', usecols=[2], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
