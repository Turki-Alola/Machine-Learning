from pandas import read_csv
import matplotlib.pyplot as plt
dataset = read_csv( "international-airline-passengers.csv" , usecols=[1], engine= "python", skipfooter=3)
plt.plot(dataset)
plt.show()

#Importing the libraries
import numpy
import matplotlib.pyplot as plt 
from pandas import read_csv 
import math
from keras.models import Sequential 
from keras.layers import Dense

# fix random seed for reproducibility
numpy.random.seed(7)





#Load the Dataset:

# load the dataset
dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')


#Convert the data into appropriate arrays for processing:
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


#Split into Train and Test:
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

#Reshape for processing 
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#Create the MLP:

# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu')) #8 is the shape of the hidden layer
model.add(Dense(1))#since the output will be one value
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)




#Evaluate Model Performance:

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

