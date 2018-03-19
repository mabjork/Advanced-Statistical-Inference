import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

def loadData():
    with open("olympics.csv") as f:
        content = f.readlines()
    data = []
    for line in content:
        columns = line.split(",")
        columns = map(lambda x: x.replace("\n","").replace("'",""),columns)
        row = [int(columns[0]),float(columns[1])]
        data.append(row)

    return np.array(data)

def plotData(data, line = None):
    times = data[:,1]
    years = data[:,0]
    plt.plot(years,times,"b")

    if(line != None):
        plt.plot(years,line,"r")

    plt.show()

def calcRegLine(data):
    times = data[:,1]
    years = data[:,0]

    avg_times = np.average(times)
    avg_times_squared = np.average(map(lambda x : x**2,times))
    avg_years = np.average(years)
    avg_years_squared = np.average(map(lambda x : x**2,years))
    avg_times_years = np.average(map(lambda x: x[0]*x[1] ,zip(years,times)))
    
    slope = (avg_times_years - avg_times * avg_years) / (avg_years_squared - avg_years ** 2)
    intercept = avg_times - slope * avg_years 

    return slope,intercept

def calcAvgLoss(data,slope,intercept):
    return np.average(map(lambda x: (x[1] - intercept - slope * x[0])**2, data))

def leastSquares(x,t):
    dot_prod = x.T.dot(x)
    inverse = np.linalg.inv(dot_prod)
    w_hat = inverse.dot(x.T).dot(t)
    return w_hat

def generateX(x, polynomial):
    X = []
    for i in range(len(x)):
        row = []
        for j in range(polynomial+1):
            row.append(x[i]**j)
        X.append(row)
    return np.array(X)

def makePrediction(x_test,w_hat,k):
    X_test = generateX(x_test,k)
    prediction = X_test.dot(w_hat)
    return prediction

def generateRandomData(polynomial):
    None

def computeError(tt,prediction):
    di = (prediction - tt) ** 2
    return di

def doCrossValidation(data,polynomial):
    x = data[:,0]
    t = data[:,1]
    trainX = generateX(x,polynomial)
    traint = t
    errors = []
    for i in range(len(x)):
        
        #Select the year and time to be used as test data
        testX = trainX[i]
        testt = traint[i]

        #Remove the test data from the training set
        np.delete(trainX,i,axis=0)
        np.delete(traint,i)

        #Calculate w_hat for the spesific training set
        w_hat = leastSquares(trainX,t)

        #Calculate the error in the predicion on the test data
        prediction = testX.dot(w_hat)
        error = computeError(testt,prediction)
        errors.append(error)

        #Add the test data back into the training set
        np.insert(trainX,i,testX)
        np.insert(traint,i,testt)
    
    #Return average of all errors collected using different polynomial
    return np.average(errors)




data = loadData()

years = data[:,0]
times = data[:,1]

X = generateX(years,4)

errors = [doCrossValidation(data,k) for k in range(1,5)]

plt.plot(range(1,5),errors)
plt.show()

#slope,intercept = calcRegLine(data)

#line_data = [intercept + slope * y for y in data[:,0]]

#avg_loss = calcAvgLoss(data,slope,intercept)

#plotData(data,line_data)

#print leastSquares(data)


