# -- coding: utf-8 -- 
#   import libraris
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

#   piecewise linerar fit library
class piecewise_lin_fit:
    
    #   Initiate the libary with the supplied x and y data
    #   where y(x). For now x and y should be 1D numpy arrays.
    def __init__(self, x, y):
        #   you must supply the x and y data of which you'll be fitting 
        #   a continous piecewise linear model to where y(x)
        
        #   sort the data from least x to max x
        orderArg = np.argsort(x)
        self.xData = x[orderArg]
        self.yData = y[orderArg]
        
        #   calculate the number of data points
        self.nData = len(x)
    
        
    def fitWithBreaks(self, breaks):
    #   define a function which fits the piecewise linear function
    #   for specified break point locations
    #
    #   The function minimizes the sum of the square of the residuals for the
    #    pair of x,y data points   
    #   
    #   This is a port of 4-May-2004 Nikolai Golovchenko MATLAB code
    #   see http://golovchenko.org/docs/ContinuousPiecewiseLinearFit.pdf
    #   
    #   Alternatively see https://www.mathworks.com/matlabcentral/fileexchange/40913-piecewise-linear-least-square-fit
    #   
    #   Input:   
    #   provide the location of the end points of the breaks for each line segment
    #   
    #   Example: if your x data exists from 0 <= x <= 1 and you want three
    #   piecewise linear lines, an accpetable breaks would look like
    #   breaks = [0.0, 0.3, 0.6, 1.0]
    #   
    #   Ouput:
    #   The function returns the sum of the square of the residuals
    #   
    #   To get the parameters of the fit look for 
    #   self.paramters
    #
    #   remember that the parameters that result are part of the continous function
    #   such that:
    #   parameters = f(breaks)
        self.fitBreaks = breaks
        numberOfParameters = len(breaks)
        numberOfSegments = numberOfParameters - 1
        
        self.numberOfParameters = numberOfParameters
        self.numberOfSegments = numberOfSegments
        
        ##   Seperate Data into Segments
        #sepDataX = [[] for i in range(numberOfSegments)]
        #sepDataY = [[] for i in range(numberOfSegments)]
        #
        #for i in range(0, numberOfSegments):
        #    dataX = []
        #    dataY = []
        #    for j in range(0,self.nData):
        #        if self.xData[j] >= breaks[i]:
        #            if self.xData[j] <= breaks[i+1]:
        #                dataX.append(self.xData[j])
        #                dataY.append(self.yData[j])
        #    sepDataX[i] = np.array(dataX)
        #    sepDataY[i] = np.array(dataY)   
        sepDataX, sepDataY = self.seperateData(breaks)
        
        #   compute matricies corresponding to the system of equations
        A = np.zeros([numberOfParameters, numberOfParameters])
        B = np.zeros(numberOfParameters)
        for i in range(0,numberOfParameters):
            if i != 0:
                #   first sum
                A[i,i-1] = A[i,i-1] - sum((sepDataX[i-1] - breaks[i-1]) * (sepDataX[i-1] - breaks[i])) / ((breaks[i] - breaks[i-1]) ** 2)
                A[i,i] = A[i,i] + sum((sepDataX[i-1] - breaks[i-1]) ** 2) / ((breaks[i] - breaks[i-1]) ** 2)
                B[i] = B[i] + (sum(sepDataX[i-1] * sepDataY[i-1]) - breaks[i-1] * sum(sepDataY[i-1])) / (breaks[i] - breaks[i-1])
        
            if i != numberOfParameters - 1:
                #   second sum
                A[i,i] = A[i,i] + sum(((sepDataX[i] - breaks[i+1]) ** 2)) / ((breaks[i+1] - breaks[i]) ** 2)
                A[i,i+1] = A[i,i+1] - sum((sepDataX[i] - breaks[i]) * (sepDataX[i] - breaks[i+1])) / ((breaks[i+1] - breaks[i]) ** 2)
                B[i] = B[i] + (-sum(sepDataX[i] * sepDataY[i]) + breaks[i+1] * sum(sepDataY[i])) / (breaks[i+1] - breaks[i])
        
        p = np.linalg.solve(A,B)

        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #plt.plot(self.xData,self.yData,'ok',label='Test 1')
        #plt.plot(sepDataX[0],sepDataY[0],'ob',label='Test 1')
        #plt.plot(sepDataX[1],sepDataY[1],'or',label='Test 1')
        #plt.plot(sepDataX[2],sepDataY[2],'oy',label='Test 1')
        #plt.plot(breaks[0:2],p[0:2],'-b')
        #plt.plot(breaks[1:3],p[1:3],'-r')
        #plt.plot(breaks[2:4],p[2:4],'-y')
        #plt.show()

        yHat = []        
        for i,j in enumerate(sepDataX):
            m = (p[i+1] - p[i])/(breaks[i+1]-breaks[i])
            yHat.append(m*(j-breaks[i]) + p[i])
        yHat = np.concatenate(yHat)
        
        #   calculate the sum of the square of residuals
        e = self.yData-yHat
        SSr = np.dot(e.T,e)


        self.fitParameters = p

        return SSr
    
    def seperateData(self, breaks):
    #   a function that seperates the data based on the breaks
        
        numberOfParameters = len(breaks)
        numberOfSegments = numberOfParameters - 1
        
        self.numberOfParameters = numberOfParameters
        self.numberOfSegments = numberOfSegments
        
        #   Seperate Data into Segments
        sepDataX = [[] for i in range(self.numberOfSegments)]
        sepDataY = [[] for i in range(self.numberOfSegments)]
        
        for i in range(0, self.numberOfSegments):
            dataX = []
            dataY = []
            aTest = self.xData >= breaks[i]
            dataX = np.extract(aTest, self.xData)
            dataY = np.extract(aTest, self.yData)
            bTest = dataX <= breaks[i+1]
            dataX = np.extract(bTest, dataX)
            dataY = np.extract(bTest, dataY)
            sepDataX[i] = np.array(dataX)
            sepDataY[i] = np.array(dataY)  
        return sepDataX, sepDataY
    
    def seperateDataX(self, breaks, x):
    #   a function that seperates the data based on the breaks for given x
        
        numberOfParameters = len(breaks)
        numberOfSegments = numberOfParameters - 1
        
        self.numberOfParameters = numberOfParameters
        self.numberOfSegments = numberOfSegments
        
        #   Seperate Data into Segments
        sepDataX = [[] for i in range(self.numberOfSegments)]
        
        for i in range(0, self.numberOfSegments):
            dataX = []
            aTest = x >= breaks[i]
            dataX = np.extract(aTest, x)
            bTest = dataX <= breaks[i+1]
            dataX = np.extract(bTest, dataX)
            sepDataX[i] = np.array(dataX)
        return sepDataX
         
    def predict(self, x, *args):#breaks, p):
        #   a function that predicts based on the supplied x values
        #    you can manully supply break point and determined p
        #   yHat = predict(x)
        #   or yHat = predict(x, p, breaks)
        if len(args) == 2:
            p = args[0]
            breaks = args[1]
        else:
            p = self.fitParameters
            breaks = self.fitBreaks
        #if p is None:
        #    p = self.fitParameters
        #if breaks is None:
        #   breaks = self.fitBreaks
        
        #   seperate the data by x on breaks
        sepDataX = self.seperateDataX(breaks,x)
        yHat = []
        for i,j in enumerate(sepDataX):
            m = (p[i+1] - p[i])/(breaks[i+1]-breaks[i])
            yHat.append(m*(j-breaks[i]) + p[i])
        yHat = np.concatenate(yHat)
        return yHat
    
    def fit(self, numberOfBreakPoints):
        #   a function which uses differntial evolution to finds the optimum
        #   location of break points for a given numberOfBreakPoints by minimizing
        #   the sum of the square of the errors
        
        res = differential_evolution(self.fitWithBreaks, bounds, strategy='best1bin',
                maxiter=1000, popsize=500, tol=0.000001, mutation=(0.5, 1), 
                recombination=0.7, seed=None, callback=None, disp=False, 
                polish=True, init='latinhypercube', atol=0)
        

            
        
    
    