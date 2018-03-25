# -- coding: utf-8 --
# MIT License
#
# Copyright (c) 2017, 2018 Charles Jekel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function
# import libraris
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import fmin_l_bfgs_b
from pyDOE import lhs


# piecewise linear fit library
class piecewise_lin_fit(object):

    # Initiate the library with the supplied x and y data
    # where y(x). For now x and y should be 1D numpy arrays.
    def __init__(self, x, y, disp_res=False):
        # you must supply the x and y data of which you'll be fitting
        # a continuous piecewise linear model to where y(x)
        # by default pwlf won't print the optimization results
        # initialize results printing with print=distribute
        #
        # Examples:
        # # initlize for x, y data
        # myPWLF = piecewise_lin_fit(x, y)
        #
        # # initliaze for x,y data and print optimization results
        # myPWLF = piecewise_lin_fit(x, y, disp_res=True)
        self.print = disp_res

        # sort the data from least x to max x
        orderArg = np.argsort(x)
        self.xData = x[orderArg]
        self.yData = y[orderArg]

        # calculate the number of data points
        self.nData = len(x)

        # set the first and last break x values to be the min and max of x
        self.break0 = np.min(self.xData)
        self.breakN = np.max(self.xData)

    def fitWithBreaks(self, breaks):
        # define a function which fits the piecewise linear function
        # for specified break point locations
        #
        # The function minimizes the sum of the square of the residuals for the
        #  pair of x,y data points
        #
        # This is a port of 4-May-2004 Nikolai Golovchenko MATLAB code
        # see http://golovchenko.org/docs/ContinuousPiecewiseLinearFit.pdf
        #
        # Alternatively see https://www.mathworks.com/matlabcentral/fileexchange/40913-piecewise-linear-least-square-fit
        #
        # Input:
        # provide the location of the end points of the breaks for each line segment
        #
        # Example: if your x data exists from 0 <= x <= 1 and you want three
        # piecewise linear lines, an acceptable breaks would look like
        # breaks = [0.0, 0.3, 0.6, 1.0]
        #
        # Output:
        # The function returns the sum of the square of the residuals
        #
        # To get the parameters of the fit look for
        # self.parameters
        #
        # remember that the parameters that result are part of the continuous function
        # such that:
        # parameters = f(breaks)
        self.fitBreaks = breaks
        numberOfParameters = len(breaks)
        numberOfSegments = numberOfParameters - 1

        self.numberOfParameters = numberOfParameters
        self.numberOfSegments = numberOfSegments

        sepDataX, sepDataY = self.seperateData(breaks)

        # add the separated data to the object
        self.sep_data_x = sepDataX
        self.sep_data_y = sepDataY

        # compute matrices corresponding to the system of equations
        A = np.zeros([numberOfParameters, numberOfParameters])
        B = np.zeros(numberOfParameters)
        for i in range(0, numberOfParameters):
            if i != 0:
                # first sum
                A[i, i - 1] -= sum((sepDataX[i - 1] - breaks[i - 1]) * (sepDataX[i - 1] - breaks[i])) / ((breaks[i] - breaks[i - 1]) ** 2)
                A[i, i] += sum((sepDataX[i - 1] - breaks[i - 1]) ** 2) / ((breaks[i] - breaks[i - 1]) ** 2)
                B[i] += (sum(sepDataX[i - 1] * sepDataY[i - 1]) - breaks[i - 1] * sum(sepDataY[i - 1])) / (breaks[i] - breaks[i - 1])

            if i != numberOfParameters - 1:
                # second sum
                A[i, i] += sum(((sepDataX[i] - breaks[i + 1]) ** 2)) / ((breaks[i + 1] - breaks[i]) ** 2)
                A[i, i + 1] -= sum((sepDataX[i] - breaks[i]) * (sepDataX[i] - breaks[i + 1])) / ((breaks[i + 1] - breaks[i]) ** 2)
                B[i] += (-sum(sepDataX[i] * sepDataY[i]) + breaks[i + 1] * sum(sepDataY[i])) / (breaks[i + 1] - breaks[i])

        # try to solve the regression problem
        try:
            p = np.linalg.solve(A, B)

            yHat = []
            lineSlopes = []
            for i, j in enumerate(sepDataX):
                m = (p[i + 1] - p[i]) / (breaks[i + 1] - breaks[i])
                lineSlopes.append(m)
                yHat.append(m * (j - breaks[i]) + p[i])
            yHat = np.concatenate(yHat)
            self.slopes = np.array(lineSlopes)

            # calculate the sum of the square of residuals
            e = self.yData - yHat
            SSr = np.dot(e.T, e)

            self.fitParameters = p
        except:
            # on an error, return SSr = np.print_function
            # print('ERROR: You might have a singular Matrix!!!')
            SSr = np.inf
            # this usually happens when A is singular
        return (SSr)

    def seperateData(self, breaks):
        # a function that separates the data based on the breaks

        numberOfParameters = len(breaks)
        numberOfSegments = numberOfParameters - 1

        self.numberOfParameters = numberOfParameters
        self.numberOfSegments = numberOfSegments

        # Separate Data into Segments
        sepDataX = [[] for i in range(self.numberOfSegments)]
        sepDataY = [[] for i in range(self.numberOfSegments)]

        for i in range(0, self.numberOfSegments):
            if i == 0:
                # the first index should always be inclusive
                aTest = self.xData >= breaks[i]
            else:
                # the rest of the indices should be exclusive
                aTest = self.xData > breaks[i]
            dataX = np.extract(aTest, self.xData)
            dataY = np.extract(aTest, self.yData)
            bTest = dataX <= breaks[i + 1]
            dataX = np.extract(bTest, dataX)
            dataY = np.extract(bTest, dataY)
            sepDataX[i] = np.array(dataX)
            sepDataY[i] = np.array(dataY)
        return (sepDataX, sepDataY)

    def seperateDataX(self, breaks, x):
        # a function that separates the data based on the breaks for given x

        numberOfParameters = len(breaks)
        numberOfSegments = numberOfParameters - 1

        self.numberOfParameters = numberOfParameters
        self.numberOfSegments = numberOfSegments

        # Separate Data into Segments
        sepDataX = [[] for i in range(self.numberOfSegments)]

        for i in range(0, self.numberOfSegments):
            if i == 0:
                # the first index should always be inclusive
                aTest = x >= breaks[i]
            else:
                # the rest of the indexes should be exclusive
                aTest = x > breaks[i]
            dataX = np.extract(aTest, x)
            bTest = dataX <= breaks[i + 1]
            dataX = np.extract(bTest, dataX)
            sepDataX[i] = np.array(dataX)
        return (sepDataX)

    def predict(self, x, *args):  # breaks, p):
        # a function that predicts based on the supplied x values
        # you can manfully supply break point and determined p
        # yHat = predict(x)
        # or yHat = predict(x, p, breaks)
        if len(args) == 2:
            p = args[0]
            breaks = args[1]
        else:
            p = self.fitParameters
            breaks = self.fitBreaks

        # separate the data by x on breaks
        sepDataX = self.seperateDataX(breaks, x)

        #  add the separated data to self
        self.sep_predict_data_x = sepDataX

        yHat = []
        lineSlopes = []
        for i, j in enumerate(sepDataX):
            m = (p[i + 1] - p[i]) / (breaks[i + 1] - breaks[i])
            lineSlopes.append(m)
            yHat.append(m * (j - breaks[i]) + p[i])
        yHat = np.concatenate(yHat)
        self.slopes = np.array(lineSlopes)
        return (yHat)

    def fitWithBreaksOpt(self, var):
        # same as self.fitWithBreaks, except this one is tuned to be used with
        # the optimization algorithm
        var = np.sort(var)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break0
        breaks[-1] = self.breakN

        sepDataX, sepDataY = self.seperateData(breaks)

        numberOfParameters = self.numberOfParameters

        # compute matrices corresponding to the system of equations
        A = np.zeros([numberOfParameters, numberOfParameters])
        B = np.zeros(numberOfParameters)
        for i in range(0, numberOfParameters):
            if i != 0:
                # first sum
                A[i, i - 1] -= sum((sepDataX[i - 1] - breaks[i - 1]) * (sepDataX[i - 1] - breaks[i])) / (
                            (breaks[i] - breaks[i - 1]) ** 2)
                A[i, i] += sum((sepDataX[i - 1] - breaks[i - 1]) ** 2) / ((breaks[i] - breaks[i - 1]) ** 2)
                B[i] += (sum(sepDataX[i - 1] * sepDataY[i - 1]) - breaks[i - 1] * sum(sepDataY[i - 1])) / (
                            breaks[i] - breaks[i - 1])

            if i != numberOfParameters - 1:
                # second sum
                A[i, i] += sum(((sepDataX[i] - breaks[i + 1]) ** 2)) / ((breaks[i + 1] - breaks[i]) ** 2)
                A[i, i + 1] -= sum((sepDataX[i] - breaks[i]) * (sepDataX[i] - breaks[i + 1])) / (
                            (breaks[i + 1] - breaks[i]) ** 2)
                B[i] += (-sum(sepDataX[i] * sepDataY[i]) + breaks[i + 1] * sum(sepDataY[i])) / (
                            breaks[i + 1] - breaks[i])

        try:
            p = np.linalg.solve(A, B)

            yHat = []
            lineSlopes = []
            for i, j in enumerate(sepDataX):
                m = (p[i + 1] - p[i]) / (breaks[i + 1] - breaks[i])
                lineSlopes.append(m)
                yHat.append(m * (j - breaks[i]) + p[i])
            yHat = np.concatenate(yHat)
            self.slopes = np.array(lineSlopes)

            # calculate the sum of the square of residuals
            e = self.yData - yHat
            SSr = np.dot(e.T, e)
        except:
            # if there is an error in the above calculation
            # it is likely from A being ill conditioned or indeterminant
            # this will be more efficient than calculating the determinant
            SSr = np.inf
        return (SSr)

    def fit(self, numberOfSegments, **kwargs):
        # a function which uses differential evolution to finds the optimum
        # location of break points for a given number of line segments by
        # minimizing the sum of the square of the errors
        #
        # input:
        # the number of line segments that you want to find
        # the optimum break points for
        # ex:
        # breaks = fit(3)
        #
        # output:
        # returns the break points of the optimal piecewise continua lines

        self.numberOfSegments = int(numberOfSegments)
        self.numberOfParameters = self.numberOfSegments + 1

        # self.fitBreaks = self.numberOfSegments+1

        # calculate the number of variables I have to solve for
        self.nVar = self.numberOfSegments - 1

        # initiate the bounds of the optimization
        bounds = np.zeros([self.nVar, 2])
        bounds[:, 0] = self.break0
        bounds[:, 1] = self.breakN

        if len(kwargs) == 0:
            res = differential_evolution(self.fitWithBreaksOpt, bounds, strategy='best1bin',
                                         maxiter=1000, popsize=50, tol=1e-3, mutation=(0.5, 1),
                                         recombination=0.7, seed=None, callback=None, disp=False,
                                         polish=True, init='latinhypercube', atol=1e-4)
        else:
            res = differential_evolution(self.fitWithBreaksOpt, bounds, **kwargs)
        if self.print == True:
            print(res)

        self.SSr = res.fun

        var = np.sort(res.x)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break0
        breaks[-1] = self.breakN
        self.fitBreaks = breaks
        # assign p
        self.fitWithBreaks(self.fitBreaks)

        return (self.fitBreaks)

    def fitfast(self, numberOfSegments, pop=2, **kwargs):
        # a function which uses multi start LBFGSB optimization to find the
        # location of break points for a given number of line segments by
        # minimizing the sum of the square of the errors.
        #
        # The idea is that we generate n random latin hypercube samples
        # and run LBFGSB optimization on each one. This isn't guaranteed to
        # find the global optimum. It's suppose to be a reasonable compromise
        # between speed and quality of fit. Let me know how it works.
        #
        # Since this is based on random sampling, you might want to run it
        # multiple times and save the best version... The best version will
        # have the lowest self.SSr (sum of square of residuals)
        #
        # There is no guarantee that this will be faster than fit(), however
        # you may find it much faster sometimes.
        #
        # input:
        # the number of line segments that you want to find
        # the optimum break points for
        # ex:
        # breaks = fitfast(3)
        #
        # output:
        # returns the break points of the optimal piecewise continuous lines
        #
        #
        # The default number of multi start optimizations is 2.
        # - Decreasing this number will result in a faster run time.
        # - Increasing this number will improve the likelihood of finding
        #   good results
        # - You can specify the number of starts using the following call
        # - Minimum value of pop is 2
        #
        # # finds 3 piecewise line segments with 30 multi start optimizations
        # breaks = fitfast(3,30)
        pop = int(pop)

        self.numberOfSegments = int(numberOfSegments)
        self.numberOfParameters = self.numberOfSegments + 1

        # calculate the number of variables I have to solve for
        self.nVar = self.numberOfSegments - 1

        # initiate the bounds of the optimization
        bounds = np.zeros([self.nVar, 2])
        bounds[:, 0] = self.break0
        bounds[:, 1] = self.breakN

        # perform latin hypercube sampling
        mypop = lhs(self.nVar, samples=pop, criterion='maximin')
        # scale the sampling to my variable range
        mypop = mypop * (self.breakN - self.break0) + self.break0

        x = np.zeros((pop, self.nVar))
        f = np.zeros(pop)
        d = []

        for i, x0 in enumerate(mypop):
            if len(kwargs) == 0:
                resx, resf, resd = fmin_l_bfgs_b(self.fitWithBreaksOpt, x0,
                                                 fprime=None, args=(), approx_grad=True, bounds=bounds,
                                                 m=10, factr=1e2, pgtol=1e-05, epsilon=1e-08, iprint=-1,
                                                 maxfun=15000, maxiter=15000, disp=None, callback=None)
            else:
                resx, resf, resd = fmin_l_bfgs_b(self.fitWithBreaksOpt, x0,
                                                 fprime=None, approx_grad=True, bounds=bounds, **kwargs)
            x[i, :] = resx
            f[i] = resf
            d.append(resd)
            if self.print == True:
                print(i + 1, 'of ' + str(pop) + ' complete')

        # find the best result
        best_ind = np.nanargmin(f)
        best_val = f[best_ind]
        best_break = x[best_ind]
        res = (x[best_ind], f[best_ind], d[best_ind])
        if self.print == True:
            print(res)

        self.SSr = best_val

        var = np.sort(best_break)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break0
        breaks[-1] = self.breakN
        self.fitBreaks = breaks
        # assign p
        self.fitWithBreaks(self.fitBreaks)

        return (self.fitBreaks)

    def useCustomOpt(self, numberOfSegments):
        # provide the number of line segments you want to use with your
        # custom optimization routine
        #
        # then optimize fitWithBreaksOpt(var) where var is a 1D array
        # containing the x locations of your variables
        # var has length numberOfSegments - 1, because the two break points
        # are always defined (1. the min of x, 2. the max of x)
        #
        # fitWithBreaksOpt(var) will return the sum of the square of the
        # residuals which you'll want to minimize with your optimization
        # routine

        self.numberOfSegments = int(numberOfSegments)
        self.numberOfParameters = self.numberOfSegments + 1

        # self.fitBreaks = self.numberOfSegments+1

        # calculate the number of variables I have to solve for
        self.nVar = self.numberOfSegments - 1

    def fit_break_begin_and_end_opt(self, var):
        # same as fit_fit_break_begin_and_end, but used in optimization

        var = np.sort(var)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break0
        breaks[-1] = self.breakN

        sepDataX, sepDataY = self.seperateData(breaks)

        numberOfParameters = self.nVar

        # compute matrices corresponding to the system of equations
        A = np.zeros([numberOfParameters, numberOfParameters])
        B = np.zeros(numberOfParameters)
        for i in range(0, numberOfParameters):
            if i != 0:
                # first sum
                A[i, i - 1] -= sum((sepDataX[i - 1] - breaks[i - 1]) * (sepDataX[i - 1] - breaks[i])) / (
                            (breaks[i] - breaks[i - 1]) ** 2)
                A[i, i] += sum((sepDataX[i - 1] - breaks[i - 1]) ** 2) / ((breaks[i] - breaks[i - 1]) ** 2)
                B[i] += (sum(sepDataX[i - 1] * sepDataY[i - 1]) - breaks[i - 1] * sum(sepDataY[i - 1])) / (
                            breaks[i] - breaks[i - 1])

            if i != numberOfParameters - 1:
                # second sum
                A[i, i] += sum(((sepDataX[i] - breaks[i + 1]) ** 2)) / ((breaks[i + 1] - breaks[i]) ** 2)
                A[i, i + 1] -= sum((sepDataX[i] - breaks[i]) * (sepDataX[i] - breaks[i + 1])) / (
                            (breaks[i + 1] - breaks[i]) ** 2)
                B[i] += (-sum(sepDataX[i] * sepDataY[i]) + breaks[i + 1] * sum(sepDataY[i])) / (
                            breaks[i + 1] - breaks[i])

        print('A',np.shape(A))
        print('B',np.shape(B))
        # remove top row and column of A, B
        # A = A[1:-1, 1:-1]
        # B = B[1:-1]
        # print('A',np.shape(A))
        # print('B',np.shape(B))
        # p_temp = np.linalg.solve(A, B)
        try:
            p_temp = np.linalg.solve(A, B)
            p = np.zeros(self.numberOfParameters)
            p[0] = self.y0
            p[-1] = self.yn
            p[1:-1] = p_temp
            print('p_temp',np.shape(p_temp))
            print('p', np.shape(p))
            yHat = []
            lineSlopes = []
            for i, j in enumerate(sepDataX):
                m = (p[i + 1] - p[i]) / (breaks[i + 1] - breaks[i])
                lineSlopes.append(m)
                yHat.append(m * (j - breaks[i]) + p[i])
            yHat = np.concatenate(yHat)
            self.slopes = np.array(lineSlopes)

            # calculate the sum of the square of residuals
            e = self.yData - yHat
            SSr = np.dot(e.T, e)
        except:
            # if there is an error in the above calculation
            # it is likely from A being ill conditioned or indeterminant
            # this will be more efficient than calculating the determinant
            SSr = np.inf
        return (SSr)

    def fit_break_begin_and_end(self, var):
        # define a function which fits the piecewise linear function
        # for specified break point locations where the begining and ending have
        # a fixed location
        #
        # The function minimizes the sum of the square of the residuals for the
        #  pair of x,y data points
        #
        # This is a port of 4-May-2004 Nikolai Golovchenko MATLAB code
        # see http://golovchenko.org/docs/ContinuousPiecewiseLinearFit.pdf
        #
        # Alternatively see https://www.mathworks.com/matlabcentral/fileexchange/40913-piecewise-linear-least-square-fit
        #
        # Input:
        # provide the location of the end points of the breaks for each line segment
        #
        # Example: if your x data exists from 0 <= x <= 1 and you want three
        # piecewise linear lines, an acceptable breaks would look like
        # breaks = [0.0, 0.3, 0.6, 1.0]
        #
        # Output:
        # The function returns the sum of the square of the residuals
        #
        # To get the parameters of the fit look for
        # self.parameters
        #
        # remember that the parameters that result are part of the continuous function
        # such that:
        # parameters = f(breaks)

        var = np.sort(var)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break0
        breaks[-1] = self.breakN

        sepDataX, sepDataY = self.seperateData(breaks)

        numberOfParameters = self.nVar

        # compute matrices corresponding to the system of equations
        A = np.zeros([numberOfParameters, numberOfParameters])
        B = np.zeros(numberOfParameters)
        for i in range(0, numberOfParameters):
            if i != 0:
                # first sum
                A[i, i - 1] -= sum((sepDataX[i - 1] - breaks[i - 1]) * (sepDataX[i - 1] - breaks[i])) / (
                            (breaks[i] - breaks[i - 1]) ** 2)
                A[i, i] += sum((sepDataX[i - 1] - breaks[i - 1]) ** 2) / ((breaks[i] - breaks[i - 1]) ** 2)
                B[i] += (sum(sepDataX[i - 1] * sepDataY[i - 1]) - breaks[i - 1] * sum(sepDataY[i - 1])) / (
                            breaks[i] - breaks[i - 1])

            if i != numberOfParameters - 1:
                # second sum
                A[i, i] += sum(((sepDataX[i] - breaks[i + 1]) ** 2)) / ((breaks[i + 1] - breaks[i]) ** 2)
                A[i, i + 1] -= sum((sepDataX[i] - breaks[i]) * (sepDataX[i] - breaks[i + 1])) / (
                            (breaks[i + 1] - breaks[i]) ** 2)
                B[i] += (-sum(sepDataX[i] * sepDataY[i]) + breaks[i + 1] * sum(sepDataY[i])) / (
                            breaks[i + 1] - breaks[i])

        print('A',np.shape(A))
        print('B',np.shape(B))
        # remove top row and column of A, B
        # A = A[1:-1, 1:-1]
        # B = B[1:-1]
        # print('A',np.shape(A))
        # print('B',np.shape(B))
        # p_temp = np.linalg.solve(A, B)
        try:
            p_temp = np.linalg.solve(A, B)
            print(p_temp)
            p = np.zeros(self.numberOfParameters)
            p[0] = self.y0
            p[-1] = self.yn
            p[1:-1] = p_temp
            print('p_temp',np.shape(p_temp))
            print('p', np.shape(p))
            print(p)
            yHat = []
            lineSlopes = []
            for i, j in enumerate(sepDataX):
                m = (p[i + 1] - p[i]) / (breaks[i + 1] - breaks[i])
                lineSlopes.append(m)
                yHat.append(m * (j - breaks[i]) + p[i])
            yHat = np.concatenate(yHat)
            self.slopes = np.array(lineSlopes)

            # calculate the sum of the square of residuals
            e = self.yData - yHat
            SSr = np.dot(e.T, e)

            # save the fitParameters
            self.fitParameters = p

        except:
            # if there is an error in the above calculation
            # it is likely from A being ill conditioned or indeterminant
            # this will be more efficient than calculating the determinant
            SSr = np.inf
        return (SSr)

    def fit_begin_and_end(self, numberOfSegments, y0=None, yn=None, **kwargs):
        #   a function which uses differntial evolution to finds the optimum
        #   location of break points for a given number of line segments by
        #   minimizing the sum of the square of the errors
        #
        #   The difference between fit() and fit_begin_and_end() is that
        #   fit_begin_and_end() forces the beginging and end of the continous
        #   pwlf to occur at at particlar locations
        #
        #   input:
        #   the number of line segments that you want to find
        #   the optimum break points for
        #   numberOfSegments - integer number of desired line segments
        #   y0 - desired floating point value of y(min(x))
        #      - defaults to y[0]
        #   yn - desired floating point value of y(max(x))
        #      - defaults to y[-1]
        #   ex: fit 3 line segments where at y @ min(x) = 0.0 AND
        #   y @ max(x) = 1.5
        #   breaks = fit(3,0.0,1.5)
        #
        #   output:
        #   returns the break points of the optimal piecewise contionus lines
        print('This function does not work! DO NOT USE!)
        self.numberOfSegments = int(numberOfSegments)
        self.numberOfParameters = self.numberOfSegments + 1

        #   define values for y0 and yn
        if y0 == None:
            self.y0 = self.yData[0]
        else:
            self.y0 = y0
        if yn == None:
            self.yn = self.yData[-1]
        else:
            self.yn = yn

        # self.fitBreaks = self.numberOfSegments+1

        # calculate the number of variables I have to solve for
        self.nVar = self.numberOfSegments - 1

        # initiate the bounds of the optimization
        bounds = np.zeros([self.nVar, 2])
        bounds[:, 0] = self.break0
        bounds[:, 1] = self.breakN

        if len(kwargs) == 0:
            res = differential_evolution(self.fit_break_begin_and_end_opt,
                    bounds, strategy='best1bin', maxiter=1000, popsize=50,
                    tol=1e-3, mutation=(0.5, 1), recombination=0.7, seed=None,
                    callback=None, disp=False, polish=True,
                    init='latinhypercube', atol=1e-4)
        else:
            res = differential_evolution(self.fit_break_begin_and_end_opt,
                    bounds, **kwargs)
        if self.print == True:
            print(res)

        self.SSr = res.fun

        var = np.sort(res.x)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break0
        breaks[-1] = self.breakN
        self.fitBreaks = breaks
        # assign p
        self.fit_break_begin_and_end(var)
        print('This function does not work! DO NOT USE!)

        return (self.fitBreaks)
