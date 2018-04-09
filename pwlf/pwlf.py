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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function
# import libraries
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import fmin_l_bfgs_b
from pyDOE import lhs

# piecewise linear fit library


class PiecewiseLinFit(object):

    def __init__(self, x, y, disp_res=False):
        # Initiate the library with the supplied x and y data
        # where y(x). For now x and y should be 1D numpy arrays.
        #
        # you must supply the x and y data of which you'll be fitting
        # a continuous piecewise linear model to where y(x)
        # by default pwlf won't print the optimization results
        # initialize results printing with print=distribute
        #
        # Examples:
        # # initialize for x, y data
        # my_pwlf = PiecewiseLinFit(x, y)
        #
        # # initialize for x,y data and print optimization results
        # my_pWLF = PiecewiseLinFit(x, y, disp_res=True)
        self.print = disp_res

        # x and y should be numpy arrays
        # if they are not convert to numpy array
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        if isinstance(y, np.ndarray) is False:
            y = np.array(y)

        # sort the data from least x to max x
        order_arg = np.argsort(x)
        self.x_data = x[order_arg]
        self.y_data = y[order_arg]

        # calculate the number of data points
        self.n_data = len(x)

        # set the first and last break x values to be the min and max of x
        self.break_0 = np.min(self.x_data)
        self.break_n = np.max(self.x_data)

    def fit_with_breaks(self, breaks):
        # define a function which fits the piecewise linear function
        # for specified break point locations
        #
        # The function minimizes the sum of the square of the residuals for the
        #  pair of x,y data points
        #
        # If you want to understand the math behind this read
        # http://jekel.me/2018/Continous-piecewise-linear-regression/
        #
        # Other useful resources:
        # http://golovchenko.org/docs/ContinuousPiecewiseLinearFit.pdf
        # https://www.mathworks.com/matlabcentral/fileexchange/40913-piecewise-linear-least-square-fit
        # http://www.regressionist.com/2018/02/07/continuous-piecewise-linear-fitting/
        #
        # Input:
        # provide the x locations of the end points for the breaks of each
        # line segment
        #
        # Example: if your x data exists from 0 <= x <= 1 and you want three
        # piecewise linear lines, an acceptable breaks would look like
        # breaks = [0.0, 0.3, 0.6, 1.0]
        # ssr = fit_with_breaks(breaks)
        #
        # Output:
        # The function returns the sum of the square of the residuals
        #
        # To get the beta values of the fit look for
        # self.beta
        # or to get the slope values of the lines loot for
        # self.slopes

        # Check if breaks in ndarray, if not convert to np.array
        if isinstance(breaks, np.ndarray) is False:
            breaks = np.array(breaks)

        # Sort the breaks, then store them
        breaks_order = np.argsort(breaks)
        self.fit_breaks = breaks[breaks_order]
        # store the number of parameters and line segments
        self.n_parameters = len(breaks)
        self.n_segments = self.n_parameters - 1

        # initialize the regression matrix as zeros
        A = np.zeros((self.n_data, self.n_parameters))
        # The first two columns of the matrix are always defined as
        A[:, 0] = 1.0
        A[:, 1] = self.x_data - self.fit_breaks[0]
        # Loop through the rest of A to determine the other columns
        for i in range(self.n_segments-1):
            # find the first index of x where it is greater than the break
            # point value
            int_index = np.argmax(self.x_data > self.fit_breaks[i+1])
            # only change the non-zero values of A
            A[int_index:, i+2] = self.x_data[int_index:] - self.fit_breaks[i+1]

        # try to solve the regression problem
        try:
            # least squares solver
            beta, ssr, rank, s = np.linalg.lstsq(A, self.y_data, rcond=None)
            # save the beta parameters
            self.beta = beta

            # save the slopes
            self.slopes = beta[1:]

            # ssr is only calculated if self.n_data > self.n_parameters
            # in this case I'll need to calculate ssr manually
            # where ssr = sum of square of residuals
            if self.n_data <= self.n_parameters:
                y_hat = np.dot(A, beta)
                e = y_hat - self.y_data
                ssr = np.dot(e, e)

            # if ssr still hasn't been calculated... Then try again
            if ssr.size == 0:
                y_hat = np.dot(A, beta)
                e = y_hat - self.y_data
                ssr = np.dot(e, e)

        except np.linalg.LinAlgError:
            # the computation could not converge!
            # on an error, return ssr = np.print_function
            # You might have a singular Matrix!!!
            ssr = np.inf
        if ssr is None:
            ssr = np.inf
            # something went wrong...
        return ssr

    def predict(self, x, *args):  # breaks, p):
        # a function that predicts based on the supplied x values
        # you can manfully supply break points and calculated
        # values for beta
        #
        # Examples:
        # y_hat = predict(x)
        # # or
        # y_hat = predict(x, beta, breaks)
        if len(args) == 2:
            self.beta = args[0]
            breaks = args[1]
            # Sort the breaks, then store them
            breaks_order = np.argsort(breaks)
            self.fit_breaks = breaks[breaks_order]
        # check if x is numpy array, if not convert to numpy array
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        # sort the data from least x to max x
        order_arg = np.argsort(x)
        x = x[order_arg]
        # initialize the regression matrix as zeros
        A = np.zeros((len(x), self.n_parameters))
        # The first two columns of the matrix are always defined as
        A[:, 0] = 1.0
        A[:, 1] = x - self.fit_breaks[0]
        # Loop through the rest of A to determine the other columns
        for i in range(self.n_segments-1):
            # find the first index of x where it is greater than the break
            # point value
            int_index = np.argmax(x > self.fit_breaks[i+1])
            # only change the non-zero values of A
            A[int_index:, i+2] = x[int_index:] - self.fit_breaks[i+1]

        # solve the regression problem
        y_hat = np.dot(A, self.beta)
        return y_hat

    def fit_with_breaks_opt(self, var):
        # same as self.fitWithBreaks, except this one is tuned to be used with
        # the optimization algorithm.
        #
        # Note: unlike fit_with_breaks, fit_with_breaks_opt automatically
        # assumes that the firs and last break points occur at the min and max
        # values of x

        var = np.sort(var)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break_0
        breaks[-1] = self.break_n

        # Sort the breaks, then store them
        breaks_order = np.argsort(breaks)
        breaks = breaks[breaks_order]

        # initialize the regression matrix as zeros
        A = np.zeros((self.n_data, self.n_parameters))
        # The first two columns of the matrix are always defined as
        A[:, 0] = 1.0
        A[:, 1] = self.x_data - breaks[0]
        # Loop through the rest of A to determine the other columns
        for i in range(self.n_segments-1):
            # find the first index of x where it is greater than the break
            # point value
            int_index = np.argmax(self.x_data > breaks[i+1])
            # only change the non-zero values of A
            A[int_index:, i+2] = self.x_data[int_index:] - breaks[i+1]

        # try to solve the regression problem
        try:
            # least squares solver
            beta, ssr, rank, s = np.linalg.lstsq(A, self.y_data, rcond=None)

            # save the beta parameters
            self.beta = beta

            # save the slopes
            self.slopes = beta[1:]

            # ssr is only calculated if self.n_data > self.n_parameters
            # in all other cases I'll need to calculate ssr manually
            # where ssr = sum of square of residuals
            if self.n_data <= self.n_parameters:
                y_hat = np.dot(A, beta)
                e = y_hat - self.y_data
                ssr = np.dot(e, e)

            # if ssr still hasn't been calculated... Then try again
            if ssr.size == 0:
                y_hat = np.dot(A, beta)
                e = y_hat - self.y_data
                ssr = np.dot(e, e)

        except np.linalg.LinAlgError:
            # the computation could not converge!
            # on an error, return ssr = np.print_function
            # You might have a singular Matrix!!!
            ssr = np.inf
        if ssr is None:
            ssr = np.inf
            # something went wrong...
        return ssr

    def fit(self, n_segments, **kwargs):
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

        self.n_segments = int(n_segments)
        self.n_parameters = self.n_segments + 1

        # calculate the number of variables I have to solve for
        self.nVar = self.n_segments - 1

        # initiate the bounds of the optimization
        bounds = np.zeros([self.nVar, 2])
        bounds[:, 0] = self.break_0
        bounds[:, 1] = self.break_n

        if len(kwargs) == 0:
            res = differential_evolution(self.fit_with_breaks_opt, bounds,
                                         strategy='best1bin', maxiter=1000,
                                         popsize=50, tol=1e-3,
                                         mutation=(0.5, 1), recombination=0.7,
                                         seed=None, callback=None, disp=False,
                                         polish=True, init='latinhypercube',
                                         atol=1e-4)
        else:
            res = differential_evolution(self.fit_with_breaks_opt,
                                         bounds, **kwargs)
        if self.print is True:
            print(res)

        self.ssr = res.fun

        # pull the breaks out of the result
        var = np.sort(res.x)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break_0
        breaks[-1] = self.break_n

        # assign values
        self.fit_with_breaks(breaks)

        return self.fit_breaks

    def fitfast(self, n_segments, pop=2, **kwargs):
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
        # have the lowest self.ssr (sum of square of residuals)
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
        # Examples:
        #
        # # finds 3 piecewise line segments with 30 multi start optimizations
        # breaks = fitfast(3,30)
        #
        # # finds 7 piecewise line segments with 50 multi start optimizations
        # breaks = fitfast(7,50)
        pop = int(pop)  # ensure that the population is interger

        self.n_segments = int(n_segments)
        self.n_parameters = self.n_segments + 1

        # calculate the number of variables I have to solve for
        self.nVar = self.n_segments - 1

        # initiate the bounds of the optimization
        bounds = np.zeros([self.nVar, 2])
        bounds[:, 0] = self.break_0
        bounds[:, 1] = self.break_n

        # perform latin hypercube sampling
        mypop = lhs(self.nVar, samples=pop, criterion='maximin')
        # scale the sampling to my variable range
        mypop = mypop * (self.break_n - self.break_0) + self.break_0

        x = np.zeros((pop, self.nVar))
        f = np.zeros(pop)
        d = []

        for i, x0 in enumerate(mypop):
            if len(kwargs) == 0:
                resx, resf, resd = fmin_l_bfgs_b(self.fit_with_breaks_opt, x0,
                                                 fprime=None, args=(),
                                                 approx_grad=True,
                                                 bounds=bounds, m=10,
                                                 factr=1e2, pgtol=1e-05,
                                                 epsilon=1e-08, iprint=-1,
                                                 maxfun=15000, maxiter=15000,
                                                 disp=None, callback=None)
            else:
                resx, resf, resd = fmin_l_bfgs_b(self.fit_with_breaks_opt, x0,
                                                 fprime=None, approx_grad=True,
                                                 bounds=bounds, **kwargs)
            x[i, :] = resx
            f[i] = resf
            d.append(resd)
            if self.print is True:
                print(i + 1, 'of ' + str(pop) + ' complete')

        # find the best result
        best_ind = np.nanargmin(f)
        best_val = f[best_ind]
        best_break = x[best_ind]
        res = (x[best_ind], f[best_ind], d[best_ind])
        if self.print is True:
            print(res)

        self.ssr = best_val

        # obtain the break point locations from the best result
        var = np.sort(best_break)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break_0
        breaks[-1] = self.break_n

        # assign parameters
        self.fit_with_breaks(breaks)

        return self.fit_breaks

    def use_custom_opt(self, n_segments):
        # provide the number of line segments you want to use with your
        # custom optimization routine
        #
        # then optimize fit_with_breaks_opt(var) where var is a 1D array
        # containing the x locations of your variables
        # var has length n_segments - 1, because the two break points
        # are always defined (1. the min of x, 2. the max of x)
        #
        # fit_with_breaks_opt(var) will return the sum of the square of the
        # residuals which you'll want to minimize with your optimization
        # routine
        #
        # run this function to initialize necessary attributes

        self.n_segments = int(n_segments)
        self.n_parameters = self.n_segments + 1

        # calculate the number of variables I have to solve for
        self.nVar = self.n_segments - 1


# OLD piecewise linear fit library naming convention
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
        # # initialize for x, y data
        # myPWLF = piecewise_lin_fit(x, y)
        #
        # # initialize for x,y data and print optimization results
        # myPWLF = piecewise_lin_fit(x, y, disp_res=True)
        warning = '''
                Warning: This is the old piecewise_lin_fit() class which is
                no longer going to be updated. It uses an old naming convention
                that does not follow pep8. Use PiecewiseLinFit() instead.
                '''
        print(warning)
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
        # provide the x locations of the end points for the breaks of each
        # line segment
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
        # remember that the parameters that result are part of the continuous
        # piecewise linear function
        # such that:
        # parameters = f(breaks)

        # Check if breaks in ndarray, if not convert to np.array
        if isinstance(breaks, np.ndarray):
            pass
        else:
            breaks = np.array(breaks)
        # Sort the breaks, then store them
        breaks_order = np.argsort(breaks)
        self.fitBreaks = breaks[breaks_order]
        # store the number of parameters and line segments
        self.numberOfParameters = len(breaks)
        self.numberOfSegments = self.numberOfParameters - 1

        # initialize the regression matrix as zeros
        A = np.zeros((self.nData, self.numberOfParameters))
        # The first two columns of the matrix are always defined as
        A[:, 0] = 1.0
        A[:, 1] = self.xData - self.fitBreaks[0]
        # Loop through the rest of A to determine the other columns
        for i in range(self.numberOfSegments-1):
            # find the first index of x where it is greater than the break
            # point value
            int_index = np.argmax(self.xData > self.fitBreaks[i+1])
            # only change the non-zero values of A
            A[int_index:, i+2] = self.xData[int_index:] - self.fitBreaks[i+1]

        # try to solve the regression problem
        try:
            # least squares solver
            beta, SSr, rank, s = np.linalg.lstsq(A, self.yData, rcond=None)
            # save the beta parameters
            self.beta = beta

            # save the slopes
            self.slopes = beta[1:]

            # SSr is only calculated if self.nData > self.numberOfParameters
            # in this case I'll need to calculate SSr manually
            # where SSr = sum of square of residuals
            if self.nData <= self.numberOfParameters:
                yHat = np.dot(A, beta)
                e = yHat - self.yData
                SSr = np.dot(e, e)

            # if SSr still hasn't been calculated... Then try again
            if SSr.size == 0:
                yHat = np.dot(A, beta)
                e = yHat - self.yData
                SSr = np.dot(e, e)

        except np.linalg.LinAlgError:
            # the computation could not converge!
            # on an error, return SSr = np.print_function
            # You might have a singular Matrix!!!
            SSr = np.inf
        if SSr is None:
            SSr = np.inf
            # something went wrong...
        return SSr

    def predict(self, x, *args):  # breaks, p):
        # a function that predicts based on the supplied x values
        # you can manfully supply break points and calculated
        # values for beta
        #
        # Examples:
        # yHat = predict(x)
        # # or
        # yHat = predict(x, beta, breaks)
        if len(args) == 2:
            self.beta = args[0]
            breaks = args[1]
            # Sort the breaks, then store them
            breaks_order = np.argsort(breaks)
            self.fitBreaks = breaks[breaks_order]
        # sort the data from least x to max x
        orderArg = np.argsort(x)
        x = x[orderArg]
        # initialize the regression matrix as zeros
        A = np.zeros((len(x), self.numberOfParameters))
        # The first two columns of the matrix are always defined as
        A[:, 0] = 1.0
        A[:, 1] = x - self.fitBreaks[0]
        # Loop through the rest of A to determine the other columns
        for i in range(self.numberOfSegments-1):
            # find the first index of x where it is greater than the break
            # point value
            int_index = np.argmax(x > self.fitBreaks[i+1])
            # only change the non-zero values of A
            A[int_index:, i+2] = x[int_index:] - self.fitBreaks[i+1]

        # solve the regression problem
        yHat = np.dot(A, self.beta)
        return yHat

    def fitWithBreaksOpt(self, var):
        # same as self.fitWithBreaks, except this one is tuned to be used with
        # the optimization algorithm

        var = np.sort(var)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break0
        breaks[-1] = self.breakN

        # Sort the breaks, then store them
        breaks_order = np.argsort(breaks)
        breaks = breaks[breaks_order]

        # initialize the regression matrix as zeros
        A = np.zeros((self.nData, self.numberOfParameters))
        # The first two columns of the matrix are always defined as
        A[:, 0] = 1.0
        A[:, 1] = self.xData - breaks[0]
        # Loop through the rest of A to determine the other columns
        for i in range(self.numberOfSegments-1):
            # find the first index of x where it is greater than the break
            # point value
            int_index = np.argmax(self.xData > breaks[i+1])
            # only change the non-zero values of A
            A[int_index:, i+2] = self.xData[int_index:] - breaks[i+1]

        # try to solve the regression problem
        try:
            # least squares solver
            beta, SSr, rank, s = np.linalg.lstsq(A, self.yData, rcond=None)

            # save the beta parameters
            self.beta = beta

            # save the slopes
            self.slopes = beta[1:]

            # SSr is only calculated if self.nData > self.numberOfParameters
            # in all other cases I'll need to calculate SSr manually
            # where SSr = sum of square of residuals
            if self.nData <= self.numberOfParameters:
                yHat = np.dot(A, beta)
                e = yHat - self.yData
                SSr = np.dot(e, e)

            # if SSr still hasn't been calculated... Then try again
            if SSr.size == 0:
                yHat = np.dot(A, beta)
                e = yHat - self.yData
                SSr = np.dot(e, e)

        except np.linalg.LinAlgError:
            # the computation could not converge!
            # on an error, return SSr = np.print_function
            # You might have a singular Matrix!!!
            SSr = np.inf
        if SSr is None:
            SSr = np.inf
            # something went wrong...
        return SSr

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

        # calculate the number of variables I have to solve for
        self.nVar = self.numberOfSegments - 1

        # initiate the bounds of the optimization
        bounds = np.zeros([self.nVar, 2])
        bounds[:, 0] = self.break0
        bounds[:, 1] = self.breakN

        if len(kwargs) == 0:
            res = differential_evolution(self.fitWithBreaksOpt, bounds,
                                         strategy='best1bin', maxiter=1000,
                                         popsize=50, tol=1e-3,
                                         mutation=(0.5, 1), recombination=0.7,
                                         seed=None, callback=None, disp=False,
                                         polish=True, init='latinhypercube',
                                         atol=1e-4)
        else:
            res = differential_evolution(self.fitWithBreaksOpt,
                                         bounds, **kwargs)
        if self.print is True:
            print(res)

        self.SSr = res.fun

        # pull the breaks out of the result
        var = np.sort(res.x)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break0
        breaks[-1] = self.breakN

        # assign values
        self.fitWithBreaks(breaks)

        return self.fitBreaks

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
        # Examples:
        #
        # # finds 3 piecewise line segments with 30 multi start optimizations
        # breaks = fitfast(3,30)
        #
        # # finds 7 piecewise line segments with 50 multi start optimizations
        # breaks = fitfast(7,50)
        pop = int(pop)  # ensure that the population is interger

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
                                                 fprime=None, args=(),
                                                 approx_grad=True,
                                                 bounds=bounds, m=10,
                                                 factr=1e2, pgtol=1e-05,
                                                 epsilon=1e-08, iprint=-1,
                                                 maxfun=15000, maxiter=15000,
                                                 disp=None, callback=None)
            else:
                resx, resf, resd = fmin_l_bfgs_b(self.fitWithBreaksOpt, x0,
                                                 fprime=None, approx_grad=True,
                                                 bounds=bounds, **kwargs)
            x[i, :] = resx
            f[i] = resf
            d.append(resd)
            if self.print is True:
                print(i + 1, 'of ' + str(pop) + ' complete')

        # find the best result
        best_ind = np.nanargmin(f)
        best_val = f[best_ind]
        best_break = x[best_ind]
        res = (x[best_ind], f[best_ind], d[best_ind])
        if self.print is True:
            print(res)

        self.SSr = best_val

        # obtain the break point locations from the best result
        var = np.sort(best_break)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break0
        breaks[-1] = self.breakN

        # assign parameters
        self.fitWithBreaks(breaks)

        return self.fitBreaks

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

        # calculate the number of variables I have to solve for
        self.nVar = self.numberOfSegments - 1
