# -- coding: utf-8 --
# MIT License
#
# Copyright (c) 2017-2019 Charles Jekel
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
# from scipy.optimize import differential_evolution
# from scipy.optimize import fmin_l_bfgs_b
# from scipy.optimize import minimize
from scipy import linalg
# from scipy import stats
# from pyDOE import lhs
from .pwlf import PiecewiseLinFit


def assert_2d(x):
    if x.ndim != 2:
        raise ValueError('x must be a 2D array!')


class PiecewiseMultivariate(object):

    def __init__(self, x, y, n_segments, disp_res=False, lapack_driver='gelsd',
                 degree=1, multivariate_degree=1, constant=True):
        r"""
        Fit Multivariate models using a piecewise linear functions for each
        univariate.

        A general additive model (GAM) using the 1D PiecewiseLinFit class to
        model each univariate. Initiate the library with the supplied x and y
        data. Supply the x and y data of which you'll be fitting the model as
        y(x). By default pwlf won't print the optimization results.

        Parameters
        ----------
        x : ndarray (2-D)
            The x or independent data point as a 2 dimensional numpy array.
            This should be of shape (n, m), for n number of data points, and m
            number of features.
        y : array_like
            The y or dependent data point locations as list or 1 dimensional
            numpy array.
        n_segments : int
            The desired number of line segments for each univariate model.
        disp_res : bool, optional
            Whether the optimization results should be printed. Default is
            False.
        lapack_driver : str, optional
            Which LAPACK driver is used to solve the least-squares problem.
            Default lapack_driver='gelsd'. Options are 'gelsd', 'gelsy',
            'gelss'. For more see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
            http://www.netlib.org/lapack/lug/node27.html
        degree : int, optional
            The degree of polynomial to use for each univariate
            PiecewiseLinFit model. The default is degree=1 for linear models.
            Use degree=0 for constant models.
        multivariate_degree : int, optional
            The degree of the multivariate general additive model. For now 1 <=
            degree <= 10, with default degree=1.
        constant : bool, optional
            Whether to include a constant in the additive model. Default
            constant=True.

        Attributes
        ----------
        x_data : ndarray (2-D)
            The inputted parameter x from the 2-D data set.
        y_data : ndarray (1-D)
            The inputted parameter y from the 1-D data set.
        n_data : int
            The number of data points.
        break_0 : float
            The smallest x value.
        break_n : float
            The largest x value.
        print : bool
            Whether the optimization results should be printed. Default is
            False.
        lapack_driver : str
            Which LAPACK driver is used to solve the least-squares problem.

        Methods
        -------
        TBD.

        """

        self.print = disp_res
        self.lapack_driver = lapack_driver
        self.n_segments = n_segments
        # x and y should be numpy arrays
        # if they are not convert to numpy array
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        if isinstance(y, np.ndarray) is False:
            y = np.array(y)
        assert_2d(x)
        # calculate the number of data points
        self.n_data, self.n_features = x.shape
        self.x_data = x
        self.y_data = y

        if degree < 12 and degree >= 0:
            # I actually don't know what the upper degree limit should
            self.degree = int(degree)
        else:
            not_supported = "degree = " + str(degree) + " is not supported."
            raise ValueError(not_supported)

        self.constant = constant

        if multivariate_degree < 11 and multivariate_degree >= 1:
            # I actually don't know what the upper degree limit should
            self.multivariate_degree = int(multivariate_degree)
        else:
            not_supported = "multi variate degree = " + \
                str(multivariate_degree) + " is not supported."
            raise ValueError(not_supported)

        # initialize all empty attributes
        self.models = []

    def feature_check(self, n_features):
        r"""
        Check that x has the correct number of features.
        """
        # assert the same number of features as initialized model
        if n_features != self.n_features:
            msg = """x does not have the same number of features as
                     the initialized PiecewiseMultivariate object"""
            raise ValueError(msg)

    def assemble_regression_matrix(self, x):
        r"""
        Assemble the multivariate regression matrix.
        """
        # x should be numpy arrays
        # if they are not convert to numpy array
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        assert_2d(x)
        n_data, n_features = x.shape
        # Check that x has the same number of features as the initialized model
        self.feature_check(n_features)
        A_list = []
        if self.constant:
            A_list.append(np.ones(n_data))
        for model in range(self.n_features):
            y_hat = self.models[model].predict(x[:, model])
            for model_degree in range(self.multivariate_degree):
                A_list.append(y_hat**(model_degree + 1))
        A = np.vstack(A_list).T
        return A

    def fit(self):
        r"""
        Fit the multivariate piecewise linear model.

        Fits the general additive model (GAM) using the 1D PiecewiseLinFit
        class to model each univariate.

        Examples
        --------
        TBD.

        """
        self.models = []
        for i in range(self.n_features):
            # initialize univariate model
            my_pwlf = PiecewiseLinFit(self.x_data[:, i], self.y_data,
                                      disp_res=self.print,
                                      lapack_driver=self.lapack_driver,
                                      degree=self.degree)
            # fit univariate model
            my_pwlf.fit(self.n_segments)
            # store univariate model
            self.models.append(my_pwlf)
        A = self.assemble_regression_matrix(self.x_data)
        # perform a least squares fit to find the parameters of the GAM
        beta, ssr, rank, s = linalg.lstsq(A, self.y_data,
                                          lapack_driver=self.lapack_driver)

        # save the beta parameters and ssr
        self.beta = beta
        self.ssr = ssr
        return ssr

    def predict(self, x):
        r"""
        Generate predictions from the fitted Multivariate model.
        """
        # x should be numpy arrays
        # if they are not convert to numpy array
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        assert_2d(x)
        n_data, n_features = x.shape
        # Check that x has the same number of features as the initialized model
        self.feature_check(n_features)
        A = self.assemble_regression_matrix(x)
        return np.dot(A, self.beta)
