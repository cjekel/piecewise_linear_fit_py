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
import tensorflow as tf
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import fmin_l_bfgs_b
from scipy import stats
from pyDOE import lhs

# piecewise linear fit library


class PiecewiseLinFitTF(object):

    def __init__(self, x, y, disp_res=False, dtype='float64', fast=True):
        r"""
        An object to fit a continuous piecewise linear function
        to data using TensorFlow.

        Initiate the library with the supplied x and y data.
        Supply the x and y data of which you'll be fitting
        a continuous piecewise linear model to where y(x).
        by default pwlf won't print the optimization results.;

        Parameters
        ----------
        x : array_like
            The x or independent data point locations as list or 1 dimensional
            numpy array.
        y : array_like
            The y or dependent data point locations as list or 1 dimensional
            numpy array.
        disp_res : bool, optional
            Whether the optimization results should be printed. Default is
            False.
        dtype : str, optional
            Data type to use for tensorflow, either 'float64' or 'float32'.
            Default is 'float64'.
        fast : bool, optional
            If fast is True, then the solution is computed by solving the
            normal equations using Cholesky decomposition. Default fast=True.
            https://www.tensorflow.org/api_docs/python/tf/linalg/lstsq

        Attributes
        ----------
        x_data : tf.tensor (n_data, 1)
            The inputted parameter x from the 1-D data set.
        y_data : tf.tensor (n_data, 1)
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
        dtype : type
            Data type to use for tensorflow tensors.
        fast : boolean, optional
            If fast is True, then the solution is computed by solving the
            normal equations using Cholesky decomposition. Default fast=True.
            https://www.tensorflow.org/api_docs/python/tf/linalg/lstsq

        Methods
        -------
        fit(n_segments, x_c=None, y_c=None, **kwargs)
            Fit a continuous piecewise linear function for a specified number
            of line segments.
        fitfast(n_segments, pop=2, **kwargs)
            Fit a continuous piecewise linear function for a specified number
            of line segments using a specialized optimization routine that
            should be faster than fit() for large problems. The tradeoff may
            be that fitfast() results in a lower quality model.
        fit_with_breaks(breaks)
            Fit a continuous piecewise linear function where the breakpoint
            locations are known.
        fit_with_breaks_force_points(breaks, x_c, y_c)
            Fit a continuous piecewise linear function where the breakpoint
            locations are known, and force the fit to go through points at x_c
            and y_c.
        predict(x, sorted_data=False, beta=None, breaks=None)
            Evaluate the continuous piecewise linear function at new untested
            points.
        fit_with_breaks_opt(var)
            The objective function to perform a continuous piecewise linear
            fit for a specified number of breakpoints. This is to be used
            with a custom optimization routine, and after use_custom_opt has
            been called.
        fit_force_points_opt(var)'
            Same as fit_with_breaks_opt(var), except this allows for points to
            be forced through x_c and y_c.
        use_custom_opt(n_segments, x_c=None, y_c=None)
            Function to initialize the attributes necessary to use a custom
            optimization routine. Must be used prior to calling
            fit_with_breaks_opt() or fit_force_points_opt().
        calc_slopes()
            Calculate the slopes of the lines after a piecewise linear
            function has been fitted.
        standard_errors()
            Calculate the standard error of each model parameter in the fitted
            piecewise linear function. Note, this assumes no uncertainty in
            breakpoint locations.
        prediction_variance(x, sorted_data=True)
            Calculate the prediction variance at x locations for the fitted
            piecewise linear function. Note, assumes no uncertainty in break
            point locations.
        r_squared()
            Calculate the coefficient of determination, or 'R-squared' value
            for a fitted piecewise linear function.

        Examples
        --------
        Initialize for x, y data

        >>> import pwlf
        >>> my_pwlf = pwlf.PiecewiseLinFitTF(x, y)

        Initialize for x,y data and print optimization results

        >>> my_pWLF = pwlf.PiecewiseLinFitTF(x, y, disp_res=True)

        If your data is already sorted such that x[0] <= x[1] <= ... <= x[n-1],
        use sorted_data=True for a slight performance increase while
        initializing the object

        >>> my_pWLF = pwlf.PiecewiseLinFitTF(x, y, sorted_data=True)
        """
        self.fast = fast
        if dtype == 'float64':
            self.dtype = tf.float64
        else:
            self.dtype = tf.float32
        self.print = disp_res
        # calculate the number of data points
        self.n_data = x.size

        # set the first and last break x values to be the min and max of x
        self.break_0 = np.min(x)
        self.break_n = np.max(x)

        self.x_data = tf.convert_to_tensor(x.reshape(-1, 1), dtype=self.dtype)
        self.y_data = tf.convert_to_tensor(y.reshape(-1, 1), dtype=self.dtype)

    def assemble_regression_matrix(self, breaks, x):
        r"""
        Assemble the linear regression matrix A

        Parameters
        ----------
        breaks : array_like
            The x locations where each line segment terminates. These are
            referred to as breakpoints for each line segment. This should be
            structured as a 1-D numpy array.
        x : tf.tensor (n_data, 1)
            The x locations which the linear regression matrix is assembled on.
            This must be a numpy array!

        Attributes
        ----------
        fit_breaks : ndarray (1-D)
            breakpoint locations stored as a 1-D numpy array.
        n_parameters : int
            The number of model parameters. This is equivalent to the
            len(beta).
        n_segments : int
            The number of line segments.

        Returns
        -------
        A : tf.tensor (2-D)
            The assembled linear regression matrix.

        Examples
        --------
        Assemble the linear regression matrix on the x data for some set of
        breakpoints.

        >>> import pwlf
        >>> my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        >>> breaks = [0.0, 0.5, 1.0]
        >>> A = assemble_regression_matrix(breaks, self.x_data)

        """
        # Check if breaks in ndarray, if not convert to np.array
        if isinstance(breaks, np.ndarray) is False:
            breaks = np.array(breaks)

        # Sort the breaks, then store them
        breaks_order = np.argsort(breaks)
        self.fit_breaks = breaks[breaks_order]
        # store the number of parameters and line segments
        self.n_parameters = len(breaks)
        self.n_segments = self.n_parameters - 1

        # Assemble the regression matrix
        A_list = [tf.ones_like(x)]
        A_list.append(x - self.fit_breaks[0])
        zeros = tf.zeros_like(x)
        for i in range(self.n_segments - 1):
            mask = tf.greater(x, self.fit_breaks[i+1])
            A_list.append(tf.where(mask, x - self.fit_breaks[i+1], zeros))
        A = tf.concat(A_list, axis=1)
        return A

    def fit_with_breaks(self, breaks):
        r"""
        A function which fits a continuous piecewise linear function
        for specified breakpoint locations.

        The function minimizes the sum of the square of the residuals for the
        x y data.

        If you want to understand the math behind this read
        https://jekel.me/2018/Continous-piecewise-linear-regression/

        Other useful resources:
        http://golovchenko.org/docs/ContinuousPiecewiseLinearFit.pdf
        https://www.mathworks.com/matlabcentral/fileexchange/40913-piecewise-linear-least-square-fittic
        http://www.regressionist.com/2018/02/07/continuous-piecewise-linear-fitting/

        Parameters
        ----------
        breaks : array_like
            The x locations where each line segment terminates. These are
            referred to as breakpoints for each line segment. This should be
            structured as a 1-D numpy array.

        Attributes
        ----------
        fit_breaks : ndarray (1-D)
            breakpoint locations stored as a 1-D numpy array.
        n_parameters : int
            The number of model parameters. This is equivalent to the
            len(beta).
        n_segments : int
            The number of line segments.
        beta : ndarray (1-D)
            The model parameters for the continuous piecewise linear fit.
        betaTF : tf.tensor (n_parameters, 1)
            The model parameters for the continuous piecewise linear fit.
        slopes : ndarray (1-D)
            The slope of each ling segment as a 1-D numpy array. This assumes
            that x[0] <= x[1] <= ... <= x[n]. Thus, slopes[0] is the slope
            of the first line segment.
        intercepts : ndarray (1-D)
            The y-intercept of each line segment as a 1-D numpy array.

        Returns
        -------
        ssr : float
            Returns the sum of squares of the residuals.

        Notes
        -----
        The above attributes are added or modified while running this function.

        Examples
        --------
        If your x data exists from 0 <= x <= 1 and you want three
        piecewise linear lines where the lines terminate at x = 0.0, 0.3, 0.6,
        and 1.0. This assumes that x is linearly spaced from [0, 1), and y is
        random.

        >>> import pwlf
        >>> x = np.linspace(0.0, 1.0, 10)
        >>> y = np.random.random(10)
        >>> my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        >>> breaks = [0.0, 0.3, 0.6, 1.0]
        >>> ssr = my_pwlf.fit_with_breaks(breaks)

        """

        # Check if breaks in ndarray, if not convert to np.array
        if isinstance(breaks, np.ndarray) is False:
            breaks = np.array(breaks)

        A = self.assemble_regression_matrix(breaks, self.x_data)

        # least squares solver
        beta = tf.linalg.lstsq(A, self.y_data, fast=self.fast)
        y_hat = tf.matmul(A, beta)
        e = y_hat - self.y_data
        # compute the sum of square of the residuals
        SSr = tf.matmul(tf.transpose(e), e)
        self.betaTF = beta
        with tf.Session():
            # save the beta parameters
            self.beta = beta.eval().flatten()
            ssr = SSr.eval()

        # save the slopes
        self.calc_slopes()
        return ssr[0, 0]

    def fit_with_breaks_force_points(self, breaks, x_c, y_c):
        r"""
        A function which fits a continuous piecewise linear function
        for specified breakpoint locations, where you force the
        fit to go through the data points at x_c and y_c.

        The function minimizes the sum of the square of the residuals for the
        pair of x, y data points.

        If you want to understand the math behind this read
        https://jekel.me/2018/Force-piecwise-linear-fit-through-data/

        Parameters
        ----------
        breaks : array_like
            The x locations where each line segment terminates. These are
            referred to as breakpoints for each line segment. This should be
            structured as a 1-D numpy array.
        x_c : array_like
            The x locations of the data points that the piecewise linear
            function will be forced to go through.
        y_c : array_like
            The x locations of the data points that the piecewise linear
            function will be forced to go through.

        Attributes
        ----------
        fit_breaks : ndarray (1-D)
            breakpoint locations stored as a 1-D numpy array.
        n_parameters : int
            The number of model parameters. This is equivalent to the
            len(beta).
        n_segments : int
            The number of line segments.
        beta : ndarray (1-D)
            The model parameters for the continuous piecewise linear fit.
        betaTF : tf.tensor (n_parameters, 1)
            The model parameters for the continuous piecewise linear fit.
        zeta : ndarray (1-D)
            The model parameters associated with the constraint function.
        zetaTF : tf.tensor (c_n, 1)
            The model parameters associated with the constraint function.
        slopes : ndarray (1-D)
            The slope of each ling segment as a 1-D numpy array. This assumes
            that x[0] <= x[1] <= ... <= x[n]. Thus, slopes[0] is the slope
            of the first line segment.
        intercepts : ndarray (1-D)
            The y-intercept of each line segment as a 1-D numpy array.
        x_c : tf.tensor (c_n, 1)
            The x locations of the data points that the piecewise linear
            function will be forced to go through.
        y_c : tf.tensor (c_n, 1)
            The x locations of the data points that the piecewise linear
            function will be forced to go through.
        c_n : int
            The number of constraint points. This is the same as len(x_c).

        Returns
        -------
        L : float
            Returns the Lagrangian function value. This is the sum of squares
            of the residuals plus the constraint penalty.

        Raises
        ------
        LinAlgError
            This typically means your regression problem is ill-conditioned.

        Notes
        -----
        The above attributes are added or modified while running this function.
        Input:

        Examples
        -------
        If your x data exists from 0 <= x <= 1 and you want three
        piecewise linear lines where the lines terminate at x = 0.0, 0.3, 0.6,
        and 1.0. This assumes that x is linearly spaced from [0, 1), and y is
        random. Additionally you desired that the piecewise linear function go
        through the point (0.0, 0.0)

        >>> import pwlf
        >>> x = np.linspace(0.0, 1.0, 10)
        >>> y = np.random.random(10)
        >>> x_c = [0.0]
        >>> y_c = [0.0]
        >>> my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        >>> breaks = [0.0, 0.3, 0.6, 1.0]
        >>> L = my_pwlf.fit_with_breaks_force_points(breaks, x_c, y_c)

        """
        self.c_n = len(x_c)
        if isinstance(x_c, np.ndarray) is False:
            x_c = np.array(x_c)
        if isinstance(y_c, np.ndarray) is False:
            y_c = np.array(y_c)
        # sort the x_c and y_c data points, then store them
        x_c_order = np.argsort(x_c)
        self.x_c = tf.convert_to_tensor(x_c[x_c_order].reshape(-1, 1),
                                        self.dtype)
        self.y_c = tf.convert_to_tensor(y_c[x_c_order].reshape(-1, 1),
                                        self.dtype)

        # Check if breaks in ndarray, if not convert to np.array
        if isinstance(breaks, np.ndarray) is False:
            breaks = np.array(breaks)

        A = self.assemble_regression_matrix(breaks, self.x_data)

        # penalty matrix
        zeros = tf.zeros_like(self.x_c)
        C_list = []
        C_list.append(tf.ones_like(self.x_c))
        C_list.append(self.x_c - self.fit_breaks[0])
        for i in range(self.n_segments - 1):
            mask = tf.greater(self.x_c, self.fit_breaks[i+1])
            C_list.append(tf.where(mask, self.x_c - self.fit_breaks[i+1],
                                   zeros))
        C = tf.concat(C_list, axis=1)

        # assemble the KKT matrix K
        A2T = 2.0 * tf.linalg.matmul(tf.transpose(A), A)
        CT = tf.transpose(C)
        K = tf.concat([A2T, CT], axis=1)
        Cz = tf.concat([C, tf.zeros((self.c_n, self.c_n), tf.float64)], axis=1)
        K = tf.concat([K, Cz], axis=0)
        yt = 2.0 * tf.linalg.matmul(tf.transpose(A), self.y_data)
        z = tf.concat([yt, self.y_c], axis=0)

        # solve the regression problem
        beta = tf.linalg.matmul(tf.linalg.inv(K), z)

        # compute the sum of square of the residuals
        y_hat = tf.matmul(A, beta[0:self.n_parameters])
        e = y_hat - self.y_data
        SSr = tf.matmul(tf.transpose(e), e)

        # Calculate the Lagrangian function
        p = tf.matmul(CT, beta[self.n_parameters:])

        self.beta_prime = beta
        with tf.Session():
            # save the beta parameters
            beta_prime = beta.eval()
            # save the beta parameters
            self.beta = beta_prime[0:self.n_parameters]
            self.betaTF = beta[0:self.n_parameters]
            # save the zeta parameters
            self.zeta = beta_prime[self.n_parameters:]
            self.zetaTF = beta[self.n_parameters:]
            ssr = SSr.eval()
            L = ssr[0, 0] + np.sum(np.abs(p.eval()))

        # save the slopes
        self.calc_slopes()
        return L

    def predict(self, x, beta=None, breaks=None):
        r"""
        Evaluate the fitted continuous piecewise linear function at untested
        points.

        You can manfully specify the breakpoints and calculated
        values for beta if you want to quickly predict from different models
        and the same data set.

        Parameters
        ----------
        x : array_like
            The x locations where you want to predict the output of the fitted
            continuous piecewise linear function.
        beta : none or ndarray (1-D), optional
            The model parameters for the continuous piecewise linear fit.
            Default is None.
        breaks : none or array_like, optional
            The x locations where each line segment terminates. These are
            referred to as breakpoints for each line segment. This should be
            structured as a 1-D numpy array. Default is None.

        Attributes
        ----------
        fit_breaks : ndarray (1-D)
            breakpoint locations stored as a 1-D numpy array.
        n_parameters : int
            The number of model parameters. This is equivalent to the
            len(beta).
        n_segments : int
            The number of line segments.
        beta : ndarray (1-D)
            The model parameters for the continuous piecewise linear fit.
        betaTF : tf.tensor (n_parameters, 1)
            The model parameters for the continuous piecewise linear fit.

        Returns
        -------
        y_hat : ndarray (1-D)
            The predicted values at x.

        Notes
        -----
        The above attributes are added or modified if any optional parameter
        is specified.

        Examples
        -------
        Fits a simple model, then predict at x_new locations which are
        linearly spaced.

        >>> import pwlf
        >>> x = np.linspace(0.0, 1.0, 10)
        >>> y = np.random.random(10)
        >>> my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        >>> breaks = [0.0, 0.3, 0.6, 1.0]
        >>> ssr = my_pwlf.fit_with_breaks(breaks)
        >>> x_new = np.linspace(0.0, 1.0, 100)
        >>> yhat = my_pwlf.predict(x_new)

        """
        if beta is not None and breaks is not None:
            self.beta = beta
            self.betaTF = tf.convert_to_tensor(beta.reshape(-1, 1),
                                               dtype=self.dtype)
            # Sort the breaks, then store them
            breaks_order = np.argsort(breaks)
            self.fit_breaks = breaks[breaks_order]
            self.n_parameters = len(self.fit_breaks)
            self.n_segments = self.n_parameters - 1

        # check if x is numpy array, if not convert to numpy array
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        X = tf.convert_to_tensor(x.reshape(-1, 1), dtype=self.dtype)
        A = self.assemble_regression_matrix(self.fit_breaks, X)

        # solve the regression problem
        y_hat = tf.matmul(A, self.betaTF)
        with tf.Session():
            return y_hat.eval().flatten()

    def fit_with_breaks_opt(self, var):
        r"""
        The objective function to perform a continuous piecewise linear
        fit for a specified number of breakpoints. This is to be used
        with a custom optimization routine, and after use_custom_opt has
        been called.

        This was intended for advanced users only.

        See the following example
        https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/useCustomOptimizationRoutine.py

        Parameters
        ----------
        var : array_like
            The breakpoint locations, or variable, in a custom
            optimization routine.

        Returns
        -------
        ssr : float
            The sum of square of the residuals.

        Raises
        ------
        LinAlgError
            This typically means your regression problem is ill-conditioned.

        Notes
        -----
        You should run use_custom_opt to initialize necessary object
        attributes first.

        Unlike fit_with_breaks, fit_with_breaks_opt automatically
        assumes that the first and last breakpoints occur at the min and max
        values of x.
        """

        var = np.sort(var)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break_0
        breaks[-1] = self.break_n

        A = self.assemble_regression_matrix(breaks, self.x_data)

        # least squares solver
        beta = tf.linalg.lstsq(A, self.y_data, fast=self.fast)
        y_hat = tf.matmul(A, beta)
        e = y_hat - self.y_data
        # compute the sum of square of the residuals
        SSr = tf.matmul(tf.transpose(e), e)
        self.betaTF = beta
        # save the beta parameters
        try:
            self.beta = beta.eval().flatten()
        except tf.errors.InvalidArgumentError:
            # this means the Cholesky decomposition was not successful!
            # the try; except must be placed around the executing code in tf!
            return np.inf
        ssr = SSr.eval()
        return ssr[0, 0]

    def fit_force_points_opt(self, var):
        r"""
        The objective function to perform a continuous piecewise linear
        fit for a specified number of breakpoints. This is to be used
        with a custom optimization routine, and after use_custom_opt has
        been called.

        Use this function if you intend to be force the model through
        x_c and y_c, while performing a custom optimization.

        This was intended for advanced users only.
        See the following example
        https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/useCustomOptimizationRoutine.py

        Parameters
        ----------
        var : array_like
            The breakpoint locations, or variable, in a custom
            optimization routine.

        Returns
        -------
        ssr : float
            The sum of square of the residuals.

        Raises
        ------
        LinAlgError
            This typically means your regression problem is ill-conditioned.

        Notes
        -----
        You should run use_custom_opt to initialize necessary object
        attributes first.

        Unlike fit_with_breaks_force_points, fit_force_points_opt
        automatically assumes that the first and last breakpoints occur
        at the min and max values of x.
        """

        var = np.sort(var)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break_0
        breaks[-1] = self.break_n

        # Sort the breaks, then store them
        breaks_order = np.argsort(breaks)
        breaks = breaks[breaks_order]

        A = self.assemble_regression_matrix(breaks, self.x_data)

        # penalty matrix
        zeros = tf.zeros_like(self.x_c)
        C_list = []
        C_list.append(tf.ones_like(self.x_c))
        C_list.append(self.x_c - self.fit_breaks[0])
        for i in range(self.n_segments - 1):
            mask = tf.greater(self.x_c, self.fit_breaks[i+1])
            C_list.append(tf.where(mask, self.x_c - self.fit_breaks[i+1],
                                   zeros))
        C = tf.concat(C_list, axis=1)

        # assemble the KKT matrix K
        A2T = 2.0 * tf.linalg.matmul(tf.transpose(A), A)
        CT = tf.transpose(C)
        K = tf.concat([A2T, CT], axis=1)
        Cz = tf.concat([C, tf.zeros((self.c_n, self.c_n), tf.float64)], axis=1)
        K = tf.concat([K, Cz], axis=0)
        yt = 2.0 * tf.linalg.matmul(tf.transpose(A), self.y_data)
        z = tf.concat([yt, self.y_c], axis=0)

        # solve the regression problem
        beta = tf.linalg.matmul(tf.linalg.inv(K), z)

        # compute the sum of square of the residuals
        y_hat = tf.matmul(A, beta[0:self.n_parameters])
        e = y_hat - self.y_data
        SSr = tf.matmul(tf.transpose(e), e)

        # Calculate the Lagrangian function
        p = tf.matmul(CT, beta[self.n_parameters:])

        self.beta_prime = beta
        # save the beta parameters
        try:
            beta_prime = beta.eval()
        except tf.errors.InvalidArgumentError:
            # the K matrix was not invertible!
            # the try; except must be placed around the executing code in tf!
            return np.inf
        # save the beta parameters
        self.beta = beta_prime[0:self.n_parameters]
        self.betaTF = beta[0:self.n_parameters]
        # save the zeta parameters
        self.zeta = beta_prime[self.n_parameters:]
        self.zetaTF = beta[self.n_parameters:]
        ssr = SSr.eval()
        L = ssr[0, 0] + np.sum(np.abs(p.eval()))
        return L

    def fit(self, n_segments, x_c=None, y_c=None, **kwargs):
        r"""
        Fit a continuous piecewise linear function for a specified number
        of line segments. Uses differential evolution to finds the optimum
        location of breakpoints for a given number of line segments by
        minimizing the sum of the square error.

        Parameters
        ----------
        n_segments : int
            The desired number of line segments.
        x_c : array_like, optional
            The x locations of the data points that the piecewise linear
            function will be forced to go through.
        y_c : array_like, optional
            The x locations of the data points that the piecewise linear
            function will be forced to go through.
        **kwargs : optional
            Directly passed into scipy.optimize.differential_evolution(). This
            will override any pwlf defaults when provided. See Note for more
            information.

        Attributes
        ----------
        ssr : float
            Optimal sum of square error.
        fit_breaks : ndarray (1-D)
            breakpoint locations stored as a 1-D numpy array.
        n_parameters : int
            The number of model parameters. This is equivalent to the
            len(beta).
        n_segments : int
            The number of line segments.
        nVar : int
            The number of variables in the global optimization problem.
        beta : ndarray (1-D)
            The model parameters for the continuous piecewise linear fit.
        betaTF : tf.tensor (n_parameters, 1)
            The model parameters for the continuous piecewise linear fit.
        zeta : ndarray (1-D)
            The model parameters associated with the constraint function,
            if x_c and y_c is provided. Only created if x_c and y_c provided.
        zetaTF : tf.tensor (c_n, 1)
            The model parameters associated with the constraint function,
            if x_c and y_c is provided. Only created if x_c and y_c provided.
        slopes : ndarray (1-D)
            The slope of each ling segment as a 1-D numpy array. This assumes
            that x[0] <= x[1] <= ... <= x[n]. Thus, slopes[0] is the slope
            of the first line segment.
        intercepts : ndarray (1-D)
            The y-intercept of each line segment as a 1-D numpy array.
        x_c : tf.tensor (c_n, 1)
            The x locations of the data points that the piecewise linear
            function will be forced to go through. Only created if x_c
            and y_c provided.
        y_c : ndarray (c_n, 1)
            The x locations of the data points that the piecewise linear
            function will be forced to go through. Only created if x_c
            and y_c provided.
        c_n : int
            The number of constraint points. This is the same as len(x_c).
            Only created if x_c and y_c provided.

        Returns
        -------
        fit_breaks : float
            breakpoint locations stored as a 1-D numpy array.

        Raises
        ------
        ValueError
            You probably provided x_c without y_c (or vice versa).
            You must provide both x_c and y_c if you plan to force
            the model through data point(s).

        Notes
        -----
        All **kwargs are passed into sicpy.optimize.differential_evolution.
        If any **kwargs is used, it will override my differential_evolution,
        defaults. This allows advanced users to tweak their own optimization.
        For me information see:
        https://github.com/cjekel/piecewise_linear_fit_py/issues/15#issuecomment-434717232

        Examples
        --------
        This example shows you how to fit three continuous piecewise lines to
        a dataset. This assumes that x is linearly spaced from [0, 1), and y is
        random.

        >>> import pwlf
        >>> x = np.linspace(0.0, 1.0, 10)
        >>> y = np.random.random(10)
        >>> my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        >>> breaks = my_pwlf.fit(3)

        Additionally you desired that the piecewise linear function go
        through the point (0.0, 0.0).

        >>> x_c = [0.0]
        >>> y_c = [0.0]
        >>> breaks = my_pwlf.fit(3, x_c=x_c, y_c=y_c)

        Additionally you desired that the piecewise linear function go
        through the points (0.0, 0.0) and (1.0, 1.0).

        >>> x_c = [0.0, 1.0]
        >>> y_c = [0.0, 1.0]
        >>> breaks = my_pwlf.fit(3, x_c=x_c, y_c=y_c)

        """

        # check to see if you've provided just x_c or y_c
        logic1 = x_c is not None and y_c is None
        logic2 = y_c is not None and x_c is None
        if logic1 or logic2:
            raise ValueError('You must provide both x_c and y_c!')

        # set the function to minimize
        min_function = self.fit_with_breaks_opt

        # if you've provided both x_c and y_c
        if x_c is not None and y_c is not None:
            # check if x_c and y_c are numpy array
            # if not convert to numpy array
            if isinstance(x_c, np.ndarray) is False:
                x_c = np.array(x_c)
            if isinstance(y_c, np.ndarray) is False:
                y_c = np.array(y_c)
            # sort the x_c and y_c data points, then store them
            x_c_order = np.argsort(x_c)
            x_c_np = x_c[x_c_order]
            y_c_np = y_c[x_c_order]
            self.c_n = len(x_c)
            self.x_c = tf.convert_to_tensor(x_c_np.reshape(-1, 1), self.dtype)
            self.y_c = tf.convert_to_tensor(y_c_np.reshape(-1, 1), self.dtype)
            # Use a different function to minimize
            min_function = self.fit_force_points_opt

        # store the number of line segments and number of parameters
        self.n_segments = int(n_segments)
        self.n_parameters = self.n_segments + 1

        # calculate the number of variables I have to solve for
        self.nVar = self.n_segments - 1

        # initiate the bounds of the optimization
        bounds = np.zeros([self.nVar, 2])
        bounds[:, 0] = self.break_0
        bounds[:, 1] = self.break_n

        # run the optimization
        with tf.Session():
            if len(kwargs) == 0:
                res = differential_evolution(min_function, bounds,
                                             strategy='best1bin', maxiter=1000,
                                             popsize=50, tol=1e-3,
                                             mutation=(0.5, 1),
                                             recombination=0.7, seed=None,
                                             callback=None, disp=False,
                                             polish=True,
                                             init='latinhypercube', atol=1e-4)
            else:
                res = differential_evolution(min_function, bounds, **kwargs)
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
        if x_c is None and y_c is None:
            self.fit_with_breaks(breaks)
        else:
            self.fit_with_breaks_force_points(breaks, x_c_np, y_c_np)

        return self.fit_breaks

    def fitfast(self, n_segments, pop=2, **kwargs):
        r"""
        Uses multi start LBFGSB optimization to find the location of
        breakpoints for a given number of line segments by minimizing the sum
        of the square of the errors.

        The idea is that we generate n random latin hypercube samples
        and run LBFGSB optimization on each one. This isn't guaranteed to
        find the global optimum. It's suppose to be a reasonable compromise
        between speed and quality of fit. Let me know how it works.

        Since this is based on random sampling, you might want to run it
        multiple times and save the best version... The best version will
        have the lowest self.ssr (sum of square of residuals).

        There is no guarantee that this will be faster than fit(), however
        you may find it much faster sometimes.

        Parameters
        ----------
        n_segments : int
            The desired number of line segments.
        pop : int, optional
            The number of latin hypercube samples to generate. Default pop=2.
        **kwargs : optional
            Directly passed into scipy.optimize.differential_evolution(). This
            will override any pwlf defaults when provided. See Note for more
            information.

        Attributes
        ----------
        ssr : float
            Optimal sum of square error.
        fit_breaks : ndarray (1-D)
            breakpoint locations stored as a 1-D numpy array.
        n_parameters : int
            The number of model parameters. This is equivalent to the
            len(beta).
        n_segments : int
            The number of line segments.
        nVar : int
            The number of variables in the global optimization problem.
        beta : ndarray (1-D)
            The model parameters for the continuous piecewise linear fit.
        betaTF : tf.tensor (n_parameters, 1)
            The model parameters for the continuous piecewise linear fit.
        slopes : ndarray (1-D)
            The slope of each ling segment as a 1-D numpy array. This assumes
            that x[0] <= x[1] <= ... <= x[n]. Thus, slopes[0] is the slope
            of the first line segment.
        intercepts : ndarray (1-D)
            The y-intercept of each line segment as a 1-D numpy array.

        Returns
        -------
        fit_breaks : float
            breakpoint locations stored as a 1-D numpy array.

        Notes
        -----
        The default number of multi start optimizations is 2.
            - Decreasing this number will result in a faster run time.
            - Increasing this number will improve the likelihood of finding
                good results
            - You can specify the number of starts using the following call
            - Minimum value of pop is 2

        All **kwargs are passed into sicpy.optimize.fmin_l_bfgs_b. If any
        **kwargs is used, it will override my defaults. This allows
        advanced users to tweak their own optimization. For me information see:
        https://github.com/cjekel/piecewise_linear_fit_py/issues/15#issuecomment-434717232

        Examples
        --------
        This example shows you how to fit three continuous piecewise lines to
        a dataset. This assumes that x is linearly spaced from [0, 1), and y is
        random.

        >>> import pwlf
        >>> x = np.linspace(0.0, 1.0, 10)
        >>> y = np.random.random(10)
        >>> my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        >>> breaks = my_pwlf.fitfast(3)

        You can change the number of latin hypercube samples (or starting
        point, locations) to use with pop. The following example will use 50
        samples.

        >>> breaks = my_pwlf.fitfast(3, pop=50)

        """
        pop = int(pop)  # ensure that the population is integer

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
        with tf.Session():
            for i, x0 in enumerate(mypop):
                if len(kwargs) == 0:
                    resx, resf, resd = fmin_l_bfgs_b(self.fit_with_breaks_opt,
                                                     x0, fprime=None, args=(),
                                                     approx_grad=True,
                                                     bounds=bounds, m=10,
                                                     factr=1e2, pgtol=1e-05,
                                                     epsilon=1e-08,
                                                     iprint=-1, maxfun=15000,
                                                     maxiter=15000, disp=None,
                                                     callback=None)
                else:
                    resx, resf, resd = fmin_l_bfgs_b(self.fit_with_breaks_opt,
                                                     x0, fprime=None,
                                                     approx_grad=True,
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

            # obtain the breakpoint locations from the best result
            var = np.sort(best_break)
            breaks = np.zeros(len(var) + 2)
            breaks[1:-1] = var.copy()
            breaks[0] = self.break_0
            breaks[-1] = self.break_n

        # assign parameters
        self.fit_with_breaks(breaks)

        return self.fit_breaks

    def fit_guess(self, guess_breakpoints, **kwargs):
        r"""
        Uses L-BFGS-B optimization to find the location of breakpoints
        from a guess of where breakpoint locations should be.

        In some cases you may have a good idea where the breakpoint locations
        occur. It generally won't be necessary to run a full global
        optimization to search the entire domain for the breakpoints when you
        have a good idea where the breakpoints occur. Here a local optimization
        is run from a guess of the breakpoint locations.

        Parameters
        ----------
        guess_breakpoints : array_like
            Guess where the breakpoints occur. This should be a list or numpy
            array containing the locations where it appears breakpoints occur.

        Attributes
        ----------
        ssr : float
            Optimal sum of square error.
        fit_breaks : ndarray (1-D)
            breakpoint locations stored as a 1-D numpy array.
        n_parameters : int
            The number of model parameters. This is equivalent to the
            len(beta).
        n_segments : int
            The number of line segments.
        nVar : int
            The number of variables in the global optimization problem.
        beta : ndarray (1-D)
            The model parameters for the continuous piecewise linear fit.
        betaTF : tf.tensor (n_parameters, 1)
            The model parameters for the continuous piecewise linear fit.
        slopes : ndarray (1-D)
            The slope of each ling segment as a 1-D numpy array. This assumes
            that x[0] <= x[1] <= ... <= x[n]. Thus, slopes[0] is the slope
            of the first line segment.
        intercepts : ndarray (1-D)
            The y-intercept of each line segment as a 1-D numpy array.

        Returns
        -------
        fit_breaks : float
            breakpoint locations stored as a 1-D numpy array.

        Notes
        -----
        All **kwargs are passed into sicpy.optimize.fmin_l_bfgs_b. If any
        **kwargs is used, it will override my defaults. This allows
        advanced users to tweak their own optimization. For me information see:
        https://github.com/cjekel/piecewise_linear_fit_py/issues/15#issuecomment-434717232

        You do not need to specify the x.min() or x.max() in geuss_breakpoints!

        Examples
        --------
        In this example we see two distinct linear regions, and we believe a
        breakpoint occurs at 6.0. We'll use the fit_guess() function to find
        the best breakpoint location starting with this guess.

        >>> import pwlf
        >>> x = np.array([4., 5., 6., 7., 8.])
        >>> y = np.array([11., 13., 16., 28.92, 42.81])
        >>> my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        >>> breaks = my_pwlf.fit_guess([6.0])

        Note specifying one breakpoint will result in two line segments.
        If we wanted three line segments, we'll have to specify two
        breakpoints.

        >>> breaks = my_pwlf.fit_guess([5.5, 6.0])

        """
        # calculate the number of variables I have to solve for
        self.nVar = len(guess_breakpoints)
        self.n_segments = self.nVar + 1
        self.n_parameters = self.n_segments + 1

        # initiate the bounds of the optimization
        bounds = np.zeros([self.nVar, 2])
        bounds[:, 0] = self.break_0
        bounds[:, 1] = self.break_n
        with tf.Session():
            if len(kwargs) == 0:
                resx, resf, _ = fmin_l_bfgs_b(self.fit_with_breaks_opt,
                                              guess_breakpoints,
                                              fprime=None, args=(),
                                              approx_grad=True,
                                              bounds=bounds, m=10,
                                              factr=1e2, pgtol=1e-05,
                                              epsilon=1e-08, iprint=-1,
                                              maxfun=15000, maxiter=15000,
                                              disp=None, callback=None)
            else:
                resx, resf, _ = fmin_l_bfgs_b(self.fit_with_breaks_opt,
                                              guess_breakpoints,
                                              fprime=None, approx_grad=True,
                                              bounds=bounds, **kwargs)

        self.ssr = resf

        # pull the breaks out of the result
        var = np.sort(resx)
        breaks = np.zeros(len(var) + 2)
        breaks[1:-1] = var.copy()
        breaks[0] = self.break_0
        breaks[-1] = self.break_n

        # assign values
        self.fit_with_breaks(breaks)

        return self.fit_breaks

    def use_custom_opt(self, n_segments, x_c=None, y_c=None):
        r"""
        Provide the number of line segments you want to use with your
        custom optimization routine.

        Run this function first to initialize necessary attributes!!!

        This was intended for advanced users only.

        See the following example
        https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/useCustomOptimizationRoutine.py

        Parameters
        ----------
        n_segments : int
            The x locations where each line segment terminates. These are
            referred to as breakpoints for each line segment. This should be
            structured as a 1-D numpy array.
        x_c : none or array_like, optional
            The x locations of the data points that the piecewise linear
            function will be forced to go through.
        y_c : none or array_like, optional
            The x locations of the data points that the piecewise linear
            function will be forced to go through.

        Attributes
        ----------
        n_parameters : int
            The number of model parameters. This is equivalent to the
            len(beta).
        nVar : int
            The number of variables in the global optimization problem.
        n_segments : int
            The number of line segments.
        x_c : tf.tensor (c_n, 1)
            The x locations of the data points that the piecewise linear
            function will be forced to go through.
        y_c : ndarray (c_n, 1)
            The x locations of the data points that the piecewise linear
            function will be forced to go through.
        c_n : int
            The number of constraint points. This is the same as len(x_c).

        Notes
        -----
        Optimize fit_with_breaks_opt(var) where var is a 1D array
        containing the x locations of your variables
        var has length n_segments - 1, because the two breakpoints
        are always defined (1. the min of x, 2. the max of x).

        fit_with_breaks_opt(var) will return the sum of the square of the
        residuals which you'll want to minimize with your optimization
        routine.
        """

        self.n_segments = int(n_segments)
        self.n_parameters = self.n_segments + 1

        # calculate the number of variables I have to solve for
        self.nVar = self.n_segments - 1
        if x_c is not None or y_c is not None:
            # check if x_c and y_c are numpy array
            # if not convert to numpy array
            if isinstance(x_c, np.ndarray) is False:
                x_c = np.array(x_c)
            if isinstance(y_c, np.ndarray) is False:
                y_c = np.array(y_c)
            # sort the x_c and y_c data points, then store them
            x_c_order = np.argsort(x_c)
            self.c_n = len(self.x_c)
            self.x_c = tf.convert_to_tensor(x_c[x_c_order].reshape(-1, 1),
                                            self.dtype)
            self.y_c = tf.convert_to_tensor(y_c[x_c_order].reshape(-1, 1),
                                            self.dtype)

    def standard_errors(self):
        r"""
        Calculate the standard errors for each beta parameter determined
        from the piecewise linear fit. Typically +- 1.96*se will yield the
        center of a 95% confidence region around your parameters. This
        assumes the parmaters follow a normal distribution. For more
        information see:
        https://en.wikipedia.org/wiki/Standard_error

        This calculation follows the derivation provided in [1]_. A taylor-
        series expansion is not needed since this is linear regression.

        Returns
        -------
        se : ndarray (1-D)
            Standard errors associated with each beta parameter. Specifically
            se[0] correspounds to the standard error for beta[0], and so forth.

        Raises
        ------
        ValueError
            You have probably not performed a fit yet.
        LinAlgError
            This typically means your regression problem is ill-conditioned.

        Notes
        -----
        Note, this assumes no uncertainty in breakpoint locations.

        References
        ----------
        .. [1] Coppe, A., Haftka, R. T., and Kim, N. H., “Uncertainty
            Identification of Damage Growth Parameters Using Nonlinear
            Regression,” AIAA Journal, Vol. 49, No. 12, dec 2011, pp.
            2818–2821.

        Examples
        --------
        Calculate the standard errors after performing a simple fit.

        >>> import pwlf
        >>> x = np.linspace(0.0, 1.0, 10)
        >>> y = np.random.random(10)
        >>> my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        >>> breaks = my_pwlf.fitfast(3)
        >>> se = my_pwlf.standard_errors()

        """
        try:
            nb = self.beta.size
        except ValueError:
            errmsg = 'You do not have any beta parameters. You must perform' \
                     ' a fit before using standard_errors().'
            raise ValueError(errmsg)

        ny = self.n_data

        A = self.assemble_regression_matrix(self.fit_breaks, self.x_data)
        y_hat = tf.matmul(A, self.betaTF)
        e = y_hat - self.y_data
        variance = tf.matmul(tf.transpose(e), e) / (ny - nb)
        AtAinv = tf.linalg.inv(tf.matmul(tf.transpose(A), A))
        with tf.Session():
            var = variance.eval()[0, 0] * AtAinv.eval().diagonal()
            self.se = np.sqrt(var)
            return self.se

    def prediction_variance(self, x):
        r"""
        Calculate the prediction variance for each specified x location. The
        prediction variance is the uncertainty of the model due to the lack of
        data. This can be used to find a 95% confidence interval of possible
        piecewise linear models based on the current data. This would be
        done typically as y_hat +- 1.96*np.sqrt(pre_var). The
        prediction_variance needs to be calculated at various x locations.
        For more information see:
        www2.mae.ufl.edu/haftka/vvuq/lectures/Regression-accuracy.pptx

        Parameters
        ----------
        x : array_like
            The x locations where you want the prediction variance from the
            fitted continuous piecewise linear function.

        Returns
        -------
        pre_var : ndarray (1-D)
            Numpy array (floats) of prediction variance at each x location.

        Raises
        ------
        ValueError
            You have probably not performed a fit yet.
        LinAlgError
            This typically means your regression problem is ill-conditioned.

        Notes
        -----
        This assumes that your breakpoint locations are exact! and does
        not consider the uncertainty with your breakpoint locations.

        Examples
        --------
        Calculate the prediction variance at x_new after performing a simple
        fit.

        >>> import pwlf
        >>> x = np.linspace(0.0, 1.0, 10)
        >>> y = np.random.random(10)
        >>> my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        >>> breaks = my_pwlf.fitfast(3)
        >>> x_new = np.linspace(0.0, 1.0, 100)
        >>> pre_var = my_pwlf.prediction_variance(x_new)

        see also examples/prediction_variance.py

        """
        try:
            nb = self.beta.size
        except ValueError:
            errmsg = 'You do not have any beta parameters. You must perform' \
                     ' a fit before using standard_errors().'
            raise ValueError(errmsg)

        ny = self.n_data

        # check if x is numpy array, if not convert to numpy array
        if isinstance(x, np.ndarray) is False:
            x = np.array(x)
        # convert x to a tensor
        x = tf.convert_to_tensor(x.reshape(-1, 1), dtype=self.dtype)

        # Regression matrix on training data
        Ad = self.assemble_regression_matrix(self.fit_breaks, self.x_data)

        y_hat = tf.matmul(Ad, self.betaTF)
        e = y_hat - self.y_data
        variance = tf.matmul(tf.transpose(e), e) / (ny - nb)

        # Regression matrix on prediction data
        A = self.assemble_regression_matrix(self.fit_breaks, x)
        AdtAdinv = tf.linalg.inv(tf.matmul(tf.transpose(Ad), Ad))
        AAdtAdinvAT = tf.matmul(tf.matmul(A, AdtAdinv), tf.transpose(A))

        with tf.Session():
            pre_var = variance.eval()[0, 0] * AAdtAdinvAT.eval().diagonal()
            return pre_var

    def r_squared(self):
        r"""
        Calculate the coefficient of determination ("R squared", R^2) value
        after a fit has been performed.
        For more information see:
        https://en.wikipedia.org/wiki/Coefficient_of_determination

        Returns
        -------
        rsq : float
            Coefficient of determination, or 'R squared' value.

        Raises
        ------
        ValueError
            You have probably not performed a fit yet.
        LinAlgError
            This typically means your regression problem is ill-conditioned.

        Examples
        --------
        Calculate the R squared value after performing a simple fit.

        >>> import pwlf
        >>> x = np.linspace(0.0, 1.0, 10)
        >>> y = np.random.random(10)
        >>> my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        >>> breaks = my_pwlf.fitfast(3)
        >>> rsq = my_pwlf.r_squared()

        """
        try:
            fit_breaks = self.fit_breaks
        except ValueError:
            errmsg = 'You do not have any beta parameters. You must perform' \
                     ' a fit before using standard_errors().'
            raise ValueError(errmsg)
        ssr = self.fit_with_breaks(fit_breaks)
        ybar = tf.ones_like(self.y_data) * tf.reduce_mean(self.y_data)
        # tf.reduce_mean defaults to flattening the entire tensor
        ydiff = self.y_data - ybar
        sst = tf.matmul(tf.transpose(ydiff), ydiff)
        with tf.Session():
            rsq = 1.0 - (ssr/sst.eval()[0, 0])
        return rsq

    def calc_slopes(self):
        r"""
        Calculate the slopes of the lines after a piecewise linear
        function has been fitted.

        This will also calculate the y-intercept from each line in the form
        y = mx + b. The intercepts are stored at self.intercepts.

        Attributes
        ----------
        slopes : ndarray (1-D)
            The slope of each ling segment as a 1-D numpy array. This assumes
            that x[0] <= x[1] <= ... <= x[n]. Thus, slopes[0] is the slope
            of the first line segment.
        intercepts : ndarray (1-D)
            The y-intercept of each line segment as a 1-D numpy array.

        Returns
        -------
        slopes : ndarray(1-D)
            The slope of each ling segment as a 1-D numpy array. This assumes
            that x[0] <= x[1] <= ... <= x[n]. Thus, slopes[0] is the slope
            of the first line segment.

        Examples
        --------
        Calculate the slopes after performing a simple fit

        >>> import pwlf
        >>> x = np.linspace(0.0, 1.0, 10)
        >>> y = np.random.random(10)
        >>> my_pwlf = pwlf.PiecewiseLinFit(x, y)
        >>> breaks = my_pwlf.fit(3)
        >>> slopes = my_pwlf.slopes()

        """
        y_hat = self.predict(self.fit_breaks)
        self.slopes = np.zeros(self.n_segments)
        for i in range(self.n_segments):
            self.slopes[i] = (y_hat[i+1]-y_hat[i]) / \
                        (self.fit_breaks[i+1]-self.fit_breaks[i])
        self.intercepts = y_hat[0:-1] - self.slopes*self.fit_breaks[0:-1]
        return self.slopes

    def p_values(self):
        r"""
        Calculate the p-values for each beta parameter.

        This calculates the p-values for the beta parameters under the
        assumption that your breakpoint locations are known. Section 2.4.2 of
        [2]_ defines how to calculate the p-value of individual parameters.
        This is really a marginal test since each parameter is dependent upon
        the other parameters.

        These values are typically compared to some confidence level alpha for
        significance. A 95% confidence level would have alpha = 0.05.

        Returns
        -------
        p : ndarray (1-D)
            p-values for each beta parameter where p-value[0] corresponds to
            beta[0] and so forth

        Raises
        ------
        ValueError
            You have probably not performed a fit yet.

        Notes
        -----
        This assumes that your breakpoint locations are exact! and does
        not consider the uncertainty with your breakpoint locations.

        See https://github.com/cjekel/piecewise_linear_fit_py/issues/14

        References
        ----------
        .. [2] Myers RH, Montgomery DC, Anderson-Cook CM. Response surface
            methodology . Hoboken. New Jersey: John Wiley & Sons, Inc.
            2009;20:38-44.

        Examples
        --------
        After performing a fit, one can calculate the p-value for each beta
        parameter

        >>> import pwlf
        >>> x = np.linspace(0.0, 1.0, 10)
        >>> y = np.random.random(10)
        >>> my_pwlf = pwlf.PiecewiseLinFit(x, y)
        >>> breaks = my_pwlf.fitfast(3)
        >>> x_new = np.linspace(0.0, 1.0, 100)
        >>> p = my_pwlf.p_values(x_new)

        see also examples/standard_errrors_and_p-values.py

        """
        # calculate the standard errors associated with each beta parameter
        # not that these standard errors and p-values are only meaningful if
        # you have specified the specific line segment end locations
        # at least for now...
        self.standard_errors()

        # calculate my t-value
        t = self.beta / self.se

        # degrees of freedom for t-distribution
        n = self.n_data
        try:
            k = len(self.beta) - 1
        except ValueError:
            errmsg = 'You do not have any beta parameters. You must perform' \
                     ' a fit before using standard_errors().'
            raise ValueError(errmsg)
        # calculate the p-values
        p = 2.0 * stats.t.sf(np.abs(t), df=n-k-1)
        return p
