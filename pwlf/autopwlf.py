import re
import pwlf
import numpy as np
from sympy import Symbol
from sympy.utilities import lambdify
from collections import defaultdict
from scipy.optimize import rosen, differential_evolution
from utils.io import *

class AutoPiecewiseLinFit(object):

    def __init__(self, x, y) -> None:
        r"""
        Automatically determine number of segments to do piecewise linear fit,
        and apply bounds to speed up convergence.
        Parameters
        ----------
        x : array_like
            The x or independent data point locations as list or 1 dimensional
            numpy array.
        y : array_like
            The y or dependent data point locations as list or 1 dimensional
            numpy array.
        Attributes
        ----------
        n : int
            number of segment 
        bounds : array_like
            Bounds for each breakpoint location within the optimization. This
            should have the shape of (n_segments, 2).
        slopes : ndarray(1-D)
            The slope of each ling segment as a 1-D numpy array.
        intercepts : ndarray (1-D)
            The y-intercept of each line segment as a 1-D numpy array.
        my_pwlf : PiecewiseLinFit
            Piecewise linear fit model, which is callable.
            source code https://github.com/cjekel/piecewise_linear_fit_py/blob/master/pwlf/pwlf.py
        Methods
        -------
        fit_curve_and_guess(degree=16, lb=10, ub=10)
            Fit a polynomial function first then extract roots
        piecewise_fit(lin_num: int = None, use_bound=True)
            Fit a continuous piecewise linear function for a specified number
            of line segments.
        parse_model()
            Format a lambda expression y = k*x + b and return it
        cal_slopes()
            Calculate the slopes of the lines after a piecewise linear
            function has been fitted.
        model_error()
            Evaluate model piecewise, which format as 
            (start, end): [mean of vertical distance, mean of y_data value, number of y_data on this segment]
        Examples
        --------
        Initialize for x, y data
        >>> my_pwlf = AutoPiecewiseLinFit(x, y)
        Guess number of segment first
        >>> n = my_pwlf.fit_curve_and_guess()
        Fit model with bounds specific
        >>> fit_1.piecewise_fit(n, use_bound=True)
        Calculate slopes and intercepts
        >>> fit_1.cal_slopes()
        Evalute model
        >>> fit_1.model_error()
        """


        if not all(isinstance(n, (int, float)) for n in x) or \
           not all(isinstance(n, (int, float)) for n in y):
            raise ValueError("input x and y should by int or float")

        x, y = self._switch_to_np_array(x), self._switch_to_np_array(y)
        self.x_data, self.y_data = x, y

    
    @staticmethod
    def _switch_to_np_array(input_):
        if isinstance(input_, np.ndarray) is False:
            input_ = np.array(input_)
        return input_

    
    @staticmethod
    def _unpack_expr(func):
        r"""
        Extract piecewise model's parameters.
        Parameters
        ----------
        func : callable object
            Model function
        
        Returns
        -------
        k : int
            Slope
        b : int
            Intercept
        """
        text = func.__doc__
        l = re.findall(r'return .*\n', text)[0]
        l = re.split(r' |\n', l)[1:-1]
        if len(l) == 3:
            if '*x' in l[0]:
                k, b = eval(l[0].split('*')[0]), eval(l[1]+l[2])
            elif '*' in l[2]:
                k, b = eval(l[1]+l[2].split('*')[0]), eval(l[0])
        elif len(l) == 4:
            if '*x' in l[1]:
                k, b = eval(l[0]+l[1].split('*')[0]), eval(l[2]+l[3])
            elif '*x' in l[3]:
                k, b = eval(l[2]+l[3].split('*')[0]), eval(l[0]+l[1])
        return k, b

    def fit_curve_and_guess(self, degree=16, lb=10, ub=10) -> int:
        r"""
        Guess lines number by using curve fit then extract roots.
        Then initize bounds with roots-lb and roots+ub, which can
        speed up convergence.
        Parameters
        ----------
        degree : int
            Polynomial degree to fit curve 
        lb : int
            Lower bound
        ub : int
            Up bound
        Returns
        -------
        guess : int
            The guess of piecewise lines number.
        """

        coeff = np.polyfit(self.x_data, self.y_data, degree)
        f = np.poly1d(coeff)
        f_d = np.polyder(f, 1)
        xroot = np.roots(f_d)
        xroot = list(filter(lambda x : x.imag == 0, xroot))
        xroot = list(filter(lambda x : x >= self.x_data.min() and x <= self.x_data.max(), xroot))
        self.n = len(xroot)+1
        self.bounds = np.zeros([self.n-1, 2])
        for i, e in enumerate(xroot):
            self.bounds[i][0] = e.real-lb
            self.bounds[i][1] = e.real+ub
        return self.n

    def piecewise_fit(self, lin_num: int = None, use_bound=True) -> None:
        r"""
        Use pwlf.fit() to fit model.
        Parameters
        ----------
        lin_num : None, or int
            Default lin_num=None, automatically infer segment number. 
        use_bound : bool, optional
            Defalut True, use bounds to speed up convergence
        """
        my_pwlf = pwlf.PiecewiseLinFit(self.x_data, self.y_data)
        if use_bound == True:
            if lin_num is None:
                my_pwlf.fit(self.n, bounds=self.bounds, workers=-1, updating='deferred')
            else:
                my_pwlf.fit(lin_num, bounds=self.bounds, workers=-1, updating='deferred')
        else:
            if lin_num is None:
                my_pwlf.fit(self.n, workers=-1, updating='deferred')
            else:
                my_pwlf.fit(lin_num, workers=-1, updating='deferred')
        break_predict = my_pwlf.predict(my_pwlf.fit_breaks)
        self.slopes = my_pwlf.calc_slopes()
        self.intercepts = break_predict[0:-1] - self.slopes * my_pwlf.fit_breaks[0:-1]
        self.my_pwlf = my_pwlf
        
    def parse_model(self) -> dict:
        r"""
        Format a lambda expression y = k*x + b and return it
        
        Returns
        -------
        model : dict
            Mapping (start point, end point) to linear function
        """
        x = Symbol('x')
        breaks = self.my_pwlf.fit_breaks
        model = dict()
        for i, (k, b) in enumerate(zip(self.slopes, self.intercepts)):
            expr = k * x + b
            f = lambdify(x, expr, 'numpy')
            model[(breaks[i], breaks[i+1])] = f
        return model
    
    def cal_slopes(self) -> defaultdict:
        r"""
        Calculate the slopes and intercepts of the lines after a piecewise linear
        function has been fitted.
        Returns
        -------
        model_params : defaultdict
            Mapping (start point, end point) to [slope, intercept]
        """
        model = self.parse_model()
        model_params = defaultdict(list)
        for key, val in model.items():
            k, v = self._unpack_expr(val)
            model_params[key].extend([k, v])
        return model_params

    def model_error(self) -> defaultdict:
        r"""
        Evaluate model piecewise, which format as 
        (start, end): [mean of vertical distance, mean of y_data value, number of y_data on this segment]
        Returns
        -------
        model_error_piecewise : defaultdict
        """
        def vert_dist(func, x, y):
            k, b = self._unpack_expr(func)
            d = abs(k*x - y + b) / np.sqrt(k**2+1)
            return d

        model = self.parse_model()
        model_error_piecewise = defaultdict(list)
        tmp = defaultdict(list)
        for x, y in zip(self.x_data, self.y_data):
            for k, func in model.items():
                if k[0] <= x and x < k[1]:
                    model_error_piecewise[k].append(vert_dist(func, x, y)**2)
                    tmp[k].append(y); break
        model_error_piecewise = {k:[np.mean(v), np.mean(tmp[k]), len(v)] for k, v in model_error_piecewise.items()}
        return model_error_piecewise
    
    def piecewise_fit_fast(self):
        # TODO : Further improve computation speed by restricting break points to int
        return