import re
import pwlf
import numpy as np
from sympy import Symbol
from sympy.utilities import lambdify
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import rosen, differential_evolution

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

if __name__ == "__main__":
    ## complex dataset 
    # x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
    # y1 = [50.69, 50.63, 50.01, 50.22, 49.63, 47.62, 48.5, 48.71, 48.35, 49.05, 48.15, 49.05, 49.23, 49.85, 49.76, 49.34, 48.02, 48.58, 49.01, 49.7, 51.37, 51.82, 51.85, 50.9, 51.0, 50.77, 49.21, 51.35, 51.71, 51.62, 51.71, 52.76, 51.2, 50.22, 51.07, 51.46, 52.06, 51.76, 51.22, 50.83, 50.02, 49.81, 49.8, 50.24, 49.47, 49.73, 48.7, 46.76, 45.48, 44.39, 44.7, 44.45, 42.15, 39.45, 42.58, 44.51, 46.17, 44.81, 45.53, 45.76, 45.77, 45.18, 45.05, 45.01, 46.23, 46.8, 47.1, 47.39, 46.86, 47.18, 45.9, 45.7, 45.75, 42.1, 41.1]
    # y2 = [6.02, 6.0, 5.98, 5.99, 6.02, 5.95, 6.01, 6.02, 6.0, 5.99, 5.96, 5.96, 5.99, 6.0, 6.02, 6.05, 6.04, 6.08, 6.13, 6.2, 6.33, 6.41, 6.51, 6.53, 6.54, 6.54, 6.52, 6.57, 6.59, 6.63, 6.64, 6.7, 6.7, 6.69, 6.72, 6.73, 6.74, 6.75, 6.73, 6.74, 6.73, 6.71, 6.71, 6.73, 6.73, 6.73, 6.71, 6.65, 6.61, 6.59, 6.58, 6.57, 6.56, 6.57, 6.57, 6.59, 6.7, 6.7, 6.73, 6.76, 6.78, 6.77, 6.78, 6.78, 6.79, 6.76, 6.74, 6.72, 6.72, 6.73, 6.72, 6.66, 6.69, 6.57, 6.47]
    """ a simple dataset """
    x = np.arange(0, 4*np.pi, 0.1)
    y = np.sin(x)
    """ 
    coeff is parameter of polyfit model. 
    One degree polynomial model y = a_1*x + a_0
    >>> x = np.linspace(-5, 5, 100)
    >>> y = 4 * x + 1.5
    >>> noise_y = y + np.random.randn(y.shape[-1]) * 2.5
    >>> p = plt.plot(x, noise_y, 'rx')
    >>> p = plt.plot(x, y, 'b:')
    >>> coeff = polyfit(x, noise_y, 1)
    >>> print(coeff)
    >>> [3.93921315  1.59379469] -> [a_1, a_0]
    """
    coeff = np.polyfit(x, y, deg=9)
    """
    poly1d assemble coeff to polynomial model
    >>> f = poly1d(coeff)
    >>> print(f)
    >>> 3.939 x + 1.594
    """
    f = np.poly1d(coeff)
    """np.polyder is a tool that return the derivative of the specified order of a polynomial."""
    f_d = np.polyder(f, 1)
    """
    np.roots return the roots of a polynomial with coefficients given in p.
    here xroot are [-27.02577916   8.46877068   4.36903767]
    when increasing degree to deg=9, results contain imaginary roots:
    [12.83124078+1.06520938j 12.83124078-1.06520938j 10.98184655+0.j
    7.86313591+0.j          4.70268776+0.j          1.58653459+0.j
    -0.28007017+1.04758677j -0.28007017-1.04758677j] 
    """
    xroot = np.roots(f_d)
    """
    but we just need xroot between x.min() and x.max()
    after filter, xroot are [8.468770676220245, 4.369037674890286]
    """
    min, max = x.min(), x.max()
    xroot = list(filter(lambda x : (x >= min and x < max), xroot))
    yroot = np.sin(xroot)
    fit_1 = AutoPiecewiseLinFit(x, y)
    n_1 = fit_1.fit_curve_and_guess()
    fit_1.piecewise_fit(n_1)
    model = fit_1.parse_model()
    x_hat = [key for key, _ in model.items() for key in key]
    y_hat = [value(key) for key, value in model.items() for key in key]
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    axes.plot(x,y)
    axes.scatter(xroot, yroot, color='red')
    axes.plot(x_hat, y_hat, color='green')
    plt.savefig('polyfit_deg_9.png')