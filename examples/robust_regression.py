# import our libraries
import numpy as np
import matplotlib.pyplot as plt
import pwlf
from scipy.optimize import least_squares

# generate sin wave data
np.random.seed(1213)
x = np.linspace(0, 10, num=100)
y = np.sin(x * np.pi / 2)
# add noise to the data
y = np.random.normal(0, 0.5, 100) + y

# initialize piecewise linear fit with your x and y data
my_pwlf = pwlf.PiecewiseLinFit(x, y)
num_of_line_segments = 5  # change this value
num_in_breaks = num_of_line_segments - 1
n_beta = num_of_line_segments + 1

def my_fun(breaks_and_beta):
    """
    Returns the residuals given an array of breaks and beta
    """
    inner_breaks = breaks_and_beta[:num_in_breaks]  # break point locations
    # breaks code is taken from pwlf...
    breaks = np.zeros(len(inner_breaks) + 2)
    breaks[1:-1] = inner_breaks.copy()
    breaks[0] = my_pwlf.break_0  # smallest x value
    breaks[-1] = my_pwlf.break_n  # largest x value
    beta = breaks_and_beta[num_in_breaks:]  # beta paramters (slopes and int)
    A = my_pwlf.assemble_regression_matrix(breaks,
                                           my_pwlf.x_data)
    y_hat = np.dot(A, beta)
    resids = y_hat - my_pwlf.y_data
    return resids


# fit the data three line segments and use this result as a starting point
res = my_pwlf.fitfast(num_of_line_segments)
xhat = np.linspace(0, 10, 1000)
yhat_ols = my_pwlf.predict(xhat)  # initial guess

breaks_and_beta_guess = np.zeros(num_in_breaks + n_beta)
breaks_and_beta_guess[:num_in_breaks] = res[1:num_of_line_segments]
breaks_and_beta_guess[num_in_breaks:] = my_pwlf.beta

# use the result from pwlf to start a robust regresion
# notes on soft_l1: from documentation
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
# ‘soft_l1’ : rho(z) = 2 * ((1 + z)**0.5 - 1). The smooth approximation of l1
# (absolute value) loss. Usually a good choice for robust least squares.
result = least_squares(my_fun,
                       breaks_and_beta_guess,  # initial guess
                       loss='soft_l1',
                       f_scale=0.1)  # inlier residuals less than 0.1

# put the results back into my_pwlf
breaks = np.zeros(num_of_line_segments+1)
breaks[0] = my_pwlf.break_0
breaks[-1] = my_pwlf.break_n
breaks[1:num_in_breaks+1] = result.x[:num_in_breaks]
my_pwlf.fit_breaks = breaks
beta = result.x[num_in_breaks:]
my_pwlf.beta = beta

yhat_robo = my_pwlf.predict(xhat)

plt.figure()
plt.plot(x, y, 'o')
plt.plot(xhat, yhat_ols, '-', label='OLS')
plt.plot(xhat, yhat_robo, '-', label='Robust Reg.')
plt.legend()
plt.show()
