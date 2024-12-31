import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import pwlf

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 3, 4, 4.25, 3.75, 4, 5, 6, 7]

my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=1)

# perform initial fit
breaks = my_pwlf.fit(2)


def my_fun(beta):
    # assing variables to the pwlf object
    my_pwlf.beta[0] = beta[0]  # first line offset
    my_pwlf.beta[1] = beta[1]  # first line slope
    my_pwlf.beta[2] = -1*beta[1]
    my_pwlf.fit_breaks[1] = beta[2]  # breakpoint
    # generate predictions
    y_temp = my_pwlf.predict(my_pwlf.x_data)
    # compute ssr
    e = y_temp - my_pwlf.y_data
    return np.dot(e, e)


bounds = np.zeros((3, 2))
# first line offset
bounds[0, 0] = -100.0  # lower bound
bounds[0, 1] = 100.0  # upper bound
# first line slope
bounds[1, 0] = -100.0  # lower bound
bounds[1, 1] = 100.0  # upper bound
# breakpont
bounds[2, 0] = 2.  # lower bound
bounds[2, 1] = 6.  # upper bound

res = differential_evolution(my_fun, bounds, maxiter=1000, popsize=30,
                             disp=True)

# assign optimum to my_pwlf object
my_fun(res.x)

# generate predictions
x_hat = np.linspace(min(x), max(x), 1000)
y_hat = my_pwlf.predict(x_hat)

plt.figure()
plt.plot(x, y, 'o')
plt.plot(x_hat, y_hat)
plt.show()
