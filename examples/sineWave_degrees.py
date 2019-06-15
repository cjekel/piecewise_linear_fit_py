# import our libraries
import numpy as np
import matplotlib.pyplot as plt
import pwlf

# generate sin wave data
x = np.linspace(0, 10, num=100)
y = np.sin(x * np.pi / 2)
# add noise to the data
y = np.random.normal(0, 0.05, 100) + y

# initialize piecewise linear fit with your x and y data
# pwlf lets you fit continuous model for many degree polynomials
# degree=0 constant
# degree=1 linear (default)
# degree=2 quadratic 
my_pwlf_0 = pwlf.PiecewiseLinFit(x, y, degree=0)
my_pwlf_1 = pwlf.PiecewiseLinFit(x, y, degree=1)  # default
my_pwlf_2 = pwlf.PiecewiseLinFit(x, y, degree=2)

# fit the data for four line segments
res0 = my_pwlf_0.fitfast(4, pop=10)
res1 = my_pwlf_1.fitfast(4, pop=10)
res2 = my_pwlf_2.fitfast(4, pop=10)

# assembly
A0 = my_pwlf_0.assemble_regression_matrix(my_pwlf_0.fit_breaks, x)
A1 = my_pwlf_1.assemble_regression_matrix(my_pwlf_1.fit_breaks, x)
A2 = my_pwlf_2.assemble_regression_matrix(my_pwlf_2.fit_breaks, x)

print(A0.shape, my_pwlf_0.n_parameters, my_pwlf_0.n_segments)
print(A1.shape, my_pwlf_1.n_parameters, my_pwlf_1.n_segments)
print(A2.shape, my_pwlf_2.n_parameters, my_pwlf_2.n_segments)

# predict for the determined points
xHat = np.linspace(min(x), max(x), num=10000)
yHat0 = my_pwlf_0.predict(xHat)
yHat1 = my_pwlf_1.predict(xHat)
yHat2 = my_pwlf_2.predict(xHat)

# plot the results
plt.figure()
plt.plot(x, y, 'o', label='Data')
plt.plot(xHat, yHat0, '-', label='Degree 0')
plt.plot(xHat, yHat1, '--', label='Degree 1')
plt.plot(xHat, yHat2, ':', label='Degree 2')
plt.legend()
plt.show()
