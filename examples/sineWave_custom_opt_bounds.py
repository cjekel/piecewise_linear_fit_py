# fit for a specified number of line segments
# you specify the number of line segments you want, the library does the rest

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
my_pwlf = pwlf.PiecewiseLinFit(x, y)

# define custom bounds for the interior break points
n_segments = 4
bounds = np.zeros((n_segments-1, 2))
# first lower and upper bound
bounds[0, 0] = 0.0
bounds[0, 1] = 3.5
# second lower and upper bound
bounds[1, 0] = 3.0
bounds[1, 1] = 7.0
# third lower and upper bound
bounds[2, 0] = 6.0
bounds[2, 1] = 10.0
res = my_pwlf.fit(n_segments, bounds=bounds)

# predict for the determined points
xHat = np.linspace(min(x), max(x), num=10000)
yHat = my_pwlf.predict(xHat)

# plot the results
plt.figure()
plt.plot(x, y, 'o')
plt.plot(xHat, yHat, '-')
plt.show()
