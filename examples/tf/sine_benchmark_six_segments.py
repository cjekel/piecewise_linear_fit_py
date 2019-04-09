import numpy as np
import matplotlib.pyplot as plt
import pwlf
from time import time

# set random seed
np.random.seed(256)
breaks = np.array((0.0, 0.94, 2.96, 4.93, 7.02, 9.04, 10.0))

n_data = int(1e6)

# generate sin wave data
x = np.linspace(0, 10, num=n_data)
y = np.sin(x * np.pi / 2)
# add noise to the data
y = np.random.normal(0, 0.05, size=n_data) + y

t0 = time()
# initialize piecewise linear fit with your x and y data
my_pwlf = pwlf.PiecewiseLinFitTF(x, y)

# # fit the data for four line segments
# res = my_pwlf.fit(16)
# breaks = my_pwlf.fit(6)

ssr = my_pwlf.fit_with_breaks(breaks)
t1 = time()
print('run time:', t1 - t0)
# predict for the determined points
xHat = np.linspace(min(x), max(x), num=10000)
yHat = my_pwlf.predict(xHat)

# plot the results
plt.figure()
plt.plot(x, y, 'o')
plt.plot(xHat, yHat, '-')
plt.show()
