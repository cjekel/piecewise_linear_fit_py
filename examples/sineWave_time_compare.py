# fit for a specified number of line segments
# you specify the number of line segments you want, the library does the rest

# import our libraries
import numpy as np
import matplotlib.pyplot as plt
import pwlf
from time import time

# generate sin wave data
x = np.linspace(0, 10, num=100)
y = np.sin(x * np.pi / 2)
# add noise to the data
y = np.random.normal(0, 0.05, 100) + y

# initialize piecewise linear fit with your x and y data
my_pwlf = pwlf.PiecewiseLinFit(x, y)

# fit the data for sixteen line segments
t0 = time()
res1 = my_pwlf.fit(16, disp=True)
t1 = time()
# i'm passing the argument disp=True to see the progress of the differential
# evolution so you can be sure the program isn't just hanging...

# Be patient! this one takes some time - It's a difficult problem
# using this differential evolution algo + bfgs can be over 500,000.0 function
# evaluations

# predict for the determined points
xHat1 = np.linspace(min(x), max(x), num=10000)
yHat1 = my_pwlf.predict(xHat1)

# fit the data for sixteen line segments
# using the default 50 number of multi starts
t2 = time()
res2 = my_pwlf.fitfast(16)  # this is equivalent to my_pwlf.fitfast(16,50)
t3 = time()

# predict for the determined points
xHat2 = np.linspace(min(x), max(x), num=10000)
yHat2 = my_pwlf.predict(xHat2)

print('Run time for differential_evolution', t1 - t0, 'seconds')
print('Run time for multi-start', t3 - t2, 'seconds')

# plot the results
plt.figure()
plt.plot(x, y, 'o')
plt.plot(xHat1, yHat1, '-', label='Diff. evolution')
plt.plot(xHat2, yHat2, '-', label='Multi start')
plt.legend()
plt.show()
