import numpy as np
import matplotlib.pyplot as plt
import pwlf

# https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
y = np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])

# pwlf has two approaches to perform your fit:
# 1. You can fit for a specified number of line segments.
# 2. You can specify the x locations where the continuous piecewise lines
# should terminate.

# Approach 1
my_pwlf = pwlf.PiecewiseLinFit(x, y)
# breaks will return the end location of each line segment
breaks = my_pwlf.fit(2)
print(breaks)
# The gradient change point you asked for is at breaks[1]

x_hat = np.linspace(x.min(), x.max(), 100)
y_hat = my_pwlf.predict(x_hat)

plt.figure()
plt.plot(x, y, 'o')
plt.plot(x_hat, y_hat, '-')
plt.show()