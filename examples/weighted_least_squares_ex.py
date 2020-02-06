from time import time
import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import pwlf
t0 = time()
np.random.seed(123)
n = 100
n_data_sets = 100
n_segments = 6
# generate sine data
x = np.linspace(0, 10, n)
y = np.zeros((n_data_sets, n))
for i in range(n_data_sets):
    y[i] = np.sin(x * np.pi / 2)
    # add noise to the data
    y[i] = np.random.normal(0, 0.05, 100) + y[i]
X = np.tile(x, n_data_sets)

# perform an ordinary pwlf fit to the entire data
my_pwlf = pwlf.PiecewiseLinFit(X.flatten(), y.flatten())
my_pwlf.fit(n_segments)

# compute the standard deviation in y
y_std = np.tile(np.std(y, axis=0), n_data_sets)
# set the weights to be one over the standard deviation
weights = 1.0 / y_std

# perform a weighted least squares to the data
my_pwlf_w = pwlf.PiecewiseLinFit(X.flatten(), y.flatten(), weights=weights)
my_pwlf_w.fit(n_segments)


# compare the fits
xhat = np.linsapce(0, 10, 1000)
yhat = my_pwlf.predict(xhat)
yhat_w = my_pwlf_w.predict(xhat)
t1 = time()
print('Runtime:', t1-t0)
plt.figure()
plt.plot(X.flatten(), y.flatten(), '.')
plt.plot(xhat, yhat, '-', label='Ordinary LS')
plt.plot(xhat, yhat_w, '-', label='Weighted LS')
plt.legend()
plt.show()
