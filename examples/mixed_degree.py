import numpy as np
import matplotlib.pyplot as plt
import pwlf

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [1, 2, 3, 4, 4.25, 3.75, 4, 5, 6, 7]

degree_list = [1, 0, 1]

my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=degree_list)

breaks = my_pwlf.fit(3)

# generate predictions
x_hat = np.linspace(min(x), max(x), 1000)
y_hat = my_pwlf.predict(x_hat)

plt.figure()
plt.plot(x, y, 'o')
plt.plot(x_hat, y_hat)
plt.show()
