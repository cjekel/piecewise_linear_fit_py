from __future__ import print_function
import numpy as np
import pwlf

# generate a true piecewise linear data
np.random.seed(5)
n_data = 100
x = np.linspace(0, 1, num=n_data)
y = np.random.random(n_data)
my_pwlf = pwlf.PiecewiseLinFit(x, y)
true_beta = np.random.normal(size=5)
true_breaks = np.array([0.0, 0.2, 0.5, 0.75, 1.0])
y = my_pwlf.predict(x, beta=true_beta, breaks=true_breaks)

# initialize piecewise linear fit with your x and y data
my_pwlf = pwlf.PiecewiseLinFit(x, y)
# fit the data for our specified line segment locations
res = my_pwlf.fitfast(4, pop=100)

# calculate the non-linear standard errors
se, A = my_pwlf.standard_errors(method='non-linear', step_size=1e-4)

# calculate p-values
p = my_pwlf.p_values(method='non-linear', step_size=1e-4)

parameters = np.concatenate((my_pwlf.beta, my_pwlf.fit_breaks[1:-1]))

header = ['Parmater type', 'Parameter value', 'Standard error', 't',
          'P > |t| (p-value)']
print(*header, sep=' & ')
values = np.zeros((parameters.size, 5), dtype=np.object_)
values[:, 1] = np.around(parameters, decimals=3)
values[:, 2] = np.around(se, decimals=3)
values[:, 3] = np.around(parameters / se, decimals=3)
values[:, 4] = np.around(p, decimals=3)

for i, row in enumerate(values):
    if i < my_pwlf.beta.size:
        row[0] = 'Beta'
        print(*row, sep=' & ')
    else:
        row[0] = 'Breakpoint'
        print(*row, sep=' & ')
