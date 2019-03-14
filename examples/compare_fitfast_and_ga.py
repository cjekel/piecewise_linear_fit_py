# import our libraries
import numpy as np
import matplotlib.pyplot as plt
import pwlf
from time import time

# generate sin wave data
n = 100
x = np.linspace(0, 10, num=n)
y = np.sin(x * np.pi / 2)
# add noise to the data
y = np.random.normal(0, 0.05, n) + y

# initialize piecewise linear fit with your x and y data
my_pwlf = pwlf.PiecewiseLinFit(x, y)

number_of_line_segments = 16
t0 = time()
res0 = my_pwlf.fitfast(number_of_line_segments)
t1 = time()
print('run time fitfast:', t1-t0, '(s)')
print('ssr:', my_pwlf.ssr)

t2 = time()
my_pwlf.use_custom_opt(number_of_line_segments)
total_set = set(my_pwlf.x_data)
total_set.remove(x.min())
total_set.remove(x.max())
pop, hof, stats = pwlf.genetic_algorithm(total_set, my_pwlf.nVar,
                                         my_pwlf.fit_with_breaks, ngen=20,
                                         mu=125, lam=250, cxpb=0.7, mutpb=0.2,
                                         tournsize=5, verbose=False)
ssr = my_pwlf.fit_with_breaks(list(hof[0]))
t3 = time()
print('run time ga:', t3-t2, '(s)')
print('ssr:', ssr)