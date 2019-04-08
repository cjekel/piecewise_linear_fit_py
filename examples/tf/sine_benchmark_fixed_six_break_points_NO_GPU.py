import numpy as np
import pwlf
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

breaks = np.array((0.0, 0.94, 2.96, 4.93, 7.02, 9.04, 10.0))

n = np.logspace(3, 7, num=15, dtype=np.int)
n_repeats = 10
# run_times = np.zeros((3, n.size, n_repeats))
run_times = np.load('bench_run_times/6_break_times.npy')
for i, n_data in enumerate(n):
    # set random seed
    np.random.seed(256)
    # generate sin wave data
    x = np.linspace(0, 10, num=n_data)
    y = np.sin(x * np.pi / 2)
    # add noise to the data
    y = np.random.normal(0, 0.05, size=n_data) + y
    for j in range(n_repeats):
        # PWLF TF NO GPU fit
        t4 = time()
        my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        ssr = my_pwlf.fit_with_breaks(breaks)
        t5 = time()
        run_times[2, i, j] = t5 - t4

np.save('bench_run_times/6_break_times.npy', run_times)
np.save('bench_run_times/n.npy', n)
