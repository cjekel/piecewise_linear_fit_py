import numpy as np
import matplotlib.pyplot as plt

# factor for 90% coverage with 90% confidence using Normal distribution
# with 10 samples from table XII in [1]
# [1] Montgomery, D. C., & Runger, G. C. (2014). Applied statistics and
# probability for engineers. Sixth edition. John Wiley & Sons.
k = 2.535

amd_fx_run_times = np.load('amd_fx_8350_titanXP/6_break_times.npy')
n = np.load('amd_fx_8350_titanXP/n.npy')
amd_fx_run_times_means = amd_fx_run_times.mean(axis=2)
amd_fx_run_times_stds = amd_fx_run_times.std(axis=2, ddof=1)



plt.figure()
plt.grid()
plt.errorbar(n, amd_fx_run_times_means[0], yerr=k*amd_fx_run_times_stds[0], capsize=2.0, label='Numpy')
plt.errorbar(n, amd_fx_run_times_means[1], yerr=k*amd_fx_run_times_stds[1], capsize=2.0, label='TF GPU')
plt.xlabel('Number of data points')
plt.ylabel('Run time (seconds, Lower is better)')
plt.semilogx()

plt.figure()
plt.grid()

plt.errorbar(n[1:], amd_fx_run_times_means[0,1:] - amd_fx_run_times_means[0,1:], yerr=(k*amd_fx_run_times_stds[0,1:]), capsize=2.0, label='Numpy')
plt.errorbar(n[1:], amd_fx_run_times_means[1,1:] - amd_fx_run_times_means[0,1:], yerr=(k*amd_fx_run_times_stds[1,1:]), capsize=2.0, label='TF GPU')
plt.xlabel('Number of data points')
plt.ylabel('Run time difference (Lower is better)')
plt.semilogx()


plt.figure()
plt.grid()

plt.errorbar(n[1:], amd_fx_run_times_means[0,1:]/amd_fx_run_times_means[0,1:], yerr=(k*amd_fx_run_times_stds[0,1:])/amd_fx_run_times_means[0,1:], capsize=2.0, label='Numpy')
plt.errorbar(n[1:], amd_fx_run_times_means[1,1:]/amd_fx_run_times_means[0,1:], yerr=(k*amd_fx_run_times_stds[1,1:])/amd_fx_run_times_means[0,1:], capsize=2.0, label='TF GPU')
plt.xlabel('Number of data points')
plt.ylabel('Run time relative to Numpy (Lower is better)')
plt.semilogx()


plt.figure()
plt.grid()
plt.errorbar(n[1:], amd_fx_run_times_means[0,1:]/amd_fx_run_times_means[1,1:], yerr=(k*amd_fx_run_times_stds[0,1:])/amd_fx_run_times_means[1,1:], capsize=2.0, label='Numpy')
plt.errorbar(n[1:], amd_fx_run_times_means[1,1:]/amd_fx_run_times_means[1,1:], yerr=(k*amd_fx_run_times_stds[1,1:])/amd_fx_run_times_means[1,1:], capsize=2.0, label='TF GPU')
plt.xlabel('Number of data points')
plt.ylabel('Run time relative to TF GPU (Lower is better)')
plt.semilogx()

plt.show()
