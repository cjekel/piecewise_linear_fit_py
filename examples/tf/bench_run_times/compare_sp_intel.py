import numpy as np
import matplotlib.pyplot as plt

# factor for 90% coverage with 90% confidence using Normal distribution
# with 10 samples from table XII in [1]
# [1] Montgomery, D. C., & Runger, G. C. (2014). Applied statistics and
# probability for engineers. Sixth edition. John Wiley & Sons.
k = 2.535



run_times = np.load('amd_fx_8350_titanXP/6_break_times.npy')
n = np.load('amd_fx_8350_titanXP/n.npy')
run_times1 = np.load('amd_fx_8350_titanXP_sp/6_break_times.npy')



run_times_means = run_times.mean(axis=2)
run_times_stds = run_times.std(axis=2, ddof=1)
run_times_means1 = run_times1.mean(axis=2)
run_times_stds1 = run_times1.std(axis=2, ddof=1)


plt.figure()
plt.grid()
plt.errorbar(n, run_times_means[0], yerr=k*run_times_stds[0], capsize=2.0, label='SciPy')
# plt.errorbar(n, run_times_means[1], yerr=k*run_times_stds[1], capsize=2.0, label='TF GPU')
plt.errorbar(n, run_times_means[2], yerr=k*run_times_stds[2], capsize=2.0, label='TF CPU')

plt.errorbar(n, run_times_means1[0], yerr=k*run_times_stds1[0], capsize=2.0, label='SciPy sp')
# plt.errorbar(n, run_times_means1[1], yerr=k*run_times_stds1[1], capsize=2.0, label='TF GPU sp')
plt.errorbar(n, run_times_means1[2], yerr=k*run_times_stds1[2], capsize=2.0, label='TF CPU sp')

plt.xlabel('Number of data points')
plt.ylabel('Run time (seconds, Lower is better)')
plt.semilogx()
plt.legend()

plt.show()
