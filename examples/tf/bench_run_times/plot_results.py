import numpy as np
import matplotlib.pyplot as plt

# factor for 90% coverage with 90% confidence using Normal distribution
# with 10 samples from table XII in [1]
# [1] Montgomery, D. C., & Runger, G. C. (2014). Applied statistics and
# probability for engineers. Sixth edition. John Wiley & Sons.
k = 2.535


def load_and_plot_data(run_strs, n_strs, j=1):

    run_times = np.load(run_strs)
    n = np.load(n_strs)



    run_times_means = run_times.mean(axis=2)
    run_times_stds = run_times.std(axis=2, ddof=1)



    plt.figure()
    plt.grid()
    plt.errorbar(n, run_times_means[0], yerr=k*run_times_stds[0], capsize=2.0, label='Numpy')
    # plt.errorbar(n, run_times_means[1], yerr=k*run_times_stds[1], capsize=2.0, label='TF GPU')
    plt.errorbar(n, run_times_means[j], yerr=k*run_times_stds[j], capsize=2.0, label='TF CPU')

    plt.xlabel('Number of data points')
    plt.ylabel('Run time (seconds, Lower is better)')
    plt.semilogx()

    plt.figure()
    plt.grid()

    plt.errorbar(n[1:], run_times_means[0, 1:] - run_times_means[0, 1:], yerr=(k*run_times_stds[0, 1:]), capsize=2.0, label='Numpy')
    # plt.errorbar(n[1:], run_times_means[1, 1:] - run_times_means[0, 1:], yerr=(k*run_times_stds[1, 1:]), capsize=2.0, label='TF GPU')
    plt.errorbar(n[1:], run_times_means[j, 1:] - run_times_means[0, 1:], yerr=(k*run_times_stds[j, 1:]), capsize=2.0, label='TF CPU')

    plt.xlabel('Number of data points')
    plt.ylabel('Run time difference (Lower is better)')
    plt.semilogx()


    plt.figure()
    plt.grid()

    plt.errorbar(n[1:], run_times_means[0, 1:]/run_times_means[0, 1:], yerr=(k*run_times_stds[0, 1:])/run_times_means[0, 1:], capsize=2.0, label='Numpy')
    # plt.errorbar(n[1:], run_times_means[1, 1:]/run_times_means[0, 1:], yerr=(k*run_times_stds[1, 1:])/run_times_means[0, 1:], capsize=2.0, label='TF GPU')
    plt.errorbar(n[1:], run_times_means[j, 1:]/run_times_means[0, 1:], yerr=(k*run_times_stds[j, 1:])/run_times_means[0, 1:], capsize=2.0, label='TF CPU')
    plt.xlabel('Number of data points')
    plt.ylabel('Run time relative to Numpy (Lower is better)')
    plt.semilogx()


    plt.figure()
    plt.grid()
    plt.errorbar(n[1:], run_times_means[0, 1:]/run_times_means[1, 1:], yerr=(k*run_times_stds[0, 1:])/run_times_means[1, 1:], capsize=2.0, label='Numpy')
    # plt.errorbar(n[1:], run_times_means[1, 1:]/run_times_means[1, 1:], yerr=(k*run_times_stds[1, 1:])/run_times_means[1, 1:], capsize=2.0, label='TF GPU')
    plt.errorbar(n[1:], run_times_means[j, 1:]/run_times_means[1, 1:], yerr=(k*run_times_stds[j, 1:])/run_times_means[1, 1:], capsize=2.0, label='TF CPU')
    plt.xlabel('Number of data points')
    plt.ylabel('Run time relative to TF GPU (Lower is better)')
    plt.semilogx()

    plt.show()


load_and_plot_data('amd_fx_8350_titanXP/6_break_times.npy', 'amd_fx_8350_titanXP/n.npy', j=2)
load_and_plot_data('amd_ryzen_2700x/6_break_times.npy', 'amd_ryzen_2700x/n.npy')
load_and_plot_data('intel_i5_6300u/6_break_times.npy', 'intel_i5_6300u/n.npy')
