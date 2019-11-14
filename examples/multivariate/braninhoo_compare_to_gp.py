import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import GPy
import pwlf
from pyDOE import lhs

# Branin-Hoo paramters
a = 1.
b = 5.1 / (4*np.pi**2)
c = 5/np.pi
r = 6.
s = 10.
t = 1. / (8.*np.pi)


def braninhoo(x):
    A = a*(x[:, 1] - b*x[:, 0]**2+c*x[:, 0]-r)**2
    B = s*(1-t)*np.cos(x[:, 0])+s
    return A + B


n_samp = 7
np.random.seed(1231)
lhd = lhs(2, samples=n_samp, criterion='maximin')
lb = 0.0
ub = 10.0
x = (ub - lb) * lhd + lb
y = braninhoo(x)

n_segments = 2
my_mv_model = pwlf.PiecewiseMultivariate(x, y, n_segments, degree=1,
                                         multivariate_degree=2)
my_mv_model.fit()
print('pwlf fit SSR', my_mv_model.ssr)

# fit gaussian process
kernel = GPy.kern.Exponential(input_dim=2)
m = GPy.models.GPRegression(x, y.reshape(-1, 1), kernel)
m.optimize_restarts(num_restarts=10)
y_mean, y_std = m.predict(x)

e = y_mean - y.reshape(-1, 1)
ssr_gp = np.dot(e.T, e)
print('GP fit SSR:', ssr_gp[0, 0])

n_contours = 100
xt = np.linspace(lb, ub, n_contours)
x1, x2 = np.meshgrid(xt, xt)
x_hat = np.array([x1.flatten(), x2.flatten()]).T
y_true = braninhoo(x_hat)
y_hat = my_mv_model.predict(x_hat)
y_hat_gp, y_hat_gp_std = m.predict(x_hat)

# compute SSR
e_pwlf = y_true - y_hat
ssr_pwlf = np.dot(e_pwlf, e_pwlf)
e_gp = y_hat_gp - y_true.reshape(-1, 1)
ssr_gp = np.dot(e_gp.T, e_gp)

print('Validation pwlf SSR:', ssr_pwlf)
print('Valdiation gp SSR:', ssr_gp[0, 0])

fig, axs = plt.subplots(1, 3, figsize=(16, 4))
axs[0].set_title('Branin-Hoo function')
cf0 = axs[0].contourf(x1, x2, y_true.reshape(n_contours, n_contours))
axs[0].plot(x[:, 0], x[:, 1], 'ok', label='DOE')
fig.colorbar(cf0, ax=axs[0])
# axs.set_colorbar()
# fig.colorbar()
axs[0].legend()
axs[0].set_ylabel(r'$x_2$')
axs[0].set_xlabel(r'$x_1$')

axs[1].set_title('pwlf multivariate model')
cf1 = axs[1].contourf(x1, x2, y_hat.reshape(n_contours, n_contours))
axs[1].plot(x[:, 0], x[:, 1], 'ok', label='DOE')
# axs[1].set_colorbar()
fig.colorbar(cf1, ax=axs[1])
axs[1].legend()
axs[1].set_ylabel(r'$x_2$')
axs[1].set_xlabel(r'$x_1$')

axs[2].set_title('GP model')
cf2 = axs[2].contourf(x1, x2, y_hat_gp.reshape(n_contours, n_contours))
axs[2].plot(x[:, 0], x[:, 1], 'ok', label='DOE')
fig.colorbar(cf2, ax=axs[2])

# axs[2].set_colorbar()
# plt.colorbar()
axs[2].legend()
axs[2].set_ylabel(r'$x_2$')
axs[2].set_xlabel(r'$x_1$')

fig.savefig('multivariate_pwlf_vs_gp.png', dpi=300, bbox_inches='tight')

plt.figure()
plt.title('Branin-Hoo function')
plt.contourf(x1, x2, y_true.reshape(n_contours, n_contours))
plt.plot(x[:, 0], x[:, 1], 'ok', label='DOE')
plt.colorbar()
plt.legend()
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')

plt.figure()
plt.title('pwlf multivariate model')
plt.contourf(x1, x2, y_hat.reshape(n_contours, n_contours))
plt.plot(x[:, 0], x[:, 1], 'ok', label='DOE')
plt.colorbar()
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')
plt.legend()

plt.figure()
plt.title('GP model')
plt.contourf(x1, x2, y_hat_gp.reshape(n_contours, n_contours))
plt.plot(x[:, 0], x[:, 1], 'ok', label='DOE')
plt.colorbar()
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')
plt.legend()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('pwlf multivariate model')
# Plot the surface.
surf = ax.plot_surface(x1, x2, y_hat.reshape(n_contours, n_contours),
                       cmap=plt.cm.plasma,
                       linewidth=0,
                       antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5)
ax.set_ylabel(r'$x_2$')
ax.set_xlabel(r'$x_1$')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title('GP model')
# Plot the surface.
surf = ax.plot_surface(x1, x2, y_hat_gp.reshape(n_contours, n_contours),
                       cmap=plt.cm.plasma,
                       linewidth=0,
                       antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5)
ax.set_ylabel(r'$x_2$')
ax.set_xlabel(r'$x_1$')

# qq plot
plt.figure()
plt.plot(y_true, y_hat, '.')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '-k',
         label='best possible')
plt.legend()
plt.xlabel('True response')
plt.ylabel('Multivariate pwlf model')
plt.show()

# plot individual models
for i in range(2):
    plt.figure()
    plt.plot(xt, my_mv_model.models[i].predict(xt), '-',
             label='model ' + str(i))
    plt.plot(x[:, i], y, 'ok', label='Training data')
    plt.legend()
    plt.xlabel('x ' + str(i))
    plt.ylabel('y')
plt.show()
