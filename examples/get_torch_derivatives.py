from time import time
import numpy as np
import toleranceinterval as ti
import matplotlib.pyplot as plt
import pwlf
import torch


class LeastSquares:
    def __init__(self):
        pass
    
    def lstq(self, A, Y, lamb=0.01):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        # Assuming A to be full column rank
        cols = A.shape[1]
        # print (torch.matrix_rank(A))
        if cols == torch.matrix_rank(A):
            q, r = torch.qr(A)
            x = torch.inverse(r) @ q.T @ Y
        else:
            A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols)
            Y_dash = A.permute(1, 0) @ Y
            x = self.lstq(A_dash, Y_dash)
        return x



ls = LeastSquares()
# np.random.seed(1231)
n_data = 1000
# generate sin wave data
x = np.linspace(0, 10, num=n_data)
y = np.sin(x * np.pi / 2)
# add noise to the data
y = np.random.normal(0, 0.05, n_data) + y

# convert x, y to tensors
x, y = torch.as_tensor(x), torch.as_tensor(y)

n_segments = 20
n_parameters = n_segments + 1
nVar = n_segments - 1
# breaks_list = [0.0, 1.0, 3.0, 5.0, 7.0, 10.0]
breaks_list = list(np.linspace(0, 10, n_parameters))
breaks = torch.tensor(breaks_list, requires_grad=True)
degree = 1

def fit_lst_sq(breaks=breaks):
    # Assemble the regression matrix
    A_list = [torch.ones_like(x)]
    if degree >= 1:
        A_list.append(x - breaks[0])
        for i in range(n_segments - 1):
            A_list.append(torch.where(x > breaks[i+1],
                                      x - breaks[i+1],
                                      torch.zeros_like(x)))
        if degree >= 2:
            for k in range(2, degree + 1):
                A_list.append((x - breaks[0])**k)
                for i in range(n_segments - 1):
                    A_list.append(torch.where(x > breaks[i+1],
                                              (x - breaks[i+1])**k,
                                              torch.zeros_like(x)))
    else:
        A_list = [torch.ones_like(x)]
        for i in range(n_segments - 1):
            A_list.append(torch.where(x > breaks[i+1],
                                      torch.ones_like(x),
                                      torch.zeros_like(x)))
    A = torch.stack(A_list).T

    beta = ls.lstq(A, y)

    # compute the error
    y_hat = torch.matmul(A, beta)
    e = y_hat - y
    ssr = torch.dot(e, e)
    return beta, y_hat, ssr

runtimes_ag = []
for i in range(10):
    t0 = time()
    beta, y_hat, ssr = fit_lst_sq(breaks=breaks)

    # get the derivative of ssr wrt breaks
    d = torch.autograd.grad(ssr, breaks, retain_graph=True, create_graph=True)[0]
    t1 = time()
    runtimes_ag.append(t1-t0)
print(f"Runtime of average autograd {np.mean(runtimes_ag)} seconds")
print(f"Runtime of std autograd {np.std(runtimes_ag, ddof=1)} seconds")

print(d)

# compute the derivative with fd
runtimes_fd = []
for step_size in np.logspace(-10, -1, 10):
    d_fd = torch.zeros(n_parameters)
    t0 = time()
    for i in range(nVar):
        breaks_list[i+1] += step_size  # forward step
        breaks_step = torch.tensor(breaks_list)
        _, _, ssr_step = fit_lst_sq(breaks=breaks_step)
        d_fd[i+1] = (ssr_step - ssr) / step_size
        breaks_list[i+1] -= step_size  # reset step
    t1 = time()
    # compute the average absolute deviation between fd and autograd
    avg_absolute = torch.abs(d-d_fd).mean()
    print(step_size, avg_absolute)
    runtimes_fd.append(t1-t0)

print(f"Runtime average of finite differences {np.mean(runtimes_fd)} seconds")
print(f"Runtime standard decvation of finite differences {np.std(runtimes_fd, ddof=1)} seconds")

ag_bounds = ti.twoside.lognormal(runtimes_ag, 0.9, 0.9)
fd_bounds = ti.twoside.lognormal(runtimes_fd, 0.9, 0.9)

print(f"Autograd runtime bounds{ag_bounds}")
print(f"Finite difference runtime bounds{fd_bounds}")

y_hat_np = y_hat.detach().numpy()

# plot the results
plt.figure()
plt.plot(x.numpy(), y.numpy(), 'o')
plt.plot(x.numpy(), y_hat_np, '-')
plt.show()
