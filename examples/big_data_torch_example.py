from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import LBFGS


np.random.seed(1231)
torch.manual_seed(1231)

n_data = 2**24
# the batch size is how much data you will process at a time
# this should be big enough to get accurate gradients
# but not too big where you can't fit the data into memory
batch_size = 2000000

num_epochs = 2  # n umber of optimization iterations
n_segments = 33  # desired number of line segments

# generate sin wave data
x = np.linspace(0, 30, num=n_data)
y = np.sin(x * np.pi / 2)
# add noise to the data
y = np.random.normal(0, 0.05, n_data) + y

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

x_min = torch.min(x).item()
x_max = torch.max(x).item()


class PwlfNet(nn.Module):
    """
    Piecewise linear fit model as torch neural net
    """

    def __init__(self, n_segments, x_min, x_max, degree=1, n_dim=1):
        """
        Initialize pwlf model as one layer feed forward network

        Args:
            n_segments (int): Number of line segments
            x_min (float): Minimum x value
            x_max (float): Maximum x value
            degree (int): Degree of Polynomial transformation to use, default
                degree=1
            n_dim (int): Dimension of output predictions, default n_dim=1

        """
        super(PwlfNet, self).__init__()
        self.n_segments = n_segments
        self.n_parameters = n_segments + 1
        # initialize breaks linearly through the design space
        # alternative initialization could be with LHS
        breaks_guess = torch.linspace(x_min, x_max, self.n_parameters)[1:-1]
        breaks_guess.requires_grad_(True)
        self.breaks = torch.nn.Parameter(breaks_guess, requires_grad=True)
        self.fc1 = nn.Linear(self.n_parameters, n_dim, bias=False)
        self.x_min = x_min
        self.x_max = x_max
        self.degree = degree
        self.n_dim = n_dim

    def forward(self, x):
        """
        Generate predictions for x

        Args:
            x (Tensor): Tensor to generate predictions for. Must have
                shape (:, 1)

        Returns:
            (Tensor): The predictions of the pwlf neural network. Will return
                shape (:, n_dim)
        """
        # Assemble the regression matrix
        A_list = [torch.ones_like(x)]
        if self.degree >= 1:
            A_list.append(x - self.x_min)
            for i in range(self.n_segments - 2):
                A_list.append(torch.where(x > self.breaks[i+1],
                                          x - self.breaks[i+1],
                                          torch.zeros_like(x)))
            A_list.append(torch.where(x > self.x_max,
                                      x - self.x_max,
                                      torch.zeros_like(x)))
            if self.degree >= 2:
                for k in range(2, self.degree + 1):
                    A_list.append((x - x_min)**k)
                    for i in range(self.n_segments - 2):
                        A_list.append(torch.where(x > self.breaks[i+1],
                                                  (x - self.breaks[i+1])**k,
                                                  torch.zeros_like(x)))
                    A_list.append(torch.where(x > self.x_max,
                                              (x - self.x_max)**k,
                                              torch.zeros_like(x)))
        else:
            A_list = [torch.ones_like(x)]
            for i in range(self.n_segments - 2):
                A_list.append(torch.where(x > self.breaks[i+1],
                                          torch.ones_like(x),
                                          torch.zeros_like(x)))
            A_list.append(torch.where(x > self.x_max,
                                      torch.ones_like(x),
                                      torch.zeros_like(x)))
        out = torch.stack(A_list).T
        out = self.fc1(out).view(-1, self.n_dim)
        return out


# initialize the model
model = PwlfNet(n_segments, x_min, x_max)
if torch.cuda.is_available():
    model = model.cuda()

# if you wanted to see the initial break points
initial_break = model.breaks.clone()

dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=True)


criterion = torch.nn.MSELoss()  # mean squared error

# LBFGS optimizer to find both break points and beta paramters
optimizer = LBFGS(model.parameters(), lr=0.1, max_iter=25,
                  max_eval=None, tolerance_grad=1e-10,
                  tolerance_change=1e-14, history_size=100,
                  line_search_fn='strong_wolfe')

total_step = len(train_loader)
for epoch in range(num_epochs):
    t0 = time()
    for i, (X, Y) in enumerate(train_loader):
        def closure(X=X, Y=Y):
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, Y)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        t1 = time() - t0
        t0 = time()
        print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.8f}, Time: {:.4f}'
              .format(epoch+1, num_epochs, i+1, total_step, loss.item(), t1))
    print(f'Model breakpoints: {model.breaks}')

print(f'\nFinal model: {model.state_dict()}')
# breaks are the breakpoint locations
# fc1.weight are the beta paramters

if torch.cuda.is_available():
    y_hat = model.forward(x.cuda())
    y_hat = y_hat.cpu()
else:
    y_hat = model.forward(x)

plt.figure()
plt.plot(x.detach().numpy(), y.detach().numpy(), '.')
plt.plot(x.detach().numpy(), y_hat.detach().numpy(), '-')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('Fit_to_16_million_data.png')
plt.show()
