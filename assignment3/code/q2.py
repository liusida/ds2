import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x,y):
    return (1-x)**2 + 100* (y-x*x)**2

def viz(xspace=(-30, 30, 100), yspace=(-300, 800, 100), trajectory=None):
    x = np.linspace(*xspace)
    y = np.linspace(*yspace)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X,Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    if trajectory is not None:
        ax.scatter(*trajectory)
    plt.show()
viz()

# sanity check for the analytical solution.
epsilon = 1e-8
minimum = rosenbrock(np.array([1-epsilon,1,1+epsilon]),np.array([1-epsilon,1,1+epsilon]))
assert minimum[0] > minimum[1] and minimum[2] > minimum[1]

# SGD
def gradient_descent(x0=0, y0=0, lr=1e-3, n_step=int(1e4), momentum=0.0):
    import torch
    import torch.optim as optim
    x = torch.zeros([1]) + x0
    y = torch.zeros([1]) + y0
    x.requires_grad_(True)
    y.requires_grad_(True)

    optimizer = optim.SGD([x,y], lr=lr, momentum=momentum)
    ts = ([],[],[])
    for i in range(n_step):
        optimizer.zero_grad()
        z = rosenbrock(x,y)
        if i % 200==0:
            if i%1000==0:
                print(f"z = {z.detach().numpy()[0]}")
            ts[0].append(x.detach().numpy()[0])
            ts[1].append(y.detach().numpy()[0])
            ts[2].append(z.detach().numpy()[0])
        z.backward()
        # print(f"x = {x[0]} - {optimizer.param_groups[0]['lr']} * {x.grad[0]}")
        optimizer.step()
        # print(f" = {x[0]}")
        # print("")
    return x[0],y[0],ts

# routine gradient descent
x,y,ts = gradient_descent()
print(f"final result x={x:.3f}, y={y:.3f}")
viz( (0,1,100), (0,1,100), trajectory=ts)

# gradient descent with harder initialization, smaller learning rate, and momentum.
x,y,ts = gradient_descent(x0=3, y0=-2, lr=1e-5, momentum=0.99)
print(f"final result x={x:.3f}, y={y:.3f}")
viz( (0,3,100), (-2,5,100), trajectory=ts)
