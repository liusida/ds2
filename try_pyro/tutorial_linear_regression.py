import os
from functools import partial
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist

DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
# african_nations = df[df["cont_africa"] == 1]
# non_african_nations = df[df["cont_africa"] == 0]
# sns.scatterplot(non_african_nations["rugged"],
#             non_african_nations["rgdppc_2000"],
#             ax=ax[0])
# ax[0].set(xlabel="Terrain Ruggedness Index",
#           ylabel="log GDP (2000)",
#           title="Non African Nations")
# sns.scatterplot(african_nations["rugged"],
#                 african_nations["rgdppc_2000"],
#                 ax=ax[1])
# ax[1].set(xlabel="Terrain Ruggedness Index",
#           ylabel="log GDP (2000)",
#           title="African Nations")
# plt.show()

from torch import nn
from pyro.nn import PyroModule

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

df["cont_africa_x_rugged"] = df["cont_africa"] * df["rugged"]
data = torch.tensor(df[["cont_africa", "rugged", "cont_africa_x_rugged", "rgdppc_2000"]].values,
                        dtype=torch.float)
x_data, y_data = data[:, :-1], data[:, -1]

# linear_reg_model = PyroModule[nn.Linear](3, 1)


# # Define loss and optimize
# loss_fn = torch.nn.MSELoss(reduction='sum')
# optim = torch.optim.Adam(linear_reg_model.parameters(), lr=0.05)
num_iterations = 1500 

# def train():
#     # run the model forward on the data
#     y_pred = linear_reg_model(x_data).squeeze(-1)
#     # calculate the mse loss
#     loss = loss_fn(y_pred, y_data)
#     # initialize gradients to zero
#     optim.zero_grad()
#     # backpropagate
#     loss.backward()
#     # take a gradient step
#     optim.step()
#     return loss

# for j in range(num_iterations):
#     loss = train()
#     if (j + 1) % 50 == 0:
#         print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))


# # Inspect learned parameters
# print("Learned parameters:")
# for name, param in linear_reg_model.named_parameters():
#     print(name, param.data.numpy())

# fit = df.copy()
# fit["mean"] = linear_reg_model(x_data).detach().cpu().numpy()

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
# african_nations = fit[fit["cont_africa"] == 1]
# non_african_nations = fit[fit["cont_africa"] == 0]
# fig.suptitle("Regression Fit", fontsize=16)
# ax[0].plot(non_african_nations["rugged"], non_african_nations["rgdppc_2000"], "o")
# ax[0].plot(non_african_nations["rugged"], non_african_nations["mean"], linewidth=2)
# ax[0].set(xlabel="Terrain Ruggedness Index",
#           ylabel="log GDP (2000)",
#           title="Non African Nations")
# ax[1].plot(african_nations["rugged"], african_nations["rgdppc_2000"], "o")
# ax[1].plot(african_nations["rugged"], african_nations["mean"], linewidth=2)
# ax[1].set(xlabel="Terrain Ruggedness Index",
#           ylabel="log GDP (2000)",
#           title="African Nations")


from pyro.nn import PyroSample


class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean

from pyro.infer.autoguide import AutoDiagonalNormal

model = BayesianRegression(3, 1)
guide = AutoDiagonalNormal(model)


from pyro.infer import SVI, Trace_ELBO


adam = pyro.optim.Adam({"lr": 0.03})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

pyro.clear_param_store()
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(x_data, y_data)
    if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))

guide.requires_grad_(False)

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))

print(guide.quantiles([0.25, 0.5, 0.75]))

from pyro.infer import Predictive


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats


predictive = Predictive(model, guide=guide, num_samples=800,
                        return_sites=("linear.weight", "obs", "_RETURN"))
samples = predictive(x_data)
pred_summary = summary(samples)
