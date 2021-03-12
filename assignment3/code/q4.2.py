# requirements.txt:
# pyro 1.6.0
# torch 1.8.0

# reference: https://github.com/pyro-ppl/pyro/blob/dev/examples/hmm.py

import numpy as np
import torch
import pyro
from pyro.distributions import Normal,Gamma,InverseGamma,Bernoulli,Poisson
import matplotlib.pyplot as plt

def continuous_hmm(obs, N, T):
    assert obs is None or (N==obs.shape[0] and T==obs.shape[1])
    loc = pyro.sample("loc", Normal(0,1))
    log_scale = pyro.sample("log_scale", Normal(0,1))
    obs_scale = pyro.sample("obs_scale", Gamma(2,2))
    scale = torch.exp(log_scale)
    with pyro.plate("n", N):
        x = pyro.sample("x_0", Normal(loc, scale))
        ys = [] # this is for the simple test, see if this model can generate samples. not sure if it will break the inference.
        for t in pyro.markov(range(1,T)):
            # rewrite x, instead of using x[t] = ...x[t-1]...
            x = pyro.sample(f"x_{t}", Normal(loc+x, scale))
            y = pyro.sample(f"y_{t}", Normal(x, obs_scale), obs=None if obs is None else obs[:,t])
    # might delete below during inference
            ys.append(y.detach().numpy())
    return ys

if __name__ == "__main__":
    # simple test
    ys = continuous_hmm(None, 1000, 30)
    ys = np.array(ys)
    y = np.mean(ys, axis=1)
    ci = 1.96 * np.std(ys, axis=1)/y
    x= np.arange(1,ys.shape[0]+1)
    plt.plot(x,y)
    plt.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)
    plt.show()
    #hmm... I am not sure why it looks like this...

    # print(np.array(ys))