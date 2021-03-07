import numpy as np
import torch
import pyro
from pyro.distributions import Normal,Gamma,InverseGamma,Bernoulli,Poisson


def discrete_obs_switching_model(obs, model1, model2, T, N):
    log_scale = pyro.sample("log_scale", Normal(0,1))
    z = pyro.sample("z_0", Normal(0, torch.exp(log_scale)))
    xs = [] # this is for the simple test, see if this model can generate samples. not sure if it will break the inference.
    for t in pyro.markov(range(1,T)):
        z = pyro.sample(f"z_{t}", Normal(z,1))
        p = torch.exp(z)/(1+torch.exp(z))
        with pyro.plate("n", N):
            switch = pyro.sample(f"switch_{t}", Bernoulli(p))
            m1 = model1(t)
            m2 = model2(t)
            y = m1**switch+m2**(1-switch)
            x = pyro.sample(f"x_{t}", Poisson(y), obs=obs)
    # might delete below during inference
            xs.append(x.detach().numpy())
    return xs

if __name__ == "__main__":
    # simple test
    def model1(t):
        return t
    def model2(t):
        return t
    xs = discrete_obs_switching_model(None, model1, model2, 2, 3)
    print(np.array(xs))