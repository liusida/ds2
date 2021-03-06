# requirements.txt:
# pyro 1.6.0
# torch 1.8.0

import pyro
from pyro.distributions import Normal,Gamma,InverseGamma,Bernoulli,Poisson
import matplotlib.pyplot as plt
# import pyro.poutine as poutine
pyro.set_rng_seed(101)

def normal_density_estimation(obs, N):
    assert obs is None or N==obs.shape[0]
    loc = pyro.sample("loc", Normal(0,1))
    inverse_scale = pyro.sample("inverse_scale", Gamma(3,2))
    with pyro.plate("n", N):
        data = pyro.sample(f"data", Normal(loc, 1/inverse_scale), obs=obs)
    return data

if __name__ == "__main__":
    # simple test
    data = normal_density_estimation(None, 100000)
    plt.hist(data.detach().numpy(), bins="auto")
    plt.show()