import pyro
from pyro.distributions import Normal,Gamma,InverseGamma,Bernoulli,Poisson

def normal_density_estimation(obs, N):
    assert obs is None or N==obs.shape[0]
    loc = pyro.sample("loc", Normal(0,1))
    inverse_scale = pyro.sample("inverse_scale", Gamma(3,2))
    with pyro.plate("n", N):
        data = pyro.sample(f"data", Normal(loc, 1/inverse_scale), obs=obs)
    return data

data = normal_density_estimation(None, 10)
print(data)