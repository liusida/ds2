import numpy as np
from numpy.core.function_base import linspace
import scipy
import scipy.stats
import matplotlib.pyplot as plt
shape = 2
scale = 3
rv_gamma = scipy.stats.gamma(shape, scale=scale)
rv_inv_gamma = scipy.stats.invgamma(shape, scale=scale)
l = 1000
sample_gamma = rv_gamma.rvs(size=[l])
sample_inv_gamma = rv_inv_gamma.rvs(size=[l])
axes = []
fig,ax = plt.subplots()
ax.hist(sample_gamma, bins="auto", density=True, color=[1,0.4,0.4,0.4], label="gamma distribution")
axes.append(ax)
for i in range(2):
    axes.append(ax.twinx())
axes[1].hist(sample_inv_gamma, bins="auto", density=True, color=[0.4,0.4,1,0.4], label="inv-gamma distribution")

x = np.linspace(1e-8,20)
def pdf_gamma(x,shape, scale):
    return 1/(scipy.special.gamma(shape)*shape**scale) * x**(shape-1) * np.exp(-x/scale)
def pdf_inv_gamma(x, shape, scale):
    return scale**shape /scipy.special.gamma(shape) * x**(-shape-1) * np.exp(-scale/x)
y_gamma = scipy.stats.gamma.pdf(x, shape, scale=scale)
y_gamma_m = pdf_gamma(x, shape, scale)
y_inv_gamma = pdf_inv_gamma(x,shape, scale)
axes[2].plot(x,y_gamma, color=[1,0.4,0.4,0.9])
axes[2].plot(x,y_gamma_m, color=[0.4,1,0.4,0.9])
axes[2].plot(x,y_inv_gamma, color=[0.4,0.4,1,0.9])
for ax in axes:
    ax.set_xlim(0,20)
    ax.set_ylim(0,1.5)
plt.savefig("./q3_explore_2.png")
