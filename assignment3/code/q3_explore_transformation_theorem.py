import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

rv_x = scipy.stats.norm(0,1)
sample_x = rv_x.rvs(size=[1000])
sample_y = np.exp(sample_x)

def pdf(x):
    return 1/(x*math.sqrt(2*math.pi)) * np.exp(-.5*np.log(x)**2)

x = np.linspace(1e-8, 20)
y = pdf(x)
plt.hist(sample_y, density=True, bins="auto", label="sample")
plt.plot(x,y, label="manual calculated pdf")
plt.legend()
plt.show()