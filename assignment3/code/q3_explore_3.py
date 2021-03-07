# suppose x~N(0,1), y~N(0,y), find p(x,y)

import math
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pyro
from pyro.distributions import Normal,InverseGamma
print("Start")
fig = plt.figure()

def forward():
    # loc = pyro.sample("loc", Normal(0,1))
    loc = 0
    scale = pyro.sample("scale", InverseGamma(3,2))
    data = pyro.sample("data", Normal(loc,scale))
    return loc, scale, data

locs, scales, datas = [],[],[]
for i in range(10000):
    loc, scale, data = forward()
    # locs.append(loc.detach().numpy().item())
    scales.append(scale.detach().numpy().item())
    datas.append(data.detach().numpy().item())

zs, scales, datas = np.histogram2d(scales,datas,bins=50)
xs = scales[:-1] + (scales[1]-scales[0])/2
ys = datas[:-1] + (datas[1]-datas[0])/2
xs,ys = np.meshgrid(xs,ys)

ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(xs,ys,zs.T)

def pdf(s,d):
    # return 4/3 * s * 1/math.sqrt(2*math.pi) * np.exp(-2*s-d*d/(2*s*s))
    return 4/(3*s**5) * 1/math.sqrt(2*math.pi) * np.exp(-2/s-d*d/(2*s*s))

s = np.linspace(1e-8,15)
d = np.linspace(-20,20)
s,d = np.meshgrid(s,d)
z = pdf(s,d)

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(s,d,z)
plt.show()