import math
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

rvx = scipy.stats.norm(0,1)
l = 10000
rvy = rvx.rvs(size=[l])
rvy_exp = np.exp(rvy)
rvy_plus = rvy+10
rvy_times = rvy*2
ax = sns.histplot(x=rvy, label="rvy")
# sns.histplot(x=rvy_exp, label="rvy_exp")
# sns.histplot(x=rvy_plus, label="rvy_plus")
sns.histplot(x=rvy_times, label="rvy_times")
ax2 = ax.twinx()

x = np.linspace(0.01,10)
y = 1/math.sqrt(2*math.pi) * np.exp(-.5* (np.log(x))**2)
y_plus = 1/math.sqrt(2*math.pi) * np.exp(-.5* (x-10)**2)
y_times = 1/math.sqrt(2*math.pi) * np.exp(-.5* (x/2)**2)
# sns.lineplot(x=x,y=y, color="red", ax=ax2, label="y")
# sns.lineplot(x=x,y=y_plus, color="green", ax=ax2, label="y_plus")
sns.lineplot(x=x,y=y_times, color="green", ax=ax2, label="y_plus")


ax.set_ylim(0,550)
ax2.set_ylim(0, np.max(y_plus))
plt.show()

