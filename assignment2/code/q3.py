import sys
import numpy as np
import pandas as pd
np.random.seed(0)


def simulate(params, fn, m1, m2):
    ret = []
    for parameters in params:
        for N in [1,10,100,1000,10000]:
            sample_m1s = []
            sample_m2s = []
            for k in range(30):
                data = fn(parameters, N)
                sample_m1 = np.mean(data)
                sample_m2 = np.var(data)
                sample_m1s.append(sample_m1)
                sample_m2s.append(sample_m2)
            point = {
                "param(s)": f"{parameters}",
                "desc": f"N={N}",
                "m1": np.mean(sample_m1s),
                "m2": np.mean(sample_m2s),
            }
            ret.append(point)
        point = {
            "param(s)": f"{parameters}",
            "desc": "truth",
            "m1": m1(parameters),
            "m2": m2(parameters),
        }
        ret.append(point)
        point = {
            "param(s)": "",
            "desc": "Compare to truth",
            "m1": m1(parameters)/np.mean(sample_m1s),
            "m2": m2(parameters)/np.mean(sample_m2s),
        }
        ret.append(point)
    ret = pd.DataFrame(ret)
    return ret

######

def generate_uniform(parameters, size):
    (a,b) = parameters
    data = np.random.uniform(low=a, high=b, size=size)
    return data
def m1_uniform(parameters):
    (a,b) = parameters
    return (a+b)/2
def m2_uniform(parameters):
    (a,b) = parameters
    return (b-a)**2/12

######

def generate_exponential(parameters, size):
    (beta) = parameters
    data = np.random.exponential(scale=beta, size=size)
    return data
def m1_exponential(parameters):
    (beta) = parameters
    return beta
def m2_exponential(parameters):
    (beta) = parameters
    return beta**2

######

def generate_normal(parameters, size):
    (mu,sigma) = parameters
    data = np.random.normal(loc=mu, scale=sigma, size=size)
    return data
def m1_normal(parameters):
    (mu,sigma) = parameters
    return mu
def m2_normal(parameters):
    (mu,sigma) = parameters
    return sigma**2

######

def powerlaw_invcdf(y, a, gamma):
    return a * (1-y)**(1/(1-gamma))

def generate_power(parameters, size):
    (a,gamma) = parameters
    y = np.random.uniform(low=0, high=1, size=size)
    data = powerlaw_invcdf(y=y, a=a, gamma=gamma)
    return data
def m1_power(parameters):
    (a,gamma) = parameters
    if gamma>2:
        return a*(gamma-1)/(gamma-2)
    return np.inf
def m2_power(parameters):
    (a,gamma) = parameters
    if gamma>3:
        return a*a*(gamma-1)*(1/(gamma-3) - (gamma-1)/((gamma-2)*(gamma-2)))
    return np.inf

def table(df, title=""):
    print(r"""
    \newpage
    \begin{table}[h!]
        \begin{center} {\footnotesize
        \begin{tabular}{ccccc}
        \hline
        & \multicolumn{1}{c}{""",title,r"""}  \\
        \multicolumn{1}{c}{params} & \multicolumn{1}{c}{N} & \multicolumn{1}{c}{m1} & \multicolumn{1}{c}{m1}\\
        \hline
        """)
    for index, row in df.iterrows():
        print(f"{row['param(s)']} & {row['desc']} & {row['m1']:.03f} & {row['m2']:.03f} \\\\")
        if row['desc']=="Compare to truth":
            print("\\\\")
    print(r"""\hline
        \end{tabular} }
        \end{center}
        \label{turns}
    \end{table}
    """)

if __name__=='__main__':

    latex = False
    if len(sys.argv)>1:
        latex = True

    print("")

    title="Uniform"
    params_uniform = [(0,1), (3,7), (11,39), (47, 133)]
    results = simulate(params=params_uniform, fn=generate_uniform, m1=m1_uniform, m2=m2_uniform)
    if latex:
        table(results, title)
    else:
        print(title)
        print(results)
    
    title="Exponential"
    params_exponential = [1, 3.7, 11.39, 47.133]
    results = simulate(params=params_exponential, fn=generate_exponential, m1=m1_exponential, m2=m2_exponential)
    if latex:
        table(results, title)
    else:
        print(title)
        print(results)
    
    title="Normal"
    params_normal = [(0,1), (3,7), (11,39), (47, 133)]
    results = simulate(params=params_normal, fn=generate_normal, m1=m1_normal, m2=m2_normal)
    if latex:
        table(results, title)
    else:
        print(title)
        print(results)
    
    title="Power Law"
    params_power = [(1.5,1.5), (2.2,2.2), (3.3,3.3), (4.4, 4.4)]
    results = simulate(params=params_power, fn=generate_power, m1=m1_power, m2=m2_power)
    if latex:
        table(results, title)
    else:
        print(title)
        print(results)
    
    print("")