"""
aca calculo la varianza de eta en funcion del tamaño de la muestra,
con simulaciones.
"""

import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from matplotlib import colors

from scipy import stats
from math import sqrt

def f1(N):
    u = np.random.uniform(0, 1, N)
    b = np.tan(np.arccos(u))
    return b

def eta_rvs(Ne, Nb):
    # devuelve Ne muestras de eta calculados con Nb valores de beta

    from scipy import stats
    from math import sqrt

    p0 = 1/sqrt(2)

    F = stats.binom(Nb, p0)
    x = F.rvs(Ne)
    filtro = x<Nb
    eta_bin = []
    for xx in x:
        if xx < Nb:
            eta_bin.append(xx/(Nb-xx))
        else:
            eta_bin.append(0)

    return eta_bin

Ne = 10
Nbs = np.logspace(0.5, 3, 1000)
var_mc = []
for Nb in Nbs:
    etas = eta_rvs(Ne, int(Nb))
    var_mc.append(np.sqrt(np.var(etas)))
 
Ne = 100
Nbs = np.logspace(0.5, 3, 1000)
var_mc_2 = []
for Nb in Nbs:
    etas = eta_rvs(Ne, int(Nb))
    var_mc_2.append(np.sqrt(np.var(etas)))

var_teo = np.sqrt(28.14/Nbs)



plt.close('all')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

ax.plot(Nbs, var_mc, linestyle='None', marker='o', mfc='pink', mec='tomato', 
        markersize=5,
        label='varianza Monte Carlo, \n10 muestras')
ax.plot(Nbs, var_mc_2, linestyle='None', marker='o', mfc='indigo', mec=(1,1,1,0.5),
        mew=0.7, markersize=3,
        label='varianza Monte Carlo, \n100 muestras')
ax.plot(Nbs, var_teo, color='darkslategray', label='varianza teórica')
#ax.plot(Nbs, var_teo*1.6, color='darkslategray', linestyle='--', label='varianza teórica x 1.6 (?)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('tamaño de la muestra, N', fontsize=16)
ax.set_ylabel(r'Desviación estándar, $\sigma(\eta)$', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)

ax.legend(fontsize=12)
plt.tight_layout()

fig.savefig('eta_variance.pdf')
