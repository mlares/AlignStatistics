import numpy as np
from math import sqrt
from matplotlib import pyplot as plt

def f1(N):
    u = np.random.uniform(0, 1, N)
    b = np.tan(np.arccos(u))
    return b


def f2(N):
    X = np.random.normal(loc=0, scale=1, size=(N, 3))
    n =  np.linalg.norm(X, axis=1)
    s_par = X[:,2]
    s_perp = np.linalg.norm(X[:,0:2], axis=1)
    b = s_perp / s_par
    return b


def f3(N):
    X = np.random.uniform(0, 1, (3*N, 3))
    n =  np.linalg.norm(X, axis=1)
    filtro = n<1
    s_par = X[filtro][:,2]
    s_perp = np.linalg.norm(X[filtro][:,0:2], axis=1)
    b = s_perp / s_par
    return b


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

N=1000
nbins=65

ax.hist(np.log10(f1(N)), bins=nbins, ec='cadetblue', fc='none', lw=1.5,
        histtype='step', density=True, label='Método 1')
ax.hist(np.log10(f2(N)), bins=nbins, ec='coral', fc='none', lw=1.5,
        histtype='step', density=True, label='Método 2')
ax.hist(np.log10(f3(N)), bins=nbins, ec='orchid', fc='none', lw=1.5,
        histtype='step', density=True, label='Método 3')


ax.set_xlim(-2, 3)
ax.set_xlabel(r'$log_{10}(\beta)$', fontsize=16)
ax.set_ylabel(r'$1/N~~dN/dlog_{10}(\beta$)', fontsize=16)
ax.legend(loc='upper left')
ax.grid(linewidth=0.3, color='whitesmoke')

fig.savefig('br.pdf')








M = 1000
etas = []

for _ in range(5000):
    b = f1(M)
    N = sum(b>1)
    etas.append(N/(M-N))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

ax.hist(etas, bins=50, ec='cadetblue', fc='none', lw=1.5,
        histtype='step', density=True, label='Método 1')

ax.set_xlabel(r'$\eta$', fontsize=16)
ax.set_ylabel(r'$1/N~~dN/d\eta$', fontsize=16)
ax.grid(linewidth=0.3, color='whitesmoke')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)

eta0 = 1/(sqrt(2)-1)
ax.axvline(eta0, linestyle='--', color='silver')


fig.savefig('eta.pdf')

#  #################################################


#-----------
from scipy import stats
eta0 = 1/(sqrt(2)-1)
p0 = 1/sqrt(2)


#-----------
N = 1000
M = 1000
F=stats.binom(N, p0)
x=F.rvs(M)
eta_bin = x/(N-x)
#-----------
eta_mc = []
for _ in range(N):
    b = f1(M)
    N = sum(b>1)
    eta_mc.append(N/(M-N))
#-----------


#print(np.mean(e), eta0)
 
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()

ax.hist(eta_mc, bins=30, ec='cadetblue', fc='none', lw=1.5,
        histtype='step', density=True, label='Monte Carlo')
ax.hist(eta_bin, bins=30, ec='coral', fc='none', lw=1.5,
        histtype='step', density=True, label='Theoretical')

ax.set_xlabel(r'$\eta$', fontsize=16)
ax.set_ylabel(r'$1/N~~dN/d\eta$', fontsize=16)
ax.grid(linewidth=0.3, color='whitesmoke')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)

eta0 = 1/(sqrt(2)-1)
ax.axvline(eta0, linestyle='-', linewidth=1,
           color='silver', label='theoretical')
ax.axvline(np.mean(eta_mc), linestyle='-', linewidth=1,
           color='cadetblue', label='average')
ax.axvline(np.mean(eta_bin), linestyle='-', linewidth=1,
           color='coral', label='average')

ax.legend()
plt.tight_layout()

fig.savefig('eta_binom.pdf')
