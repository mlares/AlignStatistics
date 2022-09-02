"""
Compute and plot the distributions of lambda
f(lambda), F(lambda)
(Fig. 4)
"""

import numpy as np
from matplotlib import pyplot as plt

def f(t):
    # funcion densidad de probabilidad
    return np.sin(t)
f=np.vectorize(f)

def F(t):
    # funcion acumulada de probabilidad
    return 1-np.cos(t)
F=np.vectorize(F)


t = np.linspace(0, np.pi/2, 400)

fig = plt.figure(figsize=(12, 10))
ax1, ax2 = fig.subplots(2, 1)

ax1.plot(t, f(t), color='cadetblue', linewidth=4)

ax1.set_xticks((0, np.pi/4, np.pi/2))
ax1.set_xticklabels(('0', r'$\pi/4$', r'$\pi/2$'))
ax1.grid(linewidth=0.1, color='gainsboro')
ax1.set_ylabel(r'$f(\lambda)$', fontsize=22)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.tick_params(axis='both', which='minor', labelsize=20)


ax2.plot(t, F(t), color='cadetblue', linewidth=4)

ax2.set_xticks((0, np.pi/4, np.pi/2))
ax2.set_xticklabels(('0', r'$\pi/4$', r'$\pi/2$'))
ax2.grid(linewidth=0.1, color='gainsboro')
ax2.set_xlabel(r'$\lambda$', fontsize=22)
ax2.set_ylabel(r'$F(\lambda)$', fontsize=22)

ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='minor', labelsize=20)

plt.tight_layout()
fig.savefig('F.pdf')
