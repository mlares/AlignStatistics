import numpy as np
from matplotlib import pyplot as plt


t = np.linspace(0, np.pi/2, 200)

def f(t, n):
    return np.sin(2*n*t)


plt.close('all')
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1,2,1)

for n in range(0,6,2):
    ax1.plot(t, f(t,n))

ax2 = fig.add_subplot(1,2,2)

for n in range(1,6,2):
    ax2.plot(t, f(t,n))

ax1.set_xlabel('x', fontsize=16)
ax1.set_ylabel(r'$\phi_k(x)$', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.tick_params(axis='both', which='minor', labelsize=16)
ax2.set_xlabel('x', fontsize=16)
ax2.set_ylabel(r'$\phi_k(x)$', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.tick_params(axis='both', which='minor', labelsize=16)

plt.tight_layout()

fig.savefig('sinbasis.pdf')

