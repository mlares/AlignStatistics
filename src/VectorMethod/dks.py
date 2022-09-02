import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.linalg import lstsq
#import stairway
import pickle


# utils --------------------

def unifac(x, a, b):
    # acumulada de la distrib. de referencia (uniforme)
    return 1 + (x-b)/(b-a)

def ffit(x, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0):
    # modelo a ajustar
    S = 0
    S += a1 * np.sin( 2*x )
    S += a2 * np.sin( 4*x )
    S += a3 * np.sin( 6*x )
    S += a4 * np.sin( 8*x )
    S += a5 * np.sin( 10*x )
    S += a6 * np.sin( 12*x )
    return S


# set random state  ----------------------
st0 = np.random.get_state()
with open('rstate.pickle', 'wb') as handle:
    pickle.dump(st0, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('rstate.pickle', 'rb') as handle:
    st0 = pickle.load(handle)
np.random.set_state(st0)

# generar una muestra (test) ----------------------
xmin = 0
xmax = np.pi/2
xx = np.linspace(xmin, xmax, 100)

N = 100
Npars = 5

x = norm.rvs(loc=0, scale=0.9, size=3*N)
x = x[np.logical_and(x>xmin, x<xmax)]
x = x[:N]


# preparar datos
x.sort()
ac = np.array(range(N))/N
delta = ac - unifac(x, 0, np.pi/2)


## non-linear least squares --------------------------
pars_non, cov = curve_fit(ffit, x, delta, method='lm')


# PLOT ===========================================

fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(1, 3, 1)

rr = []
for _ in range(100):
    r = np.random.uniform(0, np.pi/2, N)
    r.sort()
    rr.append(r)
    ax1.plot(r, ac, color='silver', alpha=0.5)
ax1.plot(x, ac)
ax1.plot([0, np.pi/2], [0, 1], '--')


ax2 = fig.add_subplot(1, 3, 2)

ass = []
ds = []
for r in rr:
    d = ac - unifac(r, 0, np.pi/2)
    ax2.plot(r, d, color='silver', alpha=0.5)
    p, cov = curve_fit(ffit, r, d, method='lm')
    ass.append(p[0])

    m = np.argmin(abs(d))
    ds.append(d[m])
ax2.plot(x, delta)


ax2.plot(x, ffit(x, *pars_non))
ax2.plot(x, ffit(x, pars_non[0], 0, 0))
ax2.plot(x, ffit(x, 0, pars_non[1], 0))
ax2.plot(x, ffit(x, 0, 0, pars_non[2]))

ax3 = fig.add_subplot(1, 3, 3)

ax3.hist(ass)
ax3.hist(ds)
print(max(delta))
print(pars_non)
ax3.axvline(max(delta))

plt.show()
plt.tight_layout()
fig.savefig('dks.pdf')



#fig = plt.figure(figsize=(15, 10))
#
#ax1 = fig.add_subplot(1,3,1)
#ax1.plot(xx, chi2.pdf(xx, df), '-',color='tomato', 
#         lw=5, alpha=1, label='Distribución')
#ax1.hist(x, density=True, alpha=0.5, label='histograma') #, bins=int(np.sqrt(N)))
#ax1.set_xlabel('x', fontsize=16)
#ax1.set_ylabel('f', fontsize=16)
#ax1.tick_params(axis='both', which='major', labelsize=16)
#ax1.tick_params(axis='both', which='minor', labelsize=16)
#ax1.legend(fontsize=16)
#
#ax2 = fig.add_subplot(1,3,2)
#xx = np.linspace(0, np.pi/2, 200)
#ax2.step(xn, ac, where='pre', label='ECDF')
#ax2.plot([min(xn), max(xn)], [0, 1], linestyle='--', color='cadetblue', 
#         label='referencia')
#y = ffit(xx, *pars_non) + unifac(xx, 0, np.pi/2)
#ax2.plot(xx, y, label='ajuste LM')
#yl = ffit(xx, *pars_lin) + unifac(xx, 0, np.pi/2)
#ax2.plot(xx, yl, label='ajuste min. sq.')
#ax2.set_xlabel('x', fontsize=16)
#ax2.set_ylabel('ECDF(x), ajustes', fontsize=16)
#ax2.tick_params(axis='both', which='major', labelsize=16)
#ax2.tick_params(axis='both', which='minor', labelsize=16)
#ax2.legend(fontsize=16)
#
#ax3 = fig.add_subplot(1,3,3)
#ax3.plot(xn, delta, 'o', markersize=2)
#ax3.axhline(0, color='silver', linestyle='--')
#ax3.plot( xx, ffit(xx, *pars_non), label='ajuste LM' )
#ax3.plot( xx, ffit(xx, *pars_lin), label='ajuste min. sq.' )
#ax3.set_xlabel('x', fontsize=16)
#ax3.set_ylabel(r'$\Delta(x)$', fontsize=16)
#ax3.tick_params(axis='both', which='major', labelsize=16)
#ax3.tick_params(axis='both', which='minor', labelsize=16)
#ax3.legend(fontsize=16)
#
#plt.tight_layout()
#fig.savefig('fit.pdf')  
