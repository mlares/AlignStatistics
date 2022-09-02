import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return x*pow(1+x**2, -3/2)

f = np.vectorize(f)
b = np.logspace(-3, 2, 401)

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot()

ax.plot(b, f(b))
ax.set_xscale('log')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$f(\beta)$')

g = b>=1
ax.fill_between(b[g], f(b)[g], color='mediumturquoise')
ax.fill_between(b[~g], f(b)[~g], color='lavenderblush')

fig.savefig('beta_1_areas.pdf')



import sympy as s
import numpy as np

x = s.var('x')
a = s.var('a')
b = s.var('b')

f = x*s.sin(s.atan(x))/(1+x**2)

FF = s.integrate(f, (x, a, b))

l=10
FF.evalf(subs={a:1/l, b:l})

Ls = np.logspace(1, 5, 200)
avg = []
for l in Ls:
    aux = FF.evalf(subs={a:1/l, b:l})
    avg.append(aux)


fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot()

ax.plot(Ls, avg)
ax.set_xscale('log')
ax.set_xlabel(r'$L_2$')
ax.set_ylabel(r'$A(1/L_2, L_2)$')


fig.savefig('beta_avg.pdf')
