#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:44:58 2020

@author: connorcolombo
"""

import numpy as np
from scipy import linalg
from numpy.linalg import matrix_power as mpow
import matplotlib.pyplot as plt
import control

# Settings:
m,lr,lf,Ca,Iz,f,Dt = (1888.6, 1.39, 1.55, 20000, 25854, 0.019, 0.032)

# Setup:
v = np.linspace(1,40,1000).reshape(-1,1)

logsig = np.zeros(v.shape) # log10(s1/sn) (Preallocated for speed)
realpoles = np.zeros((v.size,4)) # Re(pi) where each column is a pole (Preallocated for speed)

B = np.asarray([
    [0, 0],
    [2*Ca/m, 0],
    [0, 0],
    [2*Ca*lf/Iz, 0]
])

for i in range(0,v.size):
    Vx = v[i,0]
    A = np.asarray([
        [0, 1, 0, 0],
        [0, -4*Ca/m/Vx, 4*Ca/m, -2*Ca*(lf-lr)/m/Vx],
        [0, 0, 0, 1],
        [0, -2*Ca*(lf-lr)/Iz/Vx, 2*Ca*(lf-lr)/Iz, -2*Ca*(lf**2+lr**2)/Iz/Vx]
    ])
    P = np.hstack((B,A@B, mpow(A,2)@B, mpow(A,3)@B))
    
    # Part a:
    _,s,_ = linalg.svd(P)
    logsig[i,0] = np.log10(s[1]/s[-1])
    
    # Part b:
    realpoles[i,:] = np.real(linalg.eigvals(A))
    
# Part a Results:
plt.figure()
plt.plot(v,logsig)
plt.title(r'$\log_{10}(\frac{\sigma_1}{\sigma_n})$ vs $v$')
plt.ylabel(r'$\log_{10}(\frac{\sigma_1}{\sigma_n})$')
plt.xlabel(r'$v$')
plt.show()

# Part b:
plt.figure()
plt.title(r'$Re(p_i)$ vs $v$ for each pole $p_i$')
for c in range(0,4):
    plt.subplot(2,2,c+1)
    plt.plot(v,realpoles[:,c])
    plt.ylabel(r'$Re(p_{})$'.format(c+1))
    plt.xlabel(r'$v$')
plt.tight_layout() # Ensure there's (just) enough room for the axis labels
plt.show()
    