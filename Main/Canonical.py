import math, os, time, sys
import numpy as np
from datetime import datetime

# plotting
import matplotlib.pyplot as plt

import scipy

def A_n(n,epsil,lam):
    i_list = np.arange(2,n+1)
    A_n = np.ones([n+1,n+1])*(1-lam)
    #A_n[0,:]=np.ones(n+1)*(1-lam)
    A_n[0,0]= (1-lam)*2

    A_n[1,2:]=np.ones(n-1)*(2*lam+1-lam)-2*epsil*i_list*lam
    A_n[2:,1]=np.ones(n-1)*(2*lam+1-lam)-2*epsil*i_list*lam
    A_n[1,1]=(2*lam+1-lam)
    #A_n[1:,0]=-1
    for i in range(2,n+1):
        for k in range(2,n+1):
            A_n[i,k]= lam*(2*epsil**2*k*(k-1)*(i-1)*i/(i+k-3)-2*epsil*k*i+2*k*i/(i+k-1))+(1-lam)

    b = 2*lam*np.ones(n+1)-2*lam*epsil*np.arange(0,n+1)
    b[0]=0
    b[1]=2*lam

    return A_n,b

def real_sol(alpha: float, kappa: float, F: float, f: float, epsil: float, g0: float, g1: float, N: int):
    """A function to calculate the analtyical solution"""
    if epsil<0.025:
        x_coords = np.arange(0,1+1/N,1/N)
        return x_coords, x_coords, 0
    else:
        A = np.array([[kappa,kappa-alpha*F/epsil],[kappa,kappa*np.exp(F/epsil)+alpha*F/epsil*np.exp(F/epsil)]])
        b = np.array([[g0+alpha*f/F],[g1-f/F*(kappa+alpha)]])
        [c1,c2] = np.linalg.solve(A,b)
        #Coordinates for plotting:
        x_coords = np.arange(0,1+1/N,1/N)
        y_coords = c1+c2*np.exp(F*x_coords/epsil)+f/F*x_coords
        return x_coords, y_coords, [A,b,c1,c2]

def show_approx(n,epsil, show_legend=True, show_real=True):
    A,b=A_n(n,epsil,1/2)
    coefs=np.linalg.solve(A,b)
    x,y,abc = real_sol(0,1,1,1,epsil,0,0,10000)

    y2=0
    y2prime = 0
    y2prime2 = 0
    for i in range(len(coefs)):
        y2+=coefs[i]*x**i
        if i != 0:
            y2prime += x**(i-1)*(i)*coefs[i]
            if i != 1:
                y2prime2 += x**(i-2)*(i)*(i-1)*coefs[i]
    plt.plot(x,y2,'--',label=f'n={n}')
    if show_real == True:
        plt.plot(x,y, label='exact')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'$\epsilon={epsil}, n={n}$')
    if show_legend == True:
        plt.legend()
    inner_loss = np.mean((-epsil*y2prime2+y2prime-1)**2)
    print(0.5*inner_loss+0.25*y2[0]**2+0.25*y2[1]**2)