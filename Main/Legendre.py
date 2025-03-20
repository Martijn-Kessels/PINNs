import math, os, time, sys
import numpy as np
from datetime import datetime

# plotting
import matplotlib.pyplot as plt

import scipy

def Pn(x, n: int):
    if x == 1:
        return 1
    elif x == 0:
        return (-1)**n
    
def Pn_d1(x, n: int):
    Pn_1 = 1/2*n*(n+1)
    if x == 1:
        return 2*Pn_1
    elif x == 0:
        return 2*Pn_1*(-1)**(n-1)
    
def Pn_d2(x, n: int):
    Pn_2 = 1/8*(n-1)*n*(n+1)*(n+2)
    if x==1:
        return 4*Pn_2
    elif x==0:
        return 4*Pn_2*(-1)**(n)
    
def Pn_d3(x, n: int):
    Pn_3 = 1/48*(n-2)*(n-1)*n*(n+1)*(n+2)*(n+3)
    if x == 1:
        return 8*Pn_3
    elif x == 0:
        return 8*Pn_3*(-1)**(n-1)
    
def sec_der_ip(m: int,n: int):
    """Inner product of P''n and P''m"""
    if m<=n:
        return int(Pn_d2(1,m)*Pn_d1(1,n) - Pn_d2(0,m)*Pn_d1(0,n) - Pn_d3(1,m)*Pn(1,n) + Pn_d3(0,m)*Pn(0,n))
    else:
        return int(Pn_d2(1,n)*Pn_d1(1,m) - Pn_d2(0,n)*Pn_d1(0,m) - Pn_d3(1,n)*Pn(1,m) + Pn_d3(0,n)*Pn(0,m))
    
def combined_ip(m: int,n: int):
    """Inner product of P'n and P''m plus the inner product of P''n and P'm."""
    if m<=n:
        return (Pn_d1(1,n)*Pn_d1(1,m) - Pn_d1(0,n)*Pn_d1(0,m)
                 - Pn(1,n)*Pn_d2(1,m) + Pn(0,n)*Pn_d2(0,m)
                 + Pn(1,n)*Pn_d2(1,m) - Pn(0,n)*Pn_d2(0,m))
    else:
         return (Pn_d1(1,m)*Pn_d1(1,n) - Pn_d1(0,m)*Pn_d1(0,n)
                  - Pn(1,m)*Pn_d2(1,n) + Pn(0,m)*Pn_d2(0,n)
                  + Pn(1,m)*Pn_d2(1,n) - Pn(0,m)*Pn_d2(0,n))
    
class Legendre_solution:
    def __init__(self, Np: int, epsil: float):
        self.Np = Np
        self.epsil = epsil

        N = Np + 1
        #(p'')**2
        A1 = np.zeros([N,N])
        for m in range(2,N):
            for n in range(2,N):
                if (m-n)%2 == 0:
                    A1[m,n] = sec_der_ip(m,n)

        #(p')**2
        A2 = np.zeros([N,N])
        for m in range(1,N):
            for n in range(1,N):
                if (m-n)%2 == 0:
                    A2[m,n] = min(2*m*(m+1), 2*n*(n+1))

        #p'*p''
        A3 = np.zeros([N,N])
        for m in range(1,N):
            for n in range(1,N):
                if (m-n)%2 == 1:
                    A3[m,n] = combined_ip(m,n)

        #BCs
        A4 = np.zeros([N,N])
        for m in range(N):
            for n in range(N):
                if (m-n)%2 == 0:
                    A4[m,n] = 1
        
        #p' and p''
        b1 = np.zeros(N)
        b2 = np.zeros(N)
        for m in range(N):
            if m%2 == 1:
                b1[m] = 4
            if m%2 == 0:
                b2[m] = 2*Pn_d1(1,m) - 2*Pn_d1(0,m)
        
        #Combine to one matrix A and one vector b
        self.A = epsil**2*A1 + A2 - epsil*A3 + A4
        self.b=1/2*(b1 - epsil*b2)

        self.coeffs = np.linalg.solve(self.A, self.b)

        p1 = np.ones(N)
        p0 = np.ones(N)
        for i in range(N):
            if i%2 == 1:
                p0[i] == -1

        #Calculate the loss:
        Boundary_loss = 1/2*(self.coeffs.T@p0)**2 + 1/2*(self.coeffs.T@p1)**2
        self.Loss = 1/2*(self.coeffs.T@(self.epsil**2*A1+A2-1*self.epsil*A3)@self.coeffs - 2*self.coeffs.T@self.b+1) + 1/2*Boundary_loss

    def y_coords(self):
        '''Return the y-coordinates corresponding to the solution'''
        x = np.linspace(0,1,1000)
        y = np.zeros_like(x)
        for n,c in enumerate(self.coeffs):
            y += c*scipy.special.eval_sh_legendre(n, x)
        return y

    def show_approx(self):
        '''Make a plot of the condition number'''
        x = np.linspace(0,1,1000)
        y = np.zeros_like(x)
        for n,c in enumerate(self.coeffs):
            y += c*scipy.special.eval_sh_legendre(n, x)
        plt.plot(x,y)
        plt.xlabel('x')
        plt.xlim([0,1])
        plt.ylabel('u')
        plt.title(f'Loss= {self.Loss}')

    def cond(self):
        '''Calculate condition number'''
        return np.linalg.cond(self.A)
