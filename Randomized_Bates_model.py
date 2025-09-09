# -*- coding: utf-8 -*-
"""
Randomization of the Bates model.

The code provided is based on the article 'On Randomization of Affine Diffusion Processes
with Application to Pricing of Options on VIX and S&P 500' by Lech A. Grzelak, 
L.A. Grzelak@uu.nl

@article{grzelakRAnD,
title = {On randomization of affine diffusion processes with application to pricing of options on VIX and S&P 500},
journal = {Applied Mathematics and Computation},
volume = {508},
pages = {129598},
year = {2026},
issn = {0096-3003},
doi = {https://doi.org/10.1016/j.amc.2025.129598},
url = {https://www.sciencedirect.com/science/article/pii/S0096300325003248},
author = {Lech A. Grzelak},
}

@author: LECH A. GRZELAK
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.linalg import cholesky as chol
import scipy.linalg as linalg

def CallPutOptionPriceCOSMthd(cf,CP,S0,r,tau,K,N,L):
    # cf   - characteristic function as a functon, in the book denoted as \varphi
    # CP   - C for call and P for put
    # S0   - Initial stock price
    # r    - interest rate (constant)
    # tau  - time to maturity
    # K    - list of strikes
    # N    - Number of expansion terms
    # L    - size of truncation domain (typ.:L=8 or L=10)  
        
    # reshape K to a column vector
    K = np.array(K).reshape([len(K),1])
    
    #assigning i=sqrt(-1)
    i = np.complex(0.0,1.0) 
    
    x0 = np.log(S0 / K)   
    
    # truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # sumation from k = 0 to k=N-1
    k = np.linspace(0,N-1,N).reshape([N,1])  
    u = k * np.pi / (b - a);  

    # Determine coefficients for Put Prices  
    H_k = CallPutCoefficients(CP,a,b,k)
       
    mat = np.exp(i * np.outer((x0 - a) , u))

    temp = cf(u) * H_k 
    temp[0] = 0.5 * temp[0]    
    
    value = np.exp(-r * tau) * K * np.real(mat.dot(temp))
         
    return value

""" 
Determine coefficients for Put Prices 
"""
def CallPutCoefficients(CP,a,b,k):
    if str(CP).lower()=="c" or str(CP).lower()=="1":                  
        c = 0.0
        d = b
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros([len(k),1])
        else:
            H_k      = 2.0 / (b - a) * (Chi_k - Psi_k)  
        
    elif str(CP).lower()=="p" or str(CP).lower()=="-1":
        c = a
        d = 0.0
        coef = Chi_Psi(a,b,c,d,k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k      = 2.0 / (b - a) * (- Chi_k + Psi_k)               
    
    return H_k    

def Chi_Psi(a,b,c,d,k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a)/(b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    
    chi = 1.0 / (1.0 + np.power((k * np.pi / (b - a)) , 2.0)) 
    expr1 = np.cos(k * np.pi * (d - a)/(b - a)) * np.exp(d)  - np.cos(k * np.pi 
                  * (c - a) / (b - a)) * np.exp(c)
    expr2 = k * np.pi / (b - a) * np.sin(k * np.pi * 
                        (d - a) / (b - a))   - k * np.pi / (b - a) * np.sin(k 
                        * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    
    value = {"chi":chi,"psi":psi }
    return value

def ImpliedVolatility(CP,V_market,S_0,K,tau,r,sigma0):
    error    = 1e10; # initial error
    Nmax = 100
    #Handy lambda expressions
    optPrice = lambda sigma: BS_Call_Option_Price(CP,S_0,K,sigma,tau,r)
    vega= lambda sigma: dV_dsigma(S_0,K,sigma,tau,r)
    
    sigma = sigma0
    # While the difference between the model and the arket price is large
    # follow the iteration
    i =0
    while error>10e-10 and  i <Nmax:
        f         = V_market - optPrice(sigma);
        f_prim    = -vega(sigma);
        sigma_new = sigma - f / f_prim;
    
        error=abs(sigma_new-sigma);
        sigma=sigma_new;
        i=i+1
        #print(i)
    return sigma

# Vega, dV/dsigma
def dV_dsigma(S_0,K,sigma,tau,r):
    #parameters and value of Vega
    d2   = (np.log(S_0 / (K)) + (r - 0.5 * np.power(sigma,2.0)) * tau) / float(sigma * np.sqrt(tau))
    value = K * np.exp(-r * tau) * st.norm.pdf(d2) * np.sqrt(tau)
    return value

def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):
    #Black-Scholes Call option price
    d1    = (np.log(S_0 / (K)) + (r + 0.5 * np.power(sigma,2.0)) * tau) / (sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if str(CP).lower()=="c" or str(CP).lower()=="1":
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif str(CP).lower()=="p" or str(CP).lower()=="-1":
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

def CollocationUniform(a,b,N):
    moment = lambda n: (b**(n+1)-a**(n+1))/((n+1)*(b-a))

    # Creation of Matrix M, dimension N+1 x N+1
    M = np.zeros([N+1,N+1])    
    for i in range(0,N+1):
        for j in range(0,N+1):
            M[i,j] =moment(i+j)        
    
    # Once the moments are computed use generic code to calculate quadrature pairs
    x_i, w_i = FindCollocation(M)
    
    return x_i, w_i

def FindCollocation(M):
     # Creation of UPPER diagonal matrix, R, dimension N+1 x N+1

    # Since Matrix M also includes the 0 moment we adjust the size
    N = len(M)-1    
    
    R = chol(M)   
      
    # Creation of vector alpha and beta
    alpha = np.zeros([N])
    beta = np.zeros([N-1])
    
    alpha[0] = R[0,1]
    beta[0]  = (R[1,1]/R[0,0])**2.0
    
    for i in range(1,N):
        alpha[i] = R[i,i+1]/R[i,i] - R[i-1,i]/R[i-1,i-1]
    
    for i in range(1,N-1):
        beta[i]  = (R[i+1,i+1]/R[i,i])**2.0 

    # Construction of matrix J
    J = np.diag(np.sqrt(beta),k=-1)+np.diag(alpha,k=0)+np.diag(np.sqrt(beta),k=1);
    
    # computation of the weights
    eigenValues, eigenVectors = linalg.eig(J)
        
    w_i = eigenVectors[0,:]**2.0
    x_i = np.real(eigenValues)
    
    # sorting the arguments
    idx =np.argsort(x_i)
    w_i = w_i[idx]
    x_i = x_i[idx]
    
    return x_i, w_i

def ChFBatesModel(u,r,tau,kappa,gamma,vbar,v0,rho,xiP,muJ,sigmaJ):
    i = np.complex(0.0,1.0)
    D1 = np.sqrt(np.power(kappa-gamma*rho*i*u,2)+(u*u+i*u)*gamma*gamma)
    g  = (kappa-gamma*rho*i*u-D1)/(kappa-gamma*rho*i*u+D1)
    C  = (1.0-np.exp(-D1*tau))/(gamma*gamma*(1.0-g*\
                               np.exp(-D1*tau)))*(kappa-gamma*rho*i*u-D1)
    # Note that we exclude the term -r*tau, as the discounting is performed in the COS method
    AHes= r * i*u *tau + kappa*vbar*tau/gamma/gamma *(kappa-gamma*\
        rho*i*u-D1) - 2*kappa*vbar/gamma/gamma*np.log((1.0-g*np.exp(-D1*tau))/(1.0-g))
    
    A = AHes - xiP * i * u * tau *(np.exp(muJ+0.5*sigmaJ*sigmaJ) - 1.0) + \
            xiP * tau * (np.exp(i*u*muJ - 0.5 * sigmaJ * sigmaJ * u * u) - 1.0)
    
    # Characteristic function for the Heston's model    
    cf = np.exp(A + C*v0)
    return cf 

def CHF_Bates_RAnD_randomized_gamma(u,r,tau,kappa,gamma,vbar,v0,rho,xiP,muJ,sigmaJ,x_i,w_i):
    ChF_output= np.zeros([len(u),1])
    for n in range(len(x_i)):
        gamma = x_i[n]
        cf = ChFBatesModel(u,r,tau,kappa,gamma,vbar,v0,rho,xiP,muJ,sigmaJ)
        ChF_output = ChF_output + w_i[n]*cf
        
    return ChF_output

def CHF_GBM_RAnD(u,T,r,x_i,w_i):
    i = 1j
    ChF_output= np.zeros([len(u),1])
    for n in range(len(x_i)):
        sigma = x_i[n]
        cf = np.exp((r - 0.5 * sigma**2.0) * i * u * T - 0.5 * sigma**2.0 * u**2.0 * T)
        ChF_output = ChF_output + w_i[n]*cf
    return ChF_output


def main():
    
    # Uniform randomization for the Black Scholes model
    a = 0.1
    b = 0.45
    N = 9
        
    S0 = 1
    CP = 'c'
    r      = 0.0
    muJ    = -0.1 
    sigmaJ = 0.06 
    xiP    = 0.08 
    kappa  = 0.5 
    gamma  = 0.5
    vbar   = 0.13
    rho    = -0.7
    T      = 1/12
    v0     = 0.13 
    Nk     = 25
    K      = np.linspace(0.6,1.4,Nk)
    L = 7
    Ncos = 5000
    
    # Benchmark Bates model
    cf = lambda u: ChFBatesModel(u,r,T,kappa,gamma,vbar,v0,rho,xiP,muJ,sigmaJ)
    
    # Call Option price using the COS Method
    CallValue_Reference = CallPutOptionPriceCOSMthd(cf,CP,S0,r,T,K,Ncos,L)
       
    # Plot option prices
    plt.figure(1)
    plt.plot(K,CallValue_Reference,'--k')
    plt.xlabel('strike K')
    plt.ylabel('Call Option price')
    plt.grid()

    # Compute implied volatilities
    IVRAnD = np.zeros([Nk])            
    for (idx, k) in enumerate(K):
        IVRAnD[idx] = ImpliedVolatility(CP,CallValue_Reference[idx],S0,k,T,r,0.35)

    # Plot implied volatilities
    plt.figure(2)
    plt.plot(K,IVRAnD,'--k')
    plt.xlabel('strike K')
    plt.ylabel('Implied Volatilities')
    plt.title('Implied volatility for Randomized Bates model (RAnD Bates)')
    plt.grid()
    
    # RANDOMIZATION of Gamma parameter with uniformly distributed random variable
    p1Vec = [0.01,0.2,0.1,0.75]
    p2Vec = [0.25,5,2,1.75]
    
    label =['reference, gamma='+str(gamma)]
    
    for n in range(0,len(p1Vec)):
        
        a = p1Vec[n]
        b = p2Vec[n]
        # Compute optimal points, here we take sigma^2 to follow uniform
        x_i, w_i = CollocationUniform(a,b,N)
        
        # Characteristic function for randomized Bates model (randomization of Gamma)
        cf_RAnD = lambda u: CHF_Bates_RAnD_randomized_gamma(u,r,T,kappa,gamma,vbar,v0,rho,xiP,muJ,sigmaJ,x_i,w_i)
        
        # Call Option price using the COS Method
        CallValue_RAnD = CallPutOptionPriceCOSMthd(cf_RAnD,CP,S0,r,T,K,Ncos,L)
        
        # Compute implied volatilities
        IVRAnD = np.zeros([Nk])            
        for (idx, k) in enumerate(K):
            IVRAnD[idx] = ImpliedVolatility(CP,CallValue_RAnD[idx],S0,k,T,r,0.35)
        
        label.append('gamma = U([' + str(a) + ','+ str(b)+'])')
        plt.plot(K,IVRAnD)
    
    plt.legend(label)


main()
