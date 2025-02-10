# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:56:42 2024

@author: massionb
"""


import numpy as np
import time 

def bisection_idb(mu_min, mu_max, m, n, K_bis=15, K=2000, lam=0.2, gamma0=0.1, rho=0.999, search_it=5, tol_stop=0.0001, verbose=True):
    # Compute low coherence frames by bisection and the IDB algorithm
    # Input:
    #   mu_min          - lower limit of the search interval
    #   mu_max          - upper limit of the search interval
    #   K_bis           - number of bisection iterations
    #   m               - dimension of vectors
    #   n               - frame size
    #   K               - number of IDB iterations
    #   lam             - trade-off factor
    #   gamma0          - initial step size
    #   rho             - step size decrease factor
    #   search_it       - number of halving steps in gradient search
    #   tol_stop        - stopping tolerance
    # Output:
    #   Fbest           - frame with best coherence
    #   coh_best        - coherence value
    #   iter_total_count- total number of IDB iterations in the bisection algorithm
    
    # BD 11.11.2022
    
    if lam==None:
        lam = adaptLambda(m, n)
    
    iter_total_count = 0
    coh_best = 1
    for i_bis in range(1,K_bis+1):
        mu = (mu_min + mu_max)/2
        
        if verbose==True: 
            print("Iteration %d" %i_bis)
            print(mu)
        F, coh = idb(m, n, mu, K=K, lam=lam, gamma0=gamma0, rho=rho, search_it=search_it, tol_stop=tol_stop, verbose=verbose)
        iter_total_count = iter_total_count + len(coh)
        
        coh_crt = min(coh)
        
        if verbose==True: 
            print(coh_crt)
            
        if coh_crt < coh_best:
            coh_best = coh_crt
            Fbest = F
        if coh_crt < mu + 10*tol_stop:      # successful design
            if coh_crt > mu_max:            # if current coherence is larger than current best, stop
                break                       # further improvement is impossible, since the interval stays the same
            mu_max = coh_best
        else:
            mu_min = mu
            mu_max = min(coh_crt, mu_max)   # maybe the upper bound can be also improved
        
        if verbose==True: 
            print(coh_best)
            print("Interval [%f,%f]" %(mu_min, mu_max))
            print("")
            
        if mu_max - mu_min < tol_stop:      # stop if search interval is very small
            break

    return Fbest, coh_best, iter_total_count

def idb(m, n, desired_coh, K=2000, lam=0.2, gamma0=0.1, rho=0.999, search_it=5, tol_stop=0.0001, verbose=True):
    # Minimize frame coherence using a distance barrier function
    # Input:
    #   m               - dimension of vectors
    #   n               - frame size
    #   desired_coh     - target coherence
    #   K               - number of IDB iterations
    #   lam             - trade-off factor
    #   gamma0          - initial step size
    #   rho             - step size decrease factor
    #   search_it       - number of halving steps in gradient search
    #   tol_stop        - stopping tolerance
    # Output:
    #   F               - frame with best coherence
    #   cohv            - vector with the coherence at each iteration
    # BD 8.08.2022, 11.11.2022
    
    start_time_it = time.time()
    F = np.random.randn(m,n)
    F = F/np.linalg.norm(F, axis=0)
    
    cohv = np.zeros(K)
    coh_min = 1
    
    time_in_part = 0.0
    for k in range(K):
        
        perm = np.random.permutation(np.arange(1,n))
        for j in perm:
            
            # 10-12% of computation in later stages
            d = F[:,j]
            v = F.T@d
            v[j] = 0
            # what atoms are too close?
            i_minus = np.where(v > desired_coh)[0]
            i_plus = np.where(v < -desired_coh)[0]
            # weighted least squares
            w_sq = np.maximum(np.abs(v) / desired_coh, 1)
            
            # gradient
            # 25-30% of computation in later stages
            g = F@(w_sq*v)
            if len(i_minus)>0:
                g += lam * np.sum(F[:, i_minus] - d[:, None], axis=1)
            if len(i_plus)>0:
                g -= lam * np.sum(F[:, i_plus] + d[:, None], axis=1)
            
            # loop trying to find better atom
            # 50-55% of computation in later stages
            g_now = gamma0
            c_max = np.max(np.abs(v))
            for i in range(search_it):   # a number of iterations to avoid getting stuck
                dn = d - g_now*g
                dn = dn / np.linalg.norm(dn)
                vn = F.T@dn
                vn[j] = 0
                if np.max(np.abs(vn)) < c_max:
                    #i = i-1
                    break
                g_now = g_now/2.0
            F[:,j] = dn       
            
        gamma0 *= rho
        cohv[k] = np.max(np.abs(F.T@F - np.eye(n)))
        if cohv[k] < coh_min:
            coh_min = cohv[k]
            F_best = F
        if coh_min - desired_coh < tol_stop:    # if coherence is near enough the target, stop
            cohv = cohv[:k+1]
            break
        
    F = F_best
    
    end_time_it = time.time()
    time_it = end_time_it-start_time_it
    if verbose == True:
        print("Time bisection step: %.3f" %time_it)
        print("Time/iteration: %.5f" %(time_it/len(cohv)))
    
    return F, cohv


def adaptLambda(m, n):
    ratio = m/n
    if ratio < 2.0:
        return 0.2
    elif 2.0 <= ratio < 10.0:
        return 0.5
    elif 10.0 <= ratio < 20.0:
        return 1.0
    elif ratio > 20.0:
        return 2.0
    

def getBound(N,M): #N=m, M=n, from Zorlein and Bossert, "Coherence Optimization and Best Complex Antipodal Spherical Codes", 2014
    if M <= N**2:
        bound = np.sqrt((M-N)/(N*(M-1)))
    elif N**2 < M and M <= 2*(N**2-1): #Too laxist, because these bounds and ranges are for the complex frames (there exists tighter bounds for real settings), but keep it as it is for comparison with Matlab
        bound = np.max([np.sqrt(1/N), np.sqrt((2*M-N**2-N)/((N+1)*(M-N))), 1 - 2*M**(-1/(N-1))])
    elif M > 2*(N**2-1):
        bound = np.max([np.sqrt((2*M-N**2-N)/((N+1)*(M-N))), 1 - 2*M**(-1/(N-1))])
            
    return bound


def run_idb():
    m = 9
    n = 100
    # m = 150
    # n = 450
    
    start_time = time.time()
    mu_min = getBound(m,n)
    F = np.random.randn(m,n)
    F = F/np.linalg.norm(F, axis=0)
    mu_max = np.max(np.abs(F.T@F - np.eye(n)))
    print("Interval [%f,%f]" %(mu_min, mu_max))
    F_IDB, coh_best, iter_total_count = bisection_idb(mu_min, mu_max, m, n, lam=None, K=2000)
    
    print("Total number of IDB iterations: %d" %iter_total_count)
    coh_IDB = np.max(np.abs(F_IDB.T@F_IDB - np.eye(n)))
    print("Final IDB coherence: %f" %coh_IDB)
    end_time = time.time()
    duration = end_time-start_time
    print("Time: %.1f" %duration)
    print("Time/iteration: %.5f" %(duration/iter_total_count))

