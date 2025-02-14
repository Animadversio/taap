"""
Grassmannian Frame Computation via Accelerated Alternating Projections

30 January 2025 - v1.0

Bastien MASSION
UCLouvain, ICTEAM
bastien.massion@uclouvain.be

Prof. Estelle MASSART
UCLouvain, ICTEAM
estelle.massart@uclouvain.be
"""

import numpy as np
from taap_utils import sig_proj_convex, spec_proj_positive_truncated, normalizeGram, constructGram, reconstructFrame, mutualCoherence, mutualCoherenceGram, lowerBound


##############################################
#
#  TAAP
#
##############################################


def taap(F_0, m, n, field, beta=2.0, N_budg=100000, tau=10**(-6), N_p=100, eps_p=10**(-1), eps_s=10**(-3), acceleration=True, verbose=True):
    if verbose == True:
        print("mu_0_AAP \tmu_AAP \t\ttarget \t\tdelta_t \tN_AAP \tN_tot")
    
    N_tot = 0
    
    G_best = constructGram(F_0, field)
    mu_best = mutualCoherence(F_0, field)
    theoretical_lower_bound = lowerBound(m, n, field)
    
    t = theoretical_lower_bound
    delta_t = mu_best - t
    
    while not (delta_t < tau or N_tot > N_budg):
        G_AAP = G_best
        mu_AAP = mu_best
        k_AAP = 0
        
        c_k_1 = 1.0
        G_k_2 = G_best
        G_k_1 = G_best
        k = 1
        
        while not (mu_AAP - t < eps_s*delta_t or k - k_AAP > N_p):
            if acceleration == True: 
                c_k = np.sqrt(4*c_k_1**2 + 1)/2 + 1/2
                Y_k = G_k_1 + (c_k_1 - 1.0)/c_k * (G_k_1-G_k_2)
                G_k = spec_proj_positive_truncated(sig_proj_convex(Y_k, n, field, t), m, n, field)
            
            else:
                G_k = spec_proj_positive_truncated(sig_proj_convex(G_k_1, n, field, t), m, n, field)
            
            mu_k = mutualCoherenceGram(G_k)
            
            if mu_AAP - mu_k > eps_p * delta_t:
                G_AAP = G_k
                mu_AAP = mu_k
                k_AAP = k
            
            k += 1
            G_k_2 = G_k_1
            G_k_1 = G_k
            c_k_1 = c_k
            N_tot += 1
        
        if verbose == True:
            print("%.6f \t%.6f \t%.6f \t%.6f \t%-6d \t%-6d" %(mu_best, mu_AAP, t, delta_t, k-1, N_tot))
        
        if mu_AAP - t < eps_s*delta_t:
            t = np.max([mu_AAP - beta*delta_t, theoretical_lower_bound])
        
        elif k - k_AAP > N_p:
            t = np.max([mu_AAP - 1/beta*delta_t, theoretical_lower_bound])
        
        G_best = normalizeGram(G_AAP)
        mu_best = mu_AAP
        delta_t = mu_best - t
        
    F_best = reconstructFrame(G_best, m, n, field)
    return F_best, mu_best, N_tot


