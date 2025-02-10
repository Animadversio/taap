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


##############################################
#
#  Projections
#
##############################################

def sig_proj_convex(G, n, field, t):
    abs_G = np.abs(G)
    G = np.where(abs_G>=t, t/abs_G *G, G)
    G[np.diag_indices_from(G)] = 1.0
    return G


from scipy import linalg as la

def spec_proj_positive_truncated(G, m, n, field):
    lam, Q_red = la.eigh(G, subset_by_index=[n-m, n-1])
    positive_spectrum = lam
    if field == "real":
        Gtilde = Q_red @ (positive_spectrum[:, None] * Q_red.T)
    elif field == "complex":
        Gtilde = Q_red @ (positive_spectrum[:, None] * np.conjugate(Q_red.T))
    return Gtilde


##############################################
#
#  Normalization
#
##############################################

def normalizeGram(G):
    xsquared = np.diag(G)
    xinv = 1/np.sqrt(xsquared)
    GHat =  G * xinv[:, None] * xinv
    return GHat

def normalizeFrame(F):
    Fhat = F @ np.diag(1/np.linalg.norm(F, axis = 0))
    return Fhat


##############################################
#
#  Gramization
#
##############################################

def constructGram(F, field):
    if field == "real":
        G = F.T@F
    elif field == "complex":
        G = np.conjugate(F.T)@F
    return G


##############################################
#
#  Framization
#
##############################################

def reconstructFrame(G, m, n, field):
    positive_spectrum, Q = np.linalg.eigh(G)
    if field == "real":
        F = np.diag(np.sqrt(positive_spectrum[n-m:]))@Q[:,n-m:].T
    elif field == "complex":
        F = np.diag(np.sqrt(positive_spectrum[n-m:]))@np.conjugate(Q[:,n-m:].T)
    return F


##############################################
#
#  Mutual Coherence
#
##############################################

# F must be a unit frame (F must has normalized columns)
def mutualCoherence(F, field):
    if field == "real":
        gram = F.T@F
    elif field == "complex":
        gram = np.conjugate(F.T) @ F
    mutual_coherence = mutualCoherenceGram(gram)
    return mutual_coherence

# G must have a diagonal of ones
def mutualCoherenceGram(G):
    n = np.shape(G)[0]
    mutual_coherence = np.max(np.abs(G - np.eye(n)))
    return mutual_coherence


##############################################
#
#  Initialization
#
##############################################

def initializeUnitFrame(m, n, field, n_runs=1):
    if field=="real":
        F_0 = np.random.normal(size=(n_runs, m, n))
        F = np.zeros((n_runs, m, n))
    elif field=="complex":
        F_0 = np.random.normal(size=(n_runs, m, 2*n)).view(np.complex128)
        F = np.zeros((n_runs, m, n), dtype = np.complex128)
        
    for i in range(n_runs): 
        F_0[i] = normalizeFrame(F_0[i])
    return F_0, F


##############################################
#
#  Lower bound on mutual coherence
#
##############################################

import scipy.special

def lowerBound(m, n, field):
    if m <= 1:
        print("Can not compute coherence when m<=1.")
        return 
    if n <= 0:
        print("Can not compute coherence when n<=0.")
        return 
    
    if m>=n: #Trivial case
        return 0.0
    
    # Welch, "Lower bounds on the maximum cross correlation of signals (Corresp.)", 1974
    welch_best = 0.0
    if n>m and (field=="real" or field=="complex"): 
        degree_welch = 1
        welch_const = np.sqrt((n-m) / ((n-1)*m))
        welch_k = welch_const
        while welch_k > welch_best:
            welch_best = welch_k
            degree_welch += 1
            binom = scipy.special.comb(m+degree_welch-1, degree_welch)
            rad = (n/binom -1.0)/(n-1.0)
            welch_k = max(rad,0.0)**(1.0/(2.0*degree_welch))
    
    # Rankin, "The Closest Packing of Spherical Caps in n Dimensions", 1955
    orthoplex = 0.0
    if n>m*(m+1)/2 and field=="real":
        orthoplex = np.sqrt(1/m)
    elif n>m**2 and field=="complex":
        orthoplex = np.sqrt(1/m)
    
    # Kabatiansky and Levenshtein, "On Bounds for Packings on a Sphere and in Space", 1978
    levenshtein = 0.0
    if n>m*(m+1)/2 and field=="real":
        levenshtein = np.sqrt((3*n-m**2-2*m)/((m+2)*(n-m)))
    elif n>m**2 and field=="complex":
        levenshtein = np.sqrt((2*n-m**2-m)/((m+1)*(n-m)))
    
    # Bukh and Cox, "Nearly orthogonal vectors and small antipodal spherical codes", 2020
    buhk_cox = 0.0
    if n>m and field=="real": 
        buhk_cox = (n-m)*(n-m+1)/(2*n + (n**2 - m*n - n)*np.sqrt(2+n-m) - (n-m)*(n-m+1))
    elif n>m and field=="complex": 
        buhk_cox = (n-m)**2/(n + (n**2 - m*n - n)*np.sqrt(1+n-m) - (n-m)**2)
    
    # Xia et al., "Achieving the Welch bound with difference sets", 2005
    xia = 0.0
    if m==1 and (field=="real" or field=="complex"):
        xia=1
    elif np.log2(n)>m-1 and (field=="real" or field=="complex"):
        xia = 1 - 2*n**(-1/(m-1))
    
    # Bajwa et al., "Two are better than one: Fundamental parameters of frame coherence", 2012
    bajwa = 0.0
    if field=="real":
        coeff = 2.0**(2-m)/n* 1/scipy.special.beta(m/2, m/2)
        bajwa = max(0,np.cos(np.pi * coeff**(1/(m-1))))
    
    bounds = [welch_best, orthoplex, levenshtein, buhk_cox, xia, bajwa]
    # print(bounds)
    best_bound_index = np.argmax(bounds)
    lower_bound = bounds[best_bound_index]
    # It seems that higher order Welch bounds are always below Xia and Levenshtein bounds
    return lower_bound

