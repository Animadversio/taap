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
import time

from taap_utils import initializeUnitFrame, mutualCoherence, lowerBound
from taap import taap


##############################################
#
#  Dimensions and field
#
##############################################

m = 10
n = 20
# field = "real"
field = "complex"
print("Dimensions: \t(%d, %d)" %(m,n))
print("Field: \t \t \t" + field)
lb = lowerBound(m, n, field)
print("Lower bound: \t%.6f" %lb)


##############################################
#
#  TAAP hyperparameters
#
##############################################

beta    = 2.0
N_budg  = 100000
tau     = 10**(-6)
N_p     = 100
eps_p   = 10**(-1)
eps_s   = 10**(-3)
accel   = True
verbose = True


##############################################
#
#  Run TAAP
#
##############################################

n_runs = 3
F_0, F_run = initializeUnitFrame(m, n, field, n_runs=n_runs)


for run_index in range(n_runs):
    mu_0 = mutualCoherence(F_0[run_index], field)
    
    start_time = time.time()
    F_run[run_index], mu_run, N_tot_run = taap(F_0[run_index], m, n, field, beta=beta, N_budg=N_budg, tau=tau, N_p=N_p, eps_p=eps_p, eps_s=eps_s, acceleration=accel, verbose=verbose)
    mu_run = mutualCoherence(F_run[run_index], field)
    end_time = time.time()
    delta_time = end_time-start_time
    time_per_it = delta_time/N_tot_run
    print("Runtime for run %d: \t%.3f" %(run_index+1,delta_time))
    print("Runtime per iteration: \t%.8f" %time_per_it)


