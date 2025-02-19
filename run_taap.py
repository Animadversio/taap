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

m = 64
n = 128
field = "real"
# field = "complex"
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
eps_p   = 10**(-3)
eps_s   = 10**(-1)
accel   = True
verbose = True


##############################################
#
#  Run TAAP
#
##############################################

n_runs = 3
F_0, F_run = initializeUnitFrame(m, n, field, n_runs=n_runs)
mu_run = np.zeros(n_runs)
N_tot_run = np.zeros(n_runs)
duration_run = np.zeros(n_runs)
time_per_it_run = np.zeros(n_runs)

for run_index in range(n_runs):
    mu_0 = mutualCoherence(F_0[run_index], field)
    
    start_time = time.time()
    F_run[run_index], mu_run[run_index], N_tot_run[run_index] = taap(F_0[run_index], m, n, field, beta=beta, N_budg=N_budg, tau=tau, N_p=N_p, eps_p=eps_p, eps_s=eps_s, acceleration=accel, verbose=verbose)
    end_time = time.time()
    duration_run[run_index] = end_time-start_time
    time_per_it_run[run_index] = duration_run[run_index]/N_tot_run[run_index]
    print("Runtime for run %d: \t%.3f" %(run_index+1,duration_run[run_index]))
    print("Runtime per iteration: \t%.6f" %time_per_it_run[run_index])
    print("Final mutual coherence: %.6f" %mu_run[run_index])
    # print(F_run[run_index])
    print()


