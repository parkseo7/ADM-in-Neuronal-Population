from __future__ import division, print_function

import os
import numpy as np
from numpy.polynomial.polynomial import polyval
import math
from math import pi

def coincidenceFactor(kappa, tau):
    '''
    The coincidence factor function with respect to kappa and tau. Returns the matrix of coincidences.
    '''
    
    return np.exp(-kappa/2*sumSquaredDiffs(tau))


def sumSquaredDiffs(tau):
    '''
    The summation of squared differences along each column of the square matrix tau, used in the
    exponential term of the coincidence factor.
    '''
    
    diffs = np.zeros(tau.shape)
    for j in range(tau.shape[1]):
        # Along each column of delays, subtract v_i - v_j
        v_j = np.reshape(tau[:,j], -1)
        diff_j = np.sum(np.abs(v_j[:,np.newaxis] - v_j)**2, 1)
        diffs[:,j] = diff_j
     
    return diffs

def sumDiffs(tau):
    '''
    The summation of differences along each column of the square matrix tau, used in the learning
    step.
    '''
    
    diffs = np.zeros(tau.shape)
    for j in range(tau.shape[1]):
        # Along each column of delays, subtract v_i - v_j
        v_j = tau[:,j]
        diff_j = np.sum(v_j[:,np.newaxis] - v_j, 1)
        diffs[:,j] = diff_j
     
    return diffs
    

def rowColMult(r):
    '''
    Given a vector, returns the matrix of r_i*r_j entries.
    '''
    
    return r[:,np.newaxis]*r
    
    
def objectiveFun(r):
    '''
    The objective function, which returns the objective value given a vector of firing rates r
    '''
    
    return 0.5*np.sum(r**2)
    
    
if __name__ == '__main__':
    pass
