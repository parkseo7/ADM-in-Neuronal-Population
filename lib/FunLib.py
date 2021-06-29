from __future__ import division, print_function

import os
import numpy as np
from numpy.polynomial.polynomial import polyval
import math
from math import pi

def coincidenceFactor(kappa, W, tau):
    '''
    The coincidence factor function with respect to kappa and tau. Returns the matrix of coincidences.
    '''
    
    return np.exp(-kappa/2*sumSquaredDiffs(W, tau))


def sumSquaredDiffs(W, tau):
    '''
    The summation of squared differences along each column of the square matrix tau, used in the
    exponential term of the coincidence factor.
    '''
    
    diffs = np.zeros(tau.shape)
    for i in range(tau.shape[0]):
        # Along each column of delays, subtract v_i - v_j
        v_i = np.reshape(tau[i,:], -1)
        W_i = np.reshape(W[i,:] > 0, -1)
        boolW_i = (W_i[:,np.newaxis] > 0).astype('float64')
        diff_i = np.sum(boolW_i * np.abs(v_i - v_i[:,np.newaxis])**2, 0)
        diffs[i,:] = diff_i
     
    return diffs

def sumDiffs(W, tau):
    '''
    The summation of differences along each column of the square matrix tau, used in the learning
    step.
    '''
    
    diffs = np.zeros(tau.shape)
    for i in range(tau.shape[0]):
        # Along each column of delays, subtract v_i - v_j
        v_i = np.reshape(tau[i,:], -1)
        W_i = np.reshape(W[i,:] > 0, -1)
        boolW_i = (W_i[:,np.newaxis] > 0).astype('float64')
        diff_i = np.sum(boolW_i * (v_i - v_i[:,np.newaxis]), 0)
        diffs[i,:] = diff_i
     
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
