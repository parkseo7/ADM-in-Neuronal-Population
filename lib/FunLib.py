from __future__ import division, print_function

import os
import numpy as np
from numpy.polynomial.polynomial import polyval
import math
from math import pi


# MAIN FUNCTIONS
def coincidenceFactorGauss(kappa, W, tau):
    '''
    An alternative way of computing the coincidence factor, so that differences are less punishing to the overall coincidence. 
    Here, we need to also adjust the gradient descent as well.
    '''
    
    coincidence = np.zeros(tau.shape)
    for i in range(tau.shape[0]):
        # Along each column of delays, subtract v_i - v_j
        v_i = np.reshape(tau[i,:], -1)
        W_i = np.reshape(W[i,:], -1)
        W_sum = np.sum(W_i)
        if W_sum != 0:
            W_i = W_i / W_sum
        diff_i = np.abs(v_i - v_i[:,np.newaxis])**2
        sum_i = np.sum(W_i * np.exp(-diff_i/(2*kappa)), 1)
        coincidence[i,:] = sum_i
    
    return coincidence


def derivLearningRateSlow(W, tau, kappa, gamma0, gamma, rates, inds):
    '''
    Return the derivative of the objective function with respect to the delay with index = inds.
    This function is to check to see if the objective function increases through gradient descent.
    Will likely lead to slow iterations.
    '''
    i = inds[0]
    j = inds[1]
    
    tau_i = tau[i,:]
    W_i = np.reshape(W[i,:], -1)
    W_sum = np.sum(W_i)
    if W_sum == 0:
        W_sum = 1.0
    W_i = W[i,:] / W_sum
    N = tau.shape[0]
 
    # Set up the linear system for dr_k/dtau_ij:
    A = np.eye(N) - W*gamma/N
    b = np.zeros((N,))
    
    diffTau_ij = tau_i - tau[i,j]
    gauss_i = diffTau_ij * np.exp(-diffTau_ij**2 / (2*kappa))
    
    derivGamma_i = W_i[j] * gauss_i
    coeff = gamma0 / kappa
    derivGamma_i[j] = coeff * np.sum(W_i * gauss_i)
   
    b[i] = np.sum(W[i,:] * derivGamma_i * rates) / N
    
    # Solve for linear system
    derivRates = np.linalg.solve(A,b)
    
    # Derivative:
    derivLearning = np.sum(rates * derivRates)
    
    #return (derivLearning, derivRates, b[i], derivGamma_i)
    return derivLearning


# OLD CODE
def derivCoincidence(W, tau, kappa, gamma0, ind):
    '''
    At row = ind, obtain the non-zero derivatives of the coincidence factor with respect to tau_{ind,k} for all k. Each index k results
    in a vector of non-zero derivatives.
    '''
    
    tau_i = tau[ind,:]
    W_i = W[ind,:]
    diff_i = tau_i[:,np.newaxis] - tau_i
    gauss_i = diff_i * np.exp(-diff_i**2/(2*kappa))
    N = tau.shape[0]
    
    # Each row vector of the output derivative matrix is the derivative of gamma corresponding tau_{i,row}
    derivGamma = np.zeros(tau.shape)
    
    # Non-diagonal entries
    derivGamma = W_i[:,np.newaxis] * gauss_i
    
    # Diagonal entries
    sumGamma = np.sum(W_i * derivGamma, 1)
    np.fill_diagonal(derivGamma, sumGamma)
    
    # Multiplier
    coeff = gamma0 / (np.sqrt(2*pi*kappa)*kappa*N)
    return coeff * derivGamma
    

def derivLearningRate(W, tau, kappa, gamma0, gamma, rates, ind):
    '''
    At row = ind, obtain the derivative dL(r)/dtau of the learning rate, to be directly used for gradient descent with eta.
    '''
    
    # Obtain coincidence derivatives
    derivGamma = derivCoincidence(W, tau, kappa, gamma0, ind)
    N = tau.shape[1] # Number of columns
    
    derivL = np.zeros((N,))
    
    # Multiplicative matrix is constant with respect to indices:
    derivRateMult = np.eye(N) - (W * gamma)/N
    
    # Solve for linear equation, derivRateMult * drdtau = derivRateConst:
    for j in range(N):
        
        # Skip if there's no connection
        if W[ind,j] == 0:
            continue 
            
        derivGamma_j = derivGamma[j,:]
        derivGammaMat_j = np.zeros(gamma.shape)
        derivGammaMat_j[ind,:] = derivGamma_j
        derivRateConst = np.matmul(W * derivGammaMat_j, rates) / N
        
        # Solve linear system
        derivRate_j = np.linalg.solve(derivRateMult, derivRateConst)
        derivL[j] = np.sum(rates *derivRate_j)
    
    return derivL

    
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
        W_i = np.reshape(W[i,:], -1)
        # boolW_i = (W_i[:,np.newaxis] > 0).astype('float64')
        # diff_i = np.sum(boolW_i * np.abs(v_i - v_i[:,np.newaxis])**2, 0)
        diff_i = np.sum(W_i * np.abs(v_i - v_i[:,np.newaxis])**2, 0)
        diffs[i,:] = diff_i
     
    return diffs


def gradient(W, gamma, tau, r, kappa):
    '''
    Returns the square matrix of gradients to employ onto the iterative step tau^{T+1} = tau^T + eta * gradient
    '''
    
    grad = np.zeros(tau.shape)
    N = tau.shape[0]
    for i in range(N):
        # Along the jth column of delays, take differences tau_{ij} - tau_{il}
        tau_i = np.reshape(tau[i,:], -1)
        gamma_i = np.reshape(gamma[i,:], -1)
        W_i = np.reshape(W[i,:], -1)
        grad_i = np.sum(W_i * (tau_i - tau_i[:,np.newaxis]) * (gamma_i + gamma_i[:,np.newaxis]) * r, 0)
        grad[i,:] = grad_i / N
    
    return -kappa * grad


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
        # boolW_i = (W_i[:,np.newaxis] > 0).astype('float64')
        # diff_i = np.sum(boolW_i * (v_i - v_i[:,np.newaxis]), 0)
        diff_i = np.sum(W_i * (v_i - v_i[:,np.newaxis]), 0)
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
