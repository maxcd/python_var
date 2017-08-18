import numpy as np
from numpy.linalg import det
from scipy.linalg import inv, pinv
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats
import matplotlib.pyplot as plt


def cll(bmat, amat, Sigma_u, nobs):
     '''
     Evaluate the concetrated log likeihood function
     for the B-model as presented in
     Lütkepohl 2005, p. 373.

     I checked this function with the corresponding function
     from the Vars package in R. It gives the same result.

     ---
     In
     ---
        bmat : np.array
            matrix specifying which shock impacts which variable
        amat : array
            matrix specifying the conpemporaneous impacts ampung the
            endogenous variables
        Sigma_u : array
            reduced form covariance matrix
        nobs : scalar
            number of observations
    ---
    Out
    ---
        cll : float
            value of the negative concentrated loglikelihood function
     '''

     T = nobs
     K = bmat.shape[0] # number of endogenous variables

     c = -1 * (K * T) / 2* np.log(2 * np.pi)
     ab = amat @ pinv(bmat)
     abt = ab.transpose()
     abba = abt @ ab

     val = ( c + T/2 * np.log( det(amat) ** 2)
           - T/2 * np.log( det(bmat) ** 2 )
           - T/2 * np.trace(abba @ Sigma_u)
           )

     return -val

def criterion(bvec, *args):
    nobs, Sigma_u  = args

    K = Sigma_u.shape[0]
    amat = np.identity(K)

    'construct a lower triangular matrix'
    bmat = np.zeros((K, K))
    np.fill_diagonal(bmat, bvec[-K:])
    bmat[1,0] = bvec[0]
    bmat[2, 0:2] = bvec[1:3]

    loglik = cll(bmat, amat, Sigma_u, nobs)

    return loglik
