
import numpy as np
from numpy.linalg import eig
import scipy.linalg as la

class VECM(object):
    
    '''
    estimate a reduced form VECM model following Luetkepohl 2017 chapter Section 3.2.2/ 
    MATLAB Script for figure 11.3
    '''
    
    def __init__(self, data=None, p=None, r=None):
        
        '''
        Inputs
        ---
            Data : array-like 
                matrix containing the data of shape (T, q) where T is the 
                number of observations and q is the number of variables 
            p : int
                integer specifying the lagorder
            r : int
                integer specifying the cointegration rank i.e. number of permanent shocks
                
        Parameters/Objects
        ---
            y : array
                full original data series of shape (q, T)
            dy : array
                differenced series of length T-p-1
            T : int
                number of observations
            q : int
                number of variables
        '''
        
        self.p = p
        y = data
        self.r = r
        T, q = data.shape
        
        ydiff = y[1:,:] - y[:-1,:] # the last column in y is the most recent observation
        ydiff = ydiff.transpose()
        y = y.transpose()
        
        ''' create the elements of the VECM in VAR(1)-form (equation 3.2.6) '''
        
        dy = ydiff[:, p-1:] # selecting the common sample after differencing
        X = np.ones((1, T-p))
        for i in range(1,p): # should go from 1 to p
            X = np.concatenate([X, ydiff[:, p-i-1:T-i-1]])
        Xt = X.transpose()
        y = y[:, p:T]
        
        ''' concentrate out the short run dynamics to estimate first the long rund parameters '''
        
        R0 = dy - ( (dy @ Xt) @ la.pinv(X @ Xt) ) @ X # residuals of regression dy on Xt-1
        
        R1 = y - ( (y @ Xt)  @ la.pinv(X @ Xt) ) @ X # residuals of regression y on Xt-1
        print(R1[:,:10])
        S00 = (R0 @ R0.transpose()) / (T - p)
        S11 = (R1 @ R1.transpose()) / (T - p)
        S01 = (R0 @ R1.transpose()) / (T - p)
        
        iS11sq = la.inv(la.sqrtm(S11))
        S01t = S01.transpose()
        Smat = iS11sq @ S01t @ la.inv(S00) @ S01 @ iS11sq
        
        ''' compute the ML estmates for alpha and beta '''
        Lam, B = eig(Smat)

        print(Lam)

        
        