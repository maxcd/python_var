
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
        for i in range(1,p): # should go from 1 to p for each lag
            X = np.concatenate([X, ydiff[:, p-i-1:T-i-1]])
        Xt = X.transpose()
        # this is the lagged level of the data y_t-1
        y = y[:, p-1:T-1]

        ''' concentrate out the short run dynamics to estimate first the long rund parameters '''

        R0 = dy - ( (dy @ Xt) @ la.pinv(X @ Xt) ) @ X # residuals of regression dy on Xt-1
        R1 = y - ( (y @ Xt)  @ la.pinv(X @ Xt) ) @ X # residuals of regression y on Xt-1
        #print(R1[:,:5])
        #print(R1x[:,:5])
        S00 = (R0 @ R0.transpose()) / (T - p)
        S11 = (R1 @ R1.transpose()) / (T - p)
        S01 = (R0 @ R1.transpose()) / (T - p)

        iS11sq = la.pinv(la.sqrtm(S11))
        S01t = S01.transpose()
        Smat = iS11sq @ S01.transpose() @ la.pinv(S00) @ S01 @ iS11sq

        ''' compute the ML estmates for alpha and beta '''
        #print(S01) # this is actually very diffent from the matrix in Helmuts code..dont know why
        eigenvals, eigenvecs = la.eig(Smat, right=True)
        index = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[index]

        lam = np.diag(eigenvals)
        B = eigenvecs[:,index]
        B = -B

        'compute estimates of the long run parameters'
        beta = iS11sq @ B[:,:r]
        alpha = S01 @ beta @ la.pinv(beta.transpose() @ S11 @ beta)

        'compute the short-run parameters for the differences terms'
        Gamma = ( (dy - alpha @ beta.transpose() @ y) @ X.transpose() @
                 la.pinv(X @ Xt) )

        'compute the Covariance-matrix'
        U = dy - alpha @ beta.transpose() @ y - Gamma @ X
        Sigma = U @ U.transpose() / (T - p)

        self.alpha = alpha
        self.beta = beta  # multiply by -1 give the same value as in helmuts code
        self.Gamma = Gamma
        self.Sigma_u = Sigma
        self.residuals = U
        self.eigenvals = lam
        self.T = T
        self.df = T - p
        self.data = data
        self.y = y
        self.X = X
        self.q = q

        # print('S00:\n')
        # print(S00)
        #
        # print('\nS11:\n')
        # print(S11)
        #
        # print('\nS01:\n')
        # print(S01, '\n')
        #
        # print('this is y\n: with dimensions', y.shape)
        # print(y[:,:3], '\n')
        # print('and the last columns of y\n')
        # print(y[:,-3:])
        '''
        this yields the exact same result as Helmuts code
        '''

    def normalize(self):
        alpha, beta , Gamma = self.alpha, self.beta, self.Gamma

        Q = beta[:self.r,:]
        alpha_norm = alpha @ Q.transpose()
        beta_norm = beta @ la.inv(Q)
        Gamma_norm = Gamma[:,1:self.q+1]

        self.alpha = alpha_norm
        self.beta = beta_norm
        self.Gamma_norm = Gamma_norm

    def perp(self, A):
        '''
        compute the orthogonal complement to a matrix a via SVD
        '''
        K, r = A.shape
        [Q, Lam, P] = la.svd(A)
        A_perp = Q[:, r:K]
        return A_perp

    def get_LR_impact(self):
        alpha, beta , Gamma = self.alpha, self.beta, self.Gamma

        alpha_perp = self.perp(alpha)
        beta_perp = self.perp(beta)

        lr_mat = np.identity(self.q) - Gamma[:,1:self.q+1]
        inv_mat = la.pinv(alpha_perp.transpose() @ lr_mat @ beta_perp)
        Xi = beta_perp @ inv_mat @ alpha_perp.transpose()

        self.Xi = Xi
