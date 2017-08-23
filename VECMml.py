
import numpy as np
from numpy.linalg import eig
import scipy.linalg as la
from scipy.optimize import root

import matplotlib.pyplot as plt

class VECM(object):

    '''
    estimate a reduced form VECM model following Luetkepohl 2017 chapter Section 3.2.2/
    MATLAB Script for figure 11.3
    '''

    def __init__(self, data=None, p=None, r=None, var_names=None, shock_names=None):

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
        self.var_names = var_names
        self.B0inv = None
        
        if shock_names is None:
            self.shock_names = var_names
        else:
            self.shock_names = shock_names
        
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
        self.beta = beta  
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
        K = Sigma.shape[0] # number of variables
        self.K = K 

    def normalize(self):
        alpha, beta , Gamma = self.alpha, self.beta, self.Gamma
        K, p = self.K, self.p

        Q = beta[:self.r,:]
        alpha_norm = alpha @ Q.transpose()
        beta_norm = beta @ la.inv(Q)
        Gamma_norm = Gamma[:,1:self.q+1]

        self.alpha = alpha_norm
        self.beta = beta_norm
        self.Gamma_norm = Gamma_norm
        
        '''
            make companion form based on normalized alpha and beta
            TODO: comnpanion form general for p lags, not only for one.
        '''
        
        KP = K * p
        I = np.identity(KP-K)
        I = np.concatenate((I, np.zeros([KP - K,K])), axis=1)
        
        
        slopes = Gamma[:,-((p-1) * (K)):]
        gam_lag = slopes
        
        if p > 2:
            for lag in range(2,p):
                print(lag)
                first = KP-K - lag*K
                last = first + K
                gam_lag = slopes[:,first:last]
                gam_lag_minus1 = slopes[:,(first+K):(last+K)]
                gam_lag += gam_lag_minus1
                slopes[:,(first):(last)] = gam_lag            

        A1 = alpha_norm @ beta_norm.T + np.identity(K) + gam_lag
        params = np.concatenate((A1, -1*slopes), axis=1)
        #print('\nparams:\n',params)
        self.companion = np.concatenate((params,I), axis=0)

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
    
    def set_restrictions(self, SR, LR):
        '''
            Specify the zero restrictions to B0inv and Upsilon:
            
            SR, LR : array of size K x K
                entry 0 
        '''
        self.SR = SR
        self.LR = LR
        
    def restriction_errors(self, B0inv_vec):
        Xi = self.Xi
        Sigma = self.Sigma_u
        K = Sigma.shape[0]
        Gamma = self.Gamma
        
        B0inv = B0inv_vec.reshape((self.K, self.K))
        # short run restrictions from helmut
        #0inv = B0inv.flatten()
        #SR = ~self.SR.flatten()
        
        B_err = B0inv[~self.SR]
        
        # LR restrictions
        
        Upsilon = Xi @ B0inv
        #Upsilon = Upsilon.flatten()
        #LR = ~self.LR.flatten()
        Ups_err = Upsilon[~self.LR]
     
        # exact identification 'restrictions'
        Sigma_err = B0inv @ B0inv.T - Sigma
        Sig_err = Sigma_err.flatten()

        
        err_vec = np.concatenate([Sig_err, B_err, Ups_err])
        return err_vec
        
    def get_B0inv(self, start=None):
        '''
            TODO: add normalization so sign
        '''
        if start is None:
            start = np.random.rand(3,3) #np.linalg.cholesky(self.Sigma_u)
        
        settings ={'xtol':1e-10, 'ftol':1e-10, 'maxiter':100000000,
                   'eps':1e-20, 'gtol':1e-20} 
        opt_res = root(self.restriction_errors, start, method='lm',
                       options=settings)
        
        self.opt_res = opt_res
        B0inv = opt_res.x.reshape((self.K, self.K))
        self.B0inv = B0inv 
        
    def get_irfs(self, nsteps, B=None, plot=False, imps=None, resps=None):       
        
        C = self.companion
        K = self.K
        P = self.p
        var_names = self.var_names
        
        
        if B is None:
            B = self.B0inv
        elif B == 'chol':
            B = np.linalg.cholesky(self.Sigma_u)
        else:
            B=np.identity(self.K)
        
        '''  get entire inpulse response matrices :
        big_IRF contains all the impulse responses to every shock in the model
        every row corresponds to the response of one of he K variables and
        every Kth column of the row is the response to the Kth shock of that variable
        '''
        
        IRF = np.concatenate((B, np.zeros([K * P - K, K])), axis=0)
        big_IRF = B
        for i in np.arange(1, nsteps):
            new_IRF = np.dot(C, IRF)
            big_IRF = np.concatenate((big_IRF, new_IRF[:K,:K]), axis=1)
            IRF = new_IRF
    
        '''reorganize such that irfs_organized[periods, #impulse, #response]
        '''
        
        irfs_organized = np.zeros((nsteps,K,K))
        for impulse in range(K):
            subset = list(np.arange(impulse ,nsteps*K , K))
            irfs_organized[:,impulse,:]=  big_IRF[:,subset].T
        
        self.irfs = irfs_organized
        
        if plot == True:
            if imps is None:
                imps = np.arange(len(self.shock_names))
            
            if resps is None:
                resps = np.arange(len(self.var_names))

            n_imp = len(imps)
            n_res = len(resps)
            
            fig, axes = plt.subplots(n_res, n_imp)

            if n_imp == 1: axes = axes[:,np.newaxis]

            for r in range(n_res):
                for i in range(n_imp):
                    if r==0 : axes[r,i].set_title(self.shock_names[i])
                    axes[r,i].plot(np.zeros(nsteps), 'k:')
                    axes[r,i].plot(irfs_organized[:,i,r], label=self.var_names[r])
                    if i==0: axes[r,i].set_ylabel(self.var_names[r])
                
            #fig.suptitle('Impulse responses', fontsize=16)
            plt.tight_layout()
            return fig
