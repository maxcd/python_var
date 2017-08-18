# -*- coding: utf-8 -*-
"""
Created on Sat May 20 13:53:14 2017

@author: Max
"""

class var2customVar:
    
    def __init__(self, varresult):
        import numpy as np
        
        self.nlags = varresult.k_ar       
        self.names = varresult.names       
        self.nvars = varresult.neqs        
        self.reduced_form = varresult.params[-(self.nlags * self.nvars):].T        
        self.Sigma = varresult.resid_corr
        
        ''' recover reduced form VAR estimates in companion form for further analysis'''
        
        K = self.nvars # number of endogenous variables
        P = self.nlags #lag order
        KP = K * P
        I = np.identity(KP-K)
        I = np.concatenate((I, np.zeros([KP - K,K])), axis=1)

        self.companion = np.concatenate((self.reduced_form,I), axis=0)
        

    def get_irfs(self, nsteps, B=None):       
        import numpy  as np
        
        C = self.companion
        K = self.nvars
        P = self.nlags
        names = self.names
        if isinstance(B, np.ndarray):
            pass
        else:
            B = np.linalg.cholesky(self.Sigma)
        
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