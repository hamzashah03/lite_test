#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.optimize import minimize
from bernoulli import logLik


# In[2]:


class GP:
    def __init__(self, mle, nexc, vcov, maxLogLik, exceedances, threshold):
        self.mle = mle
        self.nexc = nexc
        self.vcov = vcov
        self.maxLogLik = maxLogLik
        self.exceedances = exceedances
        self.threshold = threshold
    
    def coef(self):                                  # Some further arguments may exist
        return self.mle
    
    def vcov(self):
        return self.vcov
    
    def nobs(self):
        return self.nexc
    
    def logLik(self):
        return logLik(self.maxLogLik, self.nobs(), len(self.coef()))
        

def grimshaw_gp_mle(excesses): #this version approximates the result but doesn't literally replicate all iterative safeguards and zero finding logic in Grimshaw
    y = np.asarray(excesses)
    
    def neg_log_lik(params):
        a, k = params  # a = sigma, k = -xi
        sigma = a
        xi = -k
        if sigma <= 0:
            return np.inf
        term = 1 + xi * y / sigma
        if np.any(term <= 0):
            return np.inf
        return np.sum(np.log(sigma) + (1 / xi + 1) * np.log(term))
    
    sigma0 = np.mean(y)
    xi0 = 0.1
    res = minimize(neg_log_lik, x0=[sigma0, -xi0], method='L-BFGS-B', bounds=[(1e-8, None), (None, None)])
    a, k = res.x
    return {"a": a, "k": k}  # matches R return of revdbayes::grimshaw_gp_mle

def GPObsInfo(pars, excesses, eps=1e5, m=3):
    """
    Calculates the observed information matrix for a random sample (negated Hessian matrix of the generalised Pareto independence log-likelihood
    """
    
    if eps <= 0:
        raise ValueError("'eps' must be positive")
    if m < 0:
        raise ValueError("'m' must be non-negative")
        
    y = excesses
    s = pars[0] #sigma
    x = pars[1] #xi
    
    i = np.empty((2,2))
    
    i[0, 0] = -np.sum((1 - (1 + x) * y * (2 * s + x * y) / (s + x * y)**2) / s**2)
    i[0, 1] = i[1, 0] = -np.sum(y * (1 - y / s) / (1 + x * y / s)**2 / s**2)
    
    # Note that direct calculation of i[1,1] is unreliable for x close to zero.
    # If abs(x) < eps then we expand the problematic terms (all but t4 below)
    # in powers of z up to z ^ 2. The terms in 1/z and 1/z^2 cancel leaving only
    # a quadratic in z.
    
    z = x / s
    zy = z * y
    t0 = 1 + zy
    t4 = y**2 / t0**2
    
    if any(t0 <= 0):
        raise ValueError("The likelihood is 0 for this combination of data and parameters")
        
    if abs(x) < eps: 
        j = np.arange(m+1)
        def sum_fn(zy):
            return np.sum([(-1)**jj * (jj**2 + 3*jj + 2) * zy_val**jj / (jj + 3) for jj in m_idx])
        tsum = np.array([sum_fn(zy_val) for zy_val in zy])
        i[1,1] = np.sum(y**3 * tsum / s**3 - t4 / s**2)
    else:
        t1 = 2 * np.log1p(zy) / z**3
        t2 = 2 * y / (z**2 * t0)
        t3 = y**2 / (z * t0**2)
        i[1,1] = np.sum((t1 - t2 - t3)/s**3 - t4 / s**2)
        
    # ignored dimnames
    return i
    

def fitGP(data, u):
    """
    Fit a generalised Pareto distribution using maximum likelihood estimation.

    Parameters
    ----------
    data: list or np.array
        A numeric vector of raw data
    u: numeric scalar
        Extremal value threshold

    Returns
    -------
    Bernoulli
        A new GP object representing a fitted generalised Pareto model
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValueError("'data' must be a vector (list or numpy array)")
    if len(u) != 1:
        raise ValueError("'u' must have length 1")
        
    data = data[~np.isnan(data)]
    excesses = data[data > u] - u
    grimshaw_fit = grimshaw_gp_mle(excesses)
    
    mle = {"sigma[u]": sigma, "xi": xi}
    
    nexc = len(excesses)
    
    vcov = np.linalg.inv(gpObsInfo([sigma, xi], excesses))
    
    sc = np.full(self.nexc, sigma)
    
    maxLogLik = -np.sum(np.log(sc)) - np.sum(np.log1p(xi * excesses / sc) * (1 / xi + 1))
    
    exceedances = data[data > u]
    
    threshold = u
                                        
    res = GP(mle, nexc, vcov, maxLogLik, exceedances, threshold)
    
    return res
    


# In[ ]:




