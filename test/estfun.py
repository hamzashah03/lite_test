#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def estfun.Bernoulli(x):
    """
    Returns a n * k matrix containing contributions to the score function from n observations for each of the k parameters

    Arguments
    ---------
    x : Bernoulli
        A fitted model object
    """
    U = x.data / x.mle - (1 - x.data) / (1 - x.mle)
    U = U.reshape(-1, 1)
    return U

def estfun.GP(x, eps = 1e-5, m = 3):
    if eps <= 0:
        raise ValueError("'eps' must be positive")
    if m < 0:
        raise ValueError("'m' must be non-negative")
    
    pars = x.coef()
    sigma = pars[0]
    xi = pars[1]
    z = xi / sigma
    
    y = x.exceedances - x.threshold
    zy = z * y
    t0 = 1 + zy
    U = np.full((len(y), 2), np.nan)
    U[:, 0] = -1 / sigma + (xi + 1) *y / (t0 * sigma**2)
    if abs(xi) < eps:
        i = np.arange(0, m + 1)
        def sum_fn(zy):
            return np.sum((-1)**i * (i + 1) * zy_val**i / (i + 2))
        tsum = np.array([sum_fn(zy_val) for zy_val in zy])
        U[:, 1] = y**2 * tsum / sigma**2 - y / (sigma * t0)
    else:
        t1 = np.log1p(zy) / z**2
        t2 = y / (z * t0)
        U[:, 1] = (t1 - t2) / sigma**2 - y / (sigma * t0)
    
    return U


# In[ ]:




