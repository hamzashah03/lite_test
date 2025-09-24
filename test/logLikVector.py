#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from scipy.stats import binom, genpareto
from bernoulli import Bernoulli, logLik, fitBernoulli


# In[7]:

# _Class to avoid naming clash (rename or confirm if this is an issue)
class logLikVec(np.ndarray):                           # Subclass of a np.array to keep np.array while adding more attributes
    """
    Class to represent a vector of log-likelihood contributions from individual observations.
    """

    def __new__(cls, input_array, nobs, df):
        # Create np.array object and cast it as a logLikVector class
        obj = np.asarray(input_array).view(cls)
        obj.nobs = nobs
        obj.df = df

        return obj
    
    # This part might be unnecessary as in theory should adjust the vector or create a new logLikVector from view casting but probably worth keeping
    def __array_finalize__(self, obj):          
        if obj is None: 
            return

        # Copy over the previous attributes
        getattr(obj, 'nobs', None)            # In theory if the np.array got sliced into a smaller array technically the no. of observations should decrease
        getattr(obj, 'df', None)
    
    def __repr__(self):
        return f"LogLikVector(values={np.array2string(self, precision=4)}, nobs={self.nobs}, df={self.df})"
    
    def logLik(self):
        """
        Returns a logLik object representing the sum of the log-likelihood contributions
        """
        return logLik(value = np.sum(self), nobs = self.nobs, df = self.df)

    def total(self):
        """Return the sum of all values in the array"""
        return np.sum(self)
        

# Creating separate functions for logLikVector.Bernoulli and GP

def logLikVector_Bernoulli(object, pars = None, **kwargs):
    """
    Returns a vector of length 1 containing a value of the Bernoulli success probability.

    Arguments
    ---------
    object : Bernoulli
        A fitted bernoulli model
    
    pars : list
        The parameters for the fitted model. In this case it's the probability of success
    """
    # If parameter estimates aren't provided then extract from fitted object
    if pars == None:
        pars = [object.coef()]
    elif type(pars) != list:            # Ensure pars is a list
        pars = list(pars)           # Need to add a check to ensure that pars is inputted as either a list or np.array
    
    n_pars = len(pars)
    prob = pars[0]

    if prob < 0 or prob > 1:
        val = float('-inf')
    else:
        val = binom.logpmf(object.obs_data, object.nobs, prob)

    return logLikVec(val, object.nobs, n_pars)


def logLikVector_GP(object, pars = None, **kwargs):
    """
    Returns a vector of length 2 containing the values of the generalised Pareto scale sigma and shape

    Arguments
    ---------
    object : GP
        A fitted Generalised Pareto model
    
    pars : list
        The parameters for the fitted model Generalised Pareto model
    """
    # If parameter estimates aren't provided then extract from fitted object
    if pars == None:
        pars = list(object.coef)
    elif type(pars) != list:            # Ensure pars is a list
        pars = list(pars)
    
    n_pars = len(pars)
    sigma = pars[0]
    xi = pars[1]
    
    

    # Calculate log-likelihood contributions



def logLikVector(object, pars = None, **kwargs):            # Dunno if need the kwargs here or not (Prob need since _bernoulli and _GP has kwargs
    if type(object) == Bernoulli:
        return logLikVector_Bernoulli(object, pars)
    elif isinstance(object, GP):
        return logLikVector_GP(object, pars)
    else:
        raise ValueError("'object' must be either a Bernoulli or GP object")    # Define logLik class to be used for future functions


# In[ ]:


# get_ipython().system('jupyter nbconvert --to python bernoulli.ipynb')


# %%
