import numpy as np

class logLik:
    """
    Class representing the maximised log-likelihood
    """
    def __init__(self, value, nobs, df):
        self.value = value
        self.nobs = nobs
        self.df = df

    # String representation of the class
    def __repr__(self):
        return f"<Value = {self.value:.4f}, nobs = {self.nobs}, df = {self.df}>"

class Bernoulli:
    """
    Frequentist inference for the Bernoulli distribution
    """
    def __init__(self, mle, vcov, nobs, obs_data, maxLogLik):
        """
        Initialise a fitted Bernoulli model using mle
        """
        self.mle = mle
        self.vcov = vcov
        self.nobs = nobs
        self.obs_data = obs_data
        self.maxLogLik = maxLogLik

    def coef(self):
        """ Return MLE of probability of success """
        return self.mle
    
    def vcov(self):
        """ 
        Return the estimated variance of the estimator of the probability of success.
        No adjustments for cluster dependance has been made.
        """
        return self.vcov
    
    def nobs(self):
        """ Return the number of observations """
        return self.nobs
    
    def logLik(self):
        """ Return a logLik object for the Bernoulli model"""
        res = logLik(value = self.maxLogLik, nobs = self.nobs, df = 1)          # df always 1 but in R package it's shown as a function
        return res
        


    
def fitBernoulli(data):
    """
    Fit a Bernoulli distribution using maximum likelihood estimation.

    Parameters
    ----------
    data: list or np.array
        A sequence of 1's and/or 0's, (Indicating success or failure)

    Returns
    -------
    Bernoulli
        A new Bernoulli object representing a fitted bernoulli model
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValueError("'data' must be a vector (list or numpy array)")
    
    # Store raw data
    data = np.array(data, dtype = float)

    # Remove outliers
    obs_data = data[~np.isnan(data)]

    # Calculate MLE
    mle = np.mean(obs_data)

    # Number of Observations
    nobs = len(obs_data)

    # Variance Covariance Matrix 
    vcov = np.array(mle * (1-mle) / nobs)

    # No. of success and failures
    n1 = sum(obs_data)                              # Should we name this n_succ and n_fail instead?
    n0 = nobs - n1

    # Compute Maximum Log-Likelihood
    maxLoglik = n1 * np.log(mle) + n0 * np.log(1 - mle)

    res = Bernoulli(mle, vcov, nobs, obs_data, maxLoglik)

    return res

