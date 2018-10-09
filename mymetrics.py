# The difference between mymetrics and mylosses is that mymetrics are implemented
# in numpy while mylosses are implements in Keras/TensorFlow.

import numpy as np

def tnr(y, yhat): # yhat=0 | y=0
    div = np.sum(1-y)
    if div:
        return np.sum((1-y) * (1-np.round(yhat))) / np.sum(1-y)
    return 1

def fnr(y, yhat): # yhat=0 | y=1
    div = np.sum(y)
    if div:
        return np.sum(y * (1-np.round(yhat))) / np.sum(y)
    return 0
    #return np.nansum(y * (1-np.round(yhat))) / np.sum(y)

def crossentropy(y, yhat):
    yhat = np.clip(yhat, 1e-07, 1-1e-07)
    return -np.mean(y*np.log(yhat) + (1-y)*np.log(1-yhat))

def fnr_approx(y, yhat): # yhat=0 | y=1
    return np.sum(y * (1-yhat)) / np.sum(y)

def entropy_averse_term(rho):
    def fn(y, yhat):
        rho_hat = fnr_approx(y, yhat)+1e-7
        return max(0, -np.log(1-rho_hat)-rho)
    return fn
