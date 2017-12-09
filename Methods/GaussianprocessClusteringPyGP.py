__author__ = 'christiaanleysen'

import pyGPs
import scipy
import numpy as np
import timeit



ValuesY = []



def gp_likelihood_independent(hyperparams,model,xs,ys,der=False):
    global ValuesY
    """
    find the aggregated likelihoods of the Gaussian process regression
    Parameters:
    -----------

    hyperparams: hyperparameters for the Gaussian process regression that are used used.
    model: GPR model
    xs: the list of featureset
    ys: the list of valueset


    Returns:
    --------
    the accumulated likelihood of the Gaussian process regression
    """

    #set the hyperparameters
    model.covfunc.hyp=hyperparams.tolist()
    likelihoodList = []


    #accumulate all negative log marginal likelihood (model.nlZ) and the derivative (model.dnlZ)
    all_nlZ = 0
    all_dnlZ = pyGPs.inf.dnlZStruct(model.meanfunc, model.covfunc, model.likfunc)

    for x, y in zip(xs, ys):
        model.setData(x,y)
        if der:
            this_nlZ,this_dnlZ,post = model.getPosterior(der=der)
            all_nlZ += this_nlZ
            all_dnlZ = all_dnlZ.accumulateDnlZ(this_dnlZ)
            likelihoodList.append(this_nlZ)
        else:
            this_nlZ,post = model.getPosterior(der=der)
            all_nlZ += this_nlZ
            likelihoodList.append(this_nlZ)


    likelihoodList = [abs(i/np.sum(abs(i) for i in likelihoodList)) for i in likelihoodList]
    ValuesY = [i*j.tolist() for i,j in zip(ys,likelihoodList)]
    ValuesY = np.array([sum(i) for i in zip(*ValuesY)])

    returnValue = all_nlZ
    if der:
        returnValue = all_nlZ+np.sum(all_dnlZ.cov)+np.sum(all_dnlZ.mean)
    return returnValue

def optimizeHyperparameters_deprecated(initialHyperParameters,model,xs,ys,bounds=[],method='BFGS'):
    """
    Deprecated: works only with lin+RBF kernel
    Optimize the hyperparameters of the general Gaussian process regression
    Parameters:
    -----------

    initialHyperparameters: initial hyper parameters used.
    model: GPR model
    xs: the list of featureset
    ys: the list of valueset
    bounds: the bounds needed for the minimize method (if needed).
    method: the minimize method that is employed e.g. BFGS

    Returns:
    --------
    the optimal hyperparameters and the model
    """

    print('optimizing Hyperparameters...')
    start = timeit.default_timer()
    result = scipy.optimize.minimize(gp_likelihood_independent, initialHyperParameters, args=(model,xs,ys),bounds=bounds,method=method)
    stop = timeit.default_timer()
    print("minimization time:",stop - start)

    hyperparams = result.x
    k =  pyGPs.cov.Linear(log_sigma=hyperparams[0]) + pyGPs.cov.RBF(log_sigma=hyperparams[1],log_ell=hyperparams[2])
    model.setPrior(kernel=k)
    meanYValues = np.mean(ys, axis=0)
    model.setData(xs[0], meanYValues)

    return hyperparams,model


def optimizeHyperparameters(initialHyperParameters,model,xs,ys,bounds=[],method='BFGS'):
    """
    NEW: works with every kernelfunction
    Optimize the hyperparameters of the general Gaussian process regression
    Parameters:
    -----------

    initialHyperparameters: initial hyper parameters used.
    model: GPR model
    xs: the list of featureset
    ys: the list of valueset
    bounds: the bounds needed for the minimize method (if needed).
    method: the minimize method that is employed e.g. BFGS

    Returns:
    --------
    the optimal hyperparameters and the model
    """
    global ValuesY

    print('optimizing Hyperparameters...')
    start = timeit.default_timer()
    result = scipy.optimize.minimize(gp_likelihood_independent, initialHyperParameters, args=(model,xs,ys),bounds=bounds,method=method) #powell gaat lang
    stop = timeit.default_timer()
    print("minimization time:",stop - start)

    hyperparams = result.x
    model.covfunc.hyp=hyperparams.tolist()
    meanYValues = np.mean(ys, axis=0)
    model.getPosterior(xs[0],ValuesY)

    return hyperparams,model













