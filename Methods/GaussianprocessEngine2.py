import numpy as np

import matplotlib.pyplot as plt

import pyGPs
from sklearn.metrics import mean_absolute_error
from Methods.crossvalidation import KFoldWithGap
from sklearn.grid_search import GridSearchCV




import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

__author__ = 'christiaanleysen'

'''
PyGP Gaussian process regression
'''


def predictConsumption(trainSetX, trainSetY, testSetX, testSetY):
    """
    predicts the consumption

    Parameters:
    -----------
    trainSetX: training feature set
    trainSetY: training value set
    testSetX: test feature set
    testSetY: test value set

    Returns:
    --------
    a prediction of the consumption
    """

    model = pyGPs.GPR()      # specify model (GP regression)
    #TODO kernel

    k = pyGPs.cov.Linear()#+pyGPs.cov.RBF()#log_ell=0.00001, log_sigma=6.1)#+pyGPs.cov.RQ(log_ell=0.00001, log_sigma=6.1 )#pyGPs.cov.Poly(d=3)+pyGPs.cov.Linear()#pyGPs.cov.RQ()#pyGPs.cov.Linear() + pyGPs.cov.RBF()+pyGPs.cov.Matern()#log_ell=-1., log_sigma=0.8 ) # product of kernels

    #k = pyGPs.cov.RBF(log_sigma=6.1)#log_ell=0.00001, log_sigma=6.1)#+pyGPs.cov.RQ(log_ell=0.00001, log_sigma=6.1 )#pyGPs.cov.Poly(d=3)+pyGPs.cov.Linear()#pyGPs.cov.RQ()#pyGPs.cov.Linear() + pyGPs.cov.RBF()+pyGPs.cov.Matern()#log_ell=-1., log_sigma=0.8 ) # product of kernels
    #k = pyGPs.cov.RQ(log_ell=0.00001, log_sigma=6.1 )
    #m = pyGPs.mean.Linear(D=trainSetX.shape[1])
    m = pyGPs.mean.Const(trainSetY.mean())
    #model.setNoise( log_sigma = -50 )
    model.setPrior(kernel=k,mean=m)
    #model.setOptimizer("BFGS", num_restarts=5,covRange=[(-0.001,0)])




    #model.getPosterior2( x=testX, y=testY, der=True,optimizeFlag=True)

    model.getPosterior(trainSetX,trainSetY,False)# fit default model (mean zero & rbf kernel) with data
    model.optimize(trainSetX,trainSetY,numIterations=5)     # optimize hyperparamters (default optimizer: single run minimize)
    #model.getPosterior(trainSetX,trainSetY,False)


    predictedSetY, ysigma2, fm, fs2, lp = model.predict(testSetX)

    sigma = np.sqrt(ysigma2)
    MAE = mean_absolute_error(testSetY, predictedSetY)
    if np.mean(np.mean(testSetY)) == 0:
        MRE = 50
    else:
        MRE = (MAE/(np.mean(testSetY)))*100

    #return gp,MSE,sigma, predictedSetY,trainingR2Score,testR2Score
    return predictedSetY,sigma,MAE,MRE

def predictConsumptionSparse(trainSetX, trainSetY, testSetX, testSetY):
    """
    predicts the consumption with sparse GPR algorithm

    Parameters:
    -----------
    trainSetX: training feature set
    trainSetY: training value set
    testSetX: test feature set
    testSetY: test value set


    Returns:
    --------
    a prediction of the consumption
    """
    model = pyGPs.GPR_FITC()# specify model (GP regression)
    model.setData(trainSetX, trainSetY)
    k = pyGPs.cov.Linear() + pyGPs.cov.RBF() # product of kernels
    #m = pyGPs.mean.Linear(D=trainSetX.shape[1])
    #m =pyGPs.mean.Const(100)
    #model.setNoise( log_sigma = np.log(1.8) )
    model.setPrior(kernel=k)#,mean=m)



    #model.getPosterior2( x=testX, y=testY, der=True,optimizeFlag=True)
    print('posterior zoeken')
    #model.getPosterior(trainSetX,trainSetY,True)# fit default model (mean zero & rbf kernel) with data
    print('posterior gevonden')
    model.optimize()     # optimize hyperparamters (default optimizer: single run minimize)
    print('optimization done')
    predictedSetY, ysigma2, fm, fs2, lp = model.predict(testSetX)

    sigma = np.sqrt(ysigma2)
    MAE = mean_absolute_error(testSetY, predictedSetY)
    #predicted = predictedSetY.flatten()#.tolist()
    #plotGaussianProcess(testSetY,predicted,np.sqrt(ys2),'testpyGP',0,7000)

    #return gp,MSE,sigma, predictedSetY,trainingR2Score,testR2Score
    return predictedSetY,sigma,MAE



#Plot observation vs prediction
def plotObsVsPred(testSetY,predictedSetY):
    """
    makes plot of the observed vs the predicted values
    Parameters:
    -----------

    testSetY: test value set
    predictedSetY: predicted value set


    Returns:
    --------
    a plot of the observed vs the predicted values
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(testSetY, predictedSetY,color='red')
    plt.xlabel('observatie')
    plt.ylabel('Voorspelling')
    plt.title('Voorspelling vs. observatie')
    plt.plot([min(testSetY), max(testSetY)], [min(testSetY), max(testSetY)], 'green')
    plt.xlim([min(testSetY), max(testSetY)])
    plt.ylim([min(testSetY), max(testSetY)])
    plt.show()

#plot the Gaussian process regression results
def plotGaussianProcess(testSetY, predictedSetY, sigma,title,min,max):
    """
    makes plot of the gaussian process output
    Parameters:
    -----------

    testSetY: test value set
    predictedSetY: predicted value set
    sigma: variance of the prediction
    title: string with name of plot
    min: minimum value of the Y-axis
    max: maximum value of the Y-axis

    Returns:
    --------
    a plot of the predicted values and their variance and the observed values
    """

    fig = plt.figure(figsize=(20, 6))

    plt.ylabel('Consumptie (kWh)', fontsize=13)
    plt.title('Gaussian Process Regressie: '+str(title))
    plt.legend(loc='upper right')
    plt.xlim([0, len(testSetY)])
    plt.ylim([min, max])

    #xTickLabels = pd.DataFrame(predictedSetY.index[np.arange(0,len(predictedSetY),1)])
    #xTickLabels['date'] = xTickLabels['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    xTickLabels = np.arange(0,len(predictedSetY),1)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, len(predictedSetY), 1))
    ax.set_xticklabels(labels=xTickLabels, fontsize=9, rotation=90)

    plt.plot(predictedSetY, 'b-', label=u'Voorspelling',color='green')
    plt.plot(testSetY, 'r.', markersize=10, label=u'Observaties',color='red')
    x = range(len(testSetY))
    print('predictedSetY',predictedSetY)
    sigma = sigma.flatten()
    print('sigma',sigma)
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([predictedSetY - 1.9600 * sigma, (predictedSetY + 1.9600 * sigma)[::-1]]),
             alpha=.3, fc='b', ec='None', label='95% betrouwbaarheidsinterval',facecolor='green')


def param_tuning(trainSetX, trainSetY, clf, parameters, cv=8, verbose=False):
    """
    Method for tuning the paratmers of a model. As evaluation method mean absolute
    error is used. GridSearchCV is called with n_jobs=-1 to test all combinations.

    Parameters:
    ----------
    dataset: usually a training set on which to test the parameter combinations
    predictors: list of predictor names
    response: the name of the response variable
    clf: the classifier to test
    parameters: the parameters to tune via grid search
    cv: cross validation generator or integer for n-fold CV. Default is 10-fold
        cross validation with a gap of 14 days between train and validation sets
    verbose: flag to enable print out of results

    Returns:
    --------
    clf: the classifier with the best working parameter calibration


    """
    if (cv == None):
        cv = KFoldWithGap([trainSetX,trainSetY], 2*24, 8)

    train_input = trainSetX
    train_target = trainSetY
    clf = GridSearchCV(clf, parameters, cv=cv, n_jobs=-1, scoring='mean_absolute_error', verbose=False)
    clf.fit(train_input, train_target)
    if (verbose):
        print("========== PARAMETER TUNING RESULTS ===========")
        print ("Winner:")
        print (clf.best_estimator_)
        #print("Scores:")
        #for params, mean_score, scores in clf.grid_scores_:
        #    print("%0.4f (+/-%0.4f) for %r" % (mean_score, scores.std() / 2, params))
    return clf
