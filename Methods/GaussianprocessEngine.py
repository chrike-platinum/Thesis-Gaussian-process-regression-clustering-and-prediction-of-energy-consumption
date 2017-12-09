import numpy as np
from sklearn import gaussian_process
import matplotlib.pyplot as plt
import sklearn.metrics
import pandas as pd
from Methods.crossvalidation import KFoldWithGap
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_absolute_error




import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

__author__ = 'christiaanleysen'
regressionParameter = 'constant'
optimizerParameter = 'Welch'#'fmin_cobyla'
normalizeParameter = False



'''
SKlearn Gaussian process regression
'''


def param_tuning(trainSetX, trainSetY, clf, parameters, cv=4, verbose=False):
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
        print(trainSetX)
        cv = KFoldWithGap([trainSetX,trainSetY], 2*24, 2)

    train_input = trainSetX
    train_target = trainSetY
    clf = GridSearchCV(clf, parameters, cv=cv, n_jobs=-1, scoring='mean_squared_error', verbose=False)
    clf.fit(train_input, train_target)
    if (verbose):
        print("========== PARAMETER TUNING RESULTS ===========")
        print ("Winner:")
        print (clf.best_estimator_)
        #print("Scores:")
        #for params, mean_score, scores in clf.grid_scores_:
        #    print("%0.4f (+/-%0.4f) for %r" % (mean_score, scores.std() / 2, params))
    return clf




#actual Gaussion process regression to predict the consumption
def predictConsumption(trainSetX, trainSetY, testSetX, testSetY,theta, nugget, kernelfunction,tune_params=True):
    """
    predicts the consumption

    Parameters:
    -----------
    trainSetX: training feature set
    trainSetY: training value set
    testSetX: test feature set
    testSetY: test value set
    theta: hyperparam to be used
    nugget: nugget to be used
    kernelfunction: kernelfunction to be used

    Returns:
    --------
    a prediction of the model of the consumption, mean squared error,variance, predictionvalues, R2 TrainingScore, R2 TestScore
    """

     #HH1: 0.05,0.1
    #HH4: 0.0009, 0.01
    #HH4: 0.001, 0.001
    #HH4: 0.0024, 0.001 (*)\
    #HH4: 0.0018, 0.001
    #HH4: 0.0009, 0.001 (*)


    if (tune_params):
        print("Tuning GP parameters....")
        #tuning_parameters = [#{'gamma':  [0.1, 0.01,0.001],
        tuning_parameters=  [{'nugget':[0.0001,0.0009,0.001,0.0018,0.002,0.05,0.1], #[100,50,10,1,0.1, 0.01, 0.001, 0.0001, 0.00001]
                              'theta0': [0.0001,0.001,0.01,0.1,1]}]
        gps = param_tuning(trainSetX, trainSetY, gaussian_process.GaussianProcess(regr=regressionParameter,corr='squared_exponential',random_start= 6, optimizer = optimizerParameter,normalize=False), tuning_parameters, verbose=True)
        train_err = -gps.best_score_
        model = gps.best_estimator_


    #gp = gaussian_process.GaussianProcess(regr=regressionParameter,corr='linear',random_start= 6, optimizer = optimizerParameter,theta0=theta, nugget=nugget)
    model.fit(trainSetX, trainSetY)
    #TODO theta: theta0=theta,


    print('predicting....')
    predictedSetY, MSE = model.predict(testSetX, eval_MSE=True)
    sigma = np.sqrt(MSE)
    trainingR2Score = model.score(trainSetX, trainSetY)
    testR2Score = sklearn.metrics.r2_score(testSetY, predictedSetY)
    MAE = mean_absolute_error(testSetY,predictedSetY)
    if np.mean(np.mean(testSetY)) == 0:
        MRE = 50
    else:
        MRE = (MAE/(np.mean(testSetY)))*100

    #return gp,MSE,sigma, predictedSetY,trainingR2Score,testR2Score
    return model,sigma, predictedSetY,trainingR2Score,testR2Score,MAE,MRE




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

    plt.ylabel('Consumptie (W)', fontsize=13)
    plt.title('Gaussian Process Regressie')
    plt.legend(loc='upper right')
    plt.xlim([0, len(testSetY)])
    plt.ylim([0, 15000])

    xTickLabels = pd.DataFrame(predictedSetY.index[np.arange(0,len(predictedSetY.index),1)])
    xTickLabels['date'] = xTickLabels['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    ax = plt.gca()
    ax.set_xticks(np.arange(0, len(predictedSetY), 1))
    ax.set_xticklabels(labels=xTickLabels['date'], fontsize=9, rotation=90)

    plt.plot(predictedSetY, 'b-', label=u'Voorspelling',color='green')
    plt.plot(testSetY, 'r.', markersize=10, label=u'Observaties',color='red')
    x = range(len(testSetY))
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([predictedSetY - 1.9600 * sigma, (predictedSetY + 1.9600 * sigma)[::-1]]),
             alpha=.3, fc='b', ec='None', label='95% betrouwbaarheidsinterval',facecolor='green')




