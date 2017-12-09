
__author__ = 'christiaanleysen'
import features.featureMaker as fm
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from Methods.crossvalidation import KFoldWithGap
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing






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





def predictConsumption(trainSetX, trainSetY, testSetX, testSetY,tune_params=True,scaled=False):
    """
    predicts the consumption

    Parameters:
    -----------
    trainSetX: training feature set
    trainSetY: training value set
    testSetX: test feature set
    testSetY: test value set
    tune_params: Boolean for the parameter tuning
    Scaled: Boolean to scale the input

    Returns:
    --------
    a prediction of the consumption
    """

    if scaled:
        trainSetX = np.asarray([preprocessing.scale(element)for element in trainSetX])
        testSetX = np.asarray([preprocessing.scale(element )for element in testSetX])


    if (tune_params):
        print("Tuning SVR parameters....")

        tuning_parameters=  [{'epsilon':[1000,500,100,50,10,1,0.1,0.01, 0.001],
                              'C':      [0.01,0.1,1,10, 100, 1000, 10000]}]
        svrs = param_tuning(trainSetX, trainSetY, LinearSVR(), tuning_parameters, verbose=True)
        train_err = -svrs.best_score_
        model = svrs.best_estimator_




    model.fit(trainSetX,trainSetY)# fit default model (mean zero & rbf kernel) with data

    predictedSetY = model.predict(testSetX)
    MAE = mean_absolute_error(testSetY,predictedSetY)
    if np.mean(np.mean(testSetY)) == 0:
        MRE = 50
    else:
        MRE = (MAE/(np.mean(testSetY)))*100

    return  predictedSetY,testSetY,MAE,MRE


def plot(testSetY, predictedSetY,title,min,max):
    """
    makes plot of the gaussian process output
    Parameters:
    -----------

    testSetY: test value set
    predictedSetY: predicted value set
    title: string with name of plot
    min: minimum value of the Y-axis
    max: maximum value of the Y-axis

    Returns:
    --------
    a plot of the predicted values and the observed values
    """


    fig = plt.figure(figsize=(20, 6))

    plt.ylabel('Consumptie (kWh)', fontsize=13)
    plt.title('SVR: '+str(title))
    plt.legend(loc='upper right')
    plt.xlim([0, len(testSetY)])
    plt.ylim([min, max])


    xTickLabels = np.arange(0,len(predictedSetY),1)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, len(predictedSetY), 1))
    ax.set_xticklabels(labels=xTickLabels, fontsize=9, rotation=90)

    plt.plot(predictedSetY, 'b-', label=u'Voorspelling',color='green')
    plt.plot(testSetY, 'r.', markersize=10, label=u'Observaties',color='red')
    x = range(len(testSetY))
    plt.show(block=True)





