import numpy as np
import pandas as pd
import features.featureMaker as fm
import matplotlib.pyplot as plt
import Methods.GaussianprocessEngine as GPEngine
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import Methods.GaussianprocessEngine2 as GPEnginepyGP
from sklearn import preprocessing
from Results import clusterResults as CR

from Methods.crossvalidation import KFoldWithGap
from sklearn.grid_search import GridSearchCV



pd.options.display.mpl_style = 'default'

__author__ = 'christiaanleysen'


#['squared_exponential', 'generalized_exponential', 'absolute_exponential', 'linear', 'cubic']



def makefeatureVectors(dfConsumption,dfTemperature,dfSolar,beginDateTraining,EndDateTraining,beginDateTest,EndDateTest,sampleRate,selectedHouseHold,dataFrameTrainingName,dataFrameTestName):
    """
    makes featurevectors with form [Y,x1,...,xn] with Y is the value and x1,..,xn are the features
    Parameters:
    -----------

    dfConsumption: dataframe of the consumption
    dfTemperature: dataframe of the temperature
    dfSolar: dataframe of the solar info
    beginDatetraining: string repr. of the begin date of training set
    endDateTraining: string repr. of the end date of training set
    beginDateTest: string repr. of the begin date of test set
    EndDateTest: string repr. of the end date of test set
    sampleRate: string repr. of sample rate of data e.g. H = 1 hour, D = 1Day
    selectedHousehold: selected household
    dataFrameTRainingName: save name for the training dataframe
    dataFrameTestName: save name for the test dataframe

    Returns:
    --------
    training and test featurevector of the selected household
    """

    #MAKE FRESH featureVector for trainingset and testset
    #BETER PREDICTION WITHOUT THE YEAR INDEX INCLUDED
    augmentedFeaturevectorTraining = fm.makeFeatureVectorAllSelectedHousehold(dfConsumption,dfTemperature,dfSolar,beginDateTraining, EndDateTraining,sampleRate,selectedHouseHold,dataFrameTrainingName)
    augmentedFeaturevectorTest = fm.makeFeatureVectorAllSelectedHousehold(dfConsumption,dfTemperature,dfSolar,beginDateTest, EndDateTest,sampleRate,selectedHouseHold,dataFrameTestName)
    #if there's a error when changing the samplerate, check the number of selected days
    return augmentedFeaturevectorTraining, augmentedFeaturevectorTest

def loadFeatureVectors(dataframeGPTrainingName,dataframeGPTestName):
    """
   load the featurevectors
    Parameters:
    -----------
    dataFrameTRainingName: save name for the training dataframe
    dataFrameTestName: save name for the test dataframe

    Returns:
    --------
    training and test featurevector of the selected names
    """
    #LOAD PREVIOUS CALCULATED featurevector for trainingset and testset (SPEEDUP)
    augmentedFeaturevectorTraining = fm.loadFeatureVector(dataframeGPTrainingName)
    augmentedFeaturevectorTest = fm.loadFeatureVector(dataframeGPTestName)
    return augmentedFeaturevectorTraining, augmentedFeaturevectorTest

def CalculateOptimalHyperParameters(augmentedFeaturevectorTraining,kernelfunction,thetaRange,nuggetRange,nfold):
    """
    Find the optimal hyperparameters using crossvalidation

    Parameters:
    -----------
    augmentedFeaturevectorTraining: training featurevector
    kernelfunction: string repr.of kernel function e.g. 'linear'
    thetarange: range of hyperparam to be tested
    nuggetrange: range of  nugget to be tested
    nfold: number of folds crossvalidation


    Returns:
    --------
    graph of best theta and nugget

    """

    #split test and training set into target en predictors
    trainSetX = augmentedFeaturevectorTraining.values[:, 1:10]
    trainSetY = augmentedFeaturevectorTraining.values[:, 0]


    #find optimal hyperparameters for the gaussian process regression
    theta = thetaRange
    nfold = nfold
    nugget = nuggetRange
    #GPEngine.findOptimalHyperParameters(theta, nugget, nfold, trainSetX, trainSetY,kernelfunction)
    GPEngine.findOptimalHyperParametersfixesNugget(theta, nugget, nfold, trainSetX, trainSetY,kernelfunction)

    #cross validation whole dataset?
    #crossValidationSet = [trainSetX,testSetX]
    #crossValidationSet2 = [trainSetY,testSetY]
    #trainSetX, testSetX, trainSetY, testSetY = cross_validation.train_test_split(trainSetX, trainSetY, test_size=0.3, random_state=0,kernelfunction)
    return -10000
#TODO return statement aanpassen plotten moet veranderen in optimale theta en nugget teruggeven (optimize van GP werkt niet of slecht?)

def predict(augmentedFeaturevectorTraining,augmentedFeaturevectorTest,theta,nugget,kernelfunction):
    """
    predicts the consumption with sklearn toolkit

    Parameters:
    -----------
    augmentedFeaturevectorTraining: trainingset
    augmentedFeaturevectorTest: testset
    theta: hyperparam to be used
    nugget: nugget to be used
    kernelfunction: kernelfunction to be used

    Returns:
    --------
    a prediction of the model of the consumption, mean squared error,variance, predictionvalues, R2 TrainingScore, R2 TestScore
    """
    trainSetX = augmentedFeaturevectorTraining.values[:, 1:20]
    trainSetY = augmentedFeaturevectorTraining.values[:, 0]
    testSetX = augmentedFeaturevectorTest.values[:, 1:20]
    testSetY = augmentedFeaturevectorTest.values[:, 0]
    #actual Gaussian process regression

    gpEEC,sigma, EECprediction, R2TrainingScore, R2TestScore,MAE,MRE = GPEngine.predictConsumption(trainSetX, trainSetY,
                                                              testSetX, testSetY,theta, nugget,kernelfunction)



    #HH1: 0.05,0.1
    #HH4: 0.0009, 0.01
    #HH4: 0.001, 0.001
    #HH4: 0.0024, 0.001 (*)\
    #HH4: 0.0018, 0.001
    #HH4: 0.0009, 0.001 (*)

    MAE = mean_absolute_error(testSetY,EECprediction)
    MSE = mean_squared_error(testSetY,EECprediction)
    MAPE = np.mean(np.abs((testSetY - EECprediction) / (np.mean(testSetY)+0.001))) * 100 #0.001 as protection for division by zero
    if np.mean(np.mean(testSetY)) == 0:
        MRE = 50
    else:
        MRE = (MAE/(np.mean(testSetY)))*100


    return  EECprediction, sigma,MSE,MAPE,MAE,MRE,R2TrainingScore, R2TestScore


def predictpyGP(augmentedFeaturevectorTraining,augmentedFeaturevectorTest,scaled=False):
    """
    predicts the consumption with pyGP toolkit.

    Parameters:
    -----------
    augmentedFeaturevectorTraining: training set
    augmentedFeaturevectorTest: test set
    scaled: boolean wether the features must be scaled



    Returns:
    --------
    a prediction of the consumption, variance and rmse value
    """
    trainSetX = augmentedFeaturevectorTraining.values[:, 1:20]
    trainSetY = augmentedFeaturevectorTraining.values[:, 0]
    testSetX = augmentedFeaturevectorTest.values[:, 1:20]
    testSetY = augmentedFeaturevectorTest.values[:, 0]

    if scaled:
        trainSetX = np.asarray([preprocessing.scale(element)for element in trainSetX])
        #trainSetY =preprocessing.scale(trainSetY,axis=0)

        testSetX = np.asarray([preprocessing.scale(element )for element in testSetX])
        #testSetY =preprocessing.scale(testSetY,axis=0)


    #Find most similar household (e.g. min 0.99 similarity), make general model, test if prediction with general model is more accurate.
    #first cluster all household, take households from same cluster, use general model for the prediction.
    # Or general model of the general model and the observartion.

    #clusterResults = CR.clusterResultsMultipleHHDates()
    for cluster in clusterResults:
      (take most similar element and )
    predictedSetY,sigma,MAE,MRE = GPEnginepyGP.predictConsumption(trainSetX,trainSetY,testSetX,testSetY)


    return predictedSetY,sigma,MAE,MRE





def predictpyGPSparse(augmentedFeaturevectorTraining,augmentedFeaturevectorTest):
    """
    predicts the consumption with pyGP toolkit using sparse algorithm.

    Parameters:
    -----------
    augmentedFeaturevectorTraining: training set
    augmentedFeaturevectorTest: test set
    scaled: boolean wether the features must be scaled



    Returns:
    --------
    a prediction of the consumption, variance and rmse value
    """
    trainSetX = augmentedFeaturevectorTraining.values[:, 1:20]
    trainSetY = augmentedFeaturevectorTraining.values[:, 0]
    testSetX = augmentedFeaturevectorTest.values[:, 1:20]
    testSetY = augmentedFeaturevectorTest.values[:, 0]

    predictedSetY,sigma,rmse = GPEnginepyGP.predictConsumptionSparse(trainSetX,trainSetY,testSetX,testSetY)


    return predictedSetY,sigma,rmse

def plotGPResultpyGP(augmentedFeaturevectorTest,EECprediction,sigma):
    """
    makes plot of the gaussian process output
    Parameters:
    -----------

    augmentedFeaturevectorTest: testset
    EECprediction: predicted value set
    sigma: variance of the prediction


    Returns:
    --------
    a plot of the predicted values and their variance and the observed values
    """

    testSetY = augmentedFeaturevectorTest.values[:, 0]
    #convert the prediction array in a dataframe for easy plotting
    EECPredicted = augmentedFeaturevectorTest.copy()
    EECPredicted['predictedSetY'] = EECprediction
    predictedY = EECPredicted['predictedSetY']


    #Plot the observation vs the prediction
    GPEngine.plotObsVsPred(testSetY,EECprediction)

    #Plot the GP prediction results
    GPEnginepyGP.plotGaussianProcess(testSetY, predictedY, sigma,'Dagelijkse elektrische consumptie',(np.amin(testSetY)-1000),(np.amax(testSetY)+1000))
    plt.show(block=True)


def plotGPResult(augmentedFeaturevectorTest,EECprediction,sigma):
    """
    makes plot of the gaussian process output
    Parameters:
    -----------

    augmentedFeaturevectorTest: testset
    EECprediction: predicted value set
    sigma: variance of the prediction


    Returns:
    --------
    a plot of the predicted values and their variance and the observed values
    """

    testSetY = augmentedFeaturevectorTest.values[:, 0]
    #convert the prediction array in a dataframe for easy plotting
    EECPredicted = augmentedFeaturevectorTest.copy()
    EECPredicted['predictedSetY'] = EECprediction
    predictedY = EECPredicted['predictedSetY']


    #Plot the observation vs the prediction
    GPEngine.plotObsVsPred(testSetY,EECprediction)

    #Plot the GP prediction results
    GPEngine.plotGaussianProcess(testSetY, predictedY, sigma,'Dagelijkse elektrische consumptie',(np.amin(testSetY)-100000),(np.amax(testSetY)+100000))
    plt.show(block=True)



'''
if __name__ == "__main__":
    makefeatureVector('2013-09-01 00:00:00','2014-07-31 23:45:00','2014-08-14 00:00:00','2014-08-28 23:45:00','D','HH4','Measurements_Elec_other_HP_PV','trainingHH','testHH')
    featureTrainingSet, featureTestSet = loadFeatureVectors('trainingHH','testHH')
    trainSetX = featureTrainingSet.values[:, 1:10]
    trainSetY = featureTrainingSet.values[:, 0]
    theta = np.arange(0.0001,0.001,0.0001)
    result = GPEngine.findOptimalHyperParametersfixesNugget(theta,0.01,10,trainSetX,trainSetY,'linear')
    print(result)
'''