__author__ = 'christiaanleysen'
import features.featureMaker as fm
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
'''
This file is used to calculate the linear regression
'''


def predictConsumption(trainSetX, trainSetY, testSetX, testSetY,tune_params=True,scaled=False):
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

    if scaled:
        trainSetX = np.asarray([preprocessing.scale(element)for element in trainSetX])
        #trainSetY =preprocessing.scale(trainSetY,axis=0)

        testSetX = np.asarray([preprocessing.scale(element )for element in testSetX])
        #testSetY =preprocessing.scale(testSetY,axis=0)

    OLS = LinearRegression()
    OLS.fit(trainSetX,trainSetY)# fit default model (mean zero & rbf kernel) with data

    predictedSetY = OLS.predict(testSetX)
    MAE = mean_absolute_error(testSetY,predictedSetY)
    if np.mean(np.mean(testSetY)) == 0:
        MRE = 50
    else:
        MRE = (MAE/(np.mean(testSetY)))*100

    return  predictedSetY,testSetY,MAE,MRE
