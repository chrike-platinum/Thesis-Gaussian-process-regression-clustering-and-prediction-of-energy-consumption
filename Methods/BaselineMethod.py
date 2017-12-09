__author__ = 'christiaanleysen'
import features.featureMaker as fm
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np


def predictConsumption(dfConsumption,beginDateTest,endDateTest,sampleRate,nrOfpredictionDays,selectedHH):
    """
    predicts the consumption by returning the value of the predictedday-nrOfPredictiondays

    Parameters:
    -----------
    dfConsumption: dataframe of the consumption
    beginDateTest: string repr. of the begin date to test
    endDateTest: string repr. of the end date to test
    sampleRate: string repr. of rate of sampling (per hour = 'H'; per day = 'D')
    nrOfPredictiondays: int: prediction window (e.g. '2' = 2 days)
    selectedHH: selected household


    Returns:
    --------
    a prediction of the consumption using the baseline method
    """

    dfConResampled =dfConsumption.resample(sampleRate, how='sum',fill_method='bfill')
    dfConsumptionHH = fm.filterhouseHolds(dfConResampled,[selectedHH])
    shiftNr = nrOfpredictionDays
    sampleRateString = str(sampleRate)
    if ("D" in sampleRateString):
        shiftNr = shiftNr
    elif ("H" in sampleRateString):
        shiftNr = shiftNr*24
    else:
        print('STEPHANIE-ERROR: '
              'makeFeatureVector function in BaselineMethod FAILED')
    print('shiftnummer:',shiftNr)
    testSetY = dfConsumptionHH[beginDateTest:endDateTest]
    testSetYValues = dfConsumptionHH[beginDateTest:endDateTest].values[:, 0]

    dfshiftedCon = dfConsumptionHH.shift(shiftNr)
    predictedSetY = dfshiftedCon[beginDateTest:endDateTest].values[:, 0]
    MAE = mean_absolute_error(testSetYValues, predictedSetY)
    if np.mean(np.mean(testSetY)) == 0:
        MRE = 50
    else:
        MRE = (MAE/np.mean(testSetY).values)*100

    print('baselineethode', MRE)
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
    plt.title('Baseline methode: '+str(title))
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




