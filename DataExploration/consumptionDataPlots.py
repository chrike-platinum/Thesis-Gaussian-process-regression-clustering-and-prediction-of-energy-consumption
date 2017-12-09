import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import csv
import io
from pandas.tools.plotting import autocorrelation_plot
from datetime import timedelta

#Nicer plots
pd.options.display.mpl_style = 'default'

__author__ = 'christiaanleysen'
#parameters input for plotting
dataPath = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/'
fileName = 'Measurements_Elec_other_HP'
#showHH = np.arange(0)
sampleRate = 'H'
beginDate = '2013-09-01 00:00:00'
endDate = '2013-09-03 23:45:00'
combinationFunction = 'sum'
selectedHouseHolds = ['HH0','HH10','HH12']

'''
['HH0','HH1','HH2','HH3','HH4','HH5','HH6','HH7','HH8','HH9','HH10','HH11','HH12','HH13','HH14','HH15','HH16','HH17','HH18','HH19',
                      'HH20','HH21','HH22','HH23','HH24','HH25','HH26','HH27','HH28','HH29','HH30',
                      'HH31','HH32','HH33','HH34','HH35']

['HH0','HH1','HH2','HH3','HH4','HH5','HH6','HH7','HH8','HH9','HH10','HH11','HH12','HH13','HH14','HH15','HH16','HH17','HH18','HH19',
                      'HH20','HH21','HH22','HH23','HH24','HH25','HH26','HH27','HH28','HH29','HH30',
                      'HH31','HH32','HH33','HH34','HH35','HH36','HH37','HH38','HH39','HH40',
                      'HH41','HH42','HH43','HH44','HH45','HH46','HH47','HH48','HH49','HH50',
                      'HH51','HH52','HH53','HH54','HH55','HH56','HH57','HH58','HH59','HH60',
                      'HH61','HH62','HH63','HH64','HH65','HH66','HH67','HH68','HH69','HH70']
'''
#Read the input CSV file for given period
def readConsumptionDataCSV(dataPath,fileName,beginDate,endDate):
    df = pd.read_csv(dataPath+fileName+'.csv', sep=',', encoding='latin1',
    parse_dates=True, index_col=0)
    if 'dummy' in df:
        del df['dummy'] #delete the dummy column
    return df[beginDate:endDate]


#Resample the data with the given sampleRate above
def resampleConsumptionData(df,sampleRate,combinationFunction):
    dfResampled = df.resample(sampleRate, how=combinationFunction)
    #print(dfResampled)
    return dfResampled

#dfResampled.drop(dfResampled.columns[showHH], axis=1, inplace=True)

#Select houseHold that are included in the dataFrame
def filterhouseHolds(dfResampled,selectedHouseHolds):
    dfResampledSelection = dfResampled[selectedHouseHolds]
    return dfResampledSelection

def feature_standardization(dataset):
    """
    Returns a stardazied dataset with each column having zero mean and unit variance.
    mean_scaled and std_scaled is needed information for scaling back to normal scale
    """
    means_scaled = dataset.mean()
    std_scaled = dataset.std()
    dataset = (dataset-means_scaled)/std_scaled
    return dataset, means_scaled, std_scaled


#plot the data
def main():
    df = readConsumptionDataCSV(dataPath,fileName,beginDate,endDate)
    dfna = df.dropna()
    dfResampled  = resampleConsumptionData(dfna,sampleRate,combinationFunction)
    dfResampledSelection = filterhouseHolds(dfResampled,selectedHouseHolds)
    fig = dfResampledSelection.plot()
    plt.xlabel('Tijd (dagen)')
    plt.ylabel('Consumptie [W]')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,50000))#1200000
    print(np.mean(dfResampledSelection))
    fig.legend(loc=9,prop={'size':6},ncol=3)
    plt.show(block=True)


    #autocorrelation_plot(dfResampledSelection)
    #print(dfResampledSelection)
    #print('auto-correlatie: ',autocorr_days(dfResampled,'HH1',14))
    plt.xlabel('Tijdsvertraging (dagen)')
    plt.ylabel('Auto-correlatie')
    plt.plot(autocorr_days(dfResampled,'HH10',35))
    plt.plot(autocorr_days(dfResampled,'HH12',35))
    plt.plot(autocorr_days(dfResampled,'HH17',35))

    plt.show(block=True)


def get_recency_values(data, delay_days=0, delay_hrs=0):
    """
    Function for creating a column of recency values with given delay.
    In cases when no data is found for given delay, the current values are returned.

    """
    return [data[timestamp]
           for timestamp in
               [
                    timestamp - timedelta(days=delay_days, seconds=delay_hrs*60*60)
                if timestamp - timedelta(days=delay_days, seconds=delay_hrs*60*60) in data.index
                else
                    timestamp
                for timestamp in data.index
               ]
           ]

def autocorr_days(dataset, column, days, verbose=False):
    """
    Computes and returns autocorrelation values for given number of days.

    """
    data = pd.DataFrame(dataset[column])
    for i in range(1, days+1):
        data['rec_day_'+str(i)] = get_recency_values(dataset[column], delay_days=i)

    autocorr = data.corr()[column]

    if (verbose):
        print(autocorr)

    return autocorr


def autocorr_weeks(dataset, column, weeks, verbose=False):
    """
    Computes and returns autocorrelation values for given number of weeks.

    """
    data = pd.DataFrame(dataset[column])
    for i in range(1, weeks+1):
        data[i] = get_recency_values(dataset[column], delay_days=i*7)

    autocorr = data.corr()[column]

    if (verbose):
        print(autocorr)

    return autocorr




if __name__ == "__main__":
    main()


