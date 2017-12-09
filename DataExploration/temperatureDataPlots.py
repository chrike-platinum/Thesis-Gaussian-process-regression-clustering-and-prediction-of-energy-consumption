import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import csv
import io
#Nicer plots
pd.options.display.mpl_style = 'default'

__author__ = 'christiaanleysen'
#parameters input for plotting
dataPath = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/'
fileName = 'Temperature_ambient' #the words "date" and "temp" are added in head of rows
sampleRate = 'D'
beginDate = '2013-09-01 00:00:00'
endDate = '2014-08-31 23:45:00'
combinationFunction = 'mean'


#Read the input CSV file for given period
def readTemperatureDataCSV(dataPath,fileName,beginDate,endDate):
    df = pd.read_csv(dataPath+fileName+'.csv', sep=',', encoding='latin1',
    parse_dates=True, index_col=0)
    return df[beginDate:endDate]


#Resample the data with the given sampleRate above
def resampleTemperatureData(df,sampleRate,combinationFunction):
    dfResampled = df.resample(sampleRate, how=combinationFunction)
    #print(dfResampled)
    return dfResampled

def main():
    df = readTemperatureDataCSV(dataPath,fileName,beginDate,endDate)
    dfResampled  = resampleTemperatureData(df,sampleRate,combinationFunction)
    dfResampled.plot()
    plt.xlabel('Tijd')
    plt.ylabel('Temperatuur (grad. C)')
    plt.show(block=True)


if __name__ == "__main__":
    main()
