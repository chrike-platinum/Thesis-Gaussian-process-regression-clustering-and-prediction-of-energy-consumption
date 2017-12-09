import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import csv
import io
pd.options.display.mpl_style = 'default'
from sklearn import preprocessing
import pandas


__author__ = 'christiaanleysen'
#parameters input for plotting
dataPath = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/'
fileName = 'Data_SolarIrradiation'
sampleRate = 'D'
beginDate = '2013-09-01 00:00:00'
endDate = '2014-08-31 23:45:00'
combinationFunction = 'sum'
directions = ['GHI [Wh/m^2]']

'''
['GHI [Wh/m^2]','POA [Wh/m^2]','POA [Wh/m^2] 90/0','POA [Wh/m^2] 90/90','POA [Wh/m^2]90/180','POA [Wh/m^2] 90/270' ]
'''


#Read the input Excel file for given period
def readSolarDataExcel(dataPath,fileName,beginDate,endDate):
    xl = pd.ExcelFile(dataPath+fileName+'.xlsx')
    df = xl.parse(sheetname=0, header=0)
    #df_normalized = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    #df = df/df.max().astype(np.float64) #standarization

    #df = (df - df.mean())/df.std(ddof=0) #zscore normalize
    return df[beginDate:endDate]

#Resample the data with the given sampleRate above
def resampleSolarData(df,sampleRate,combinationFunction):
    dfResampled = df.resample(sampleRate, how=combinationFunction)
    #print(dfResampled)
    return dfResampled

def filterDirection(dfResampled,directions):
    dfResampledDirection = dfResampled[directions]
    return dfResampledDirection





#plot the data
def main():
    plt.show(block=True)
    df = readSolarDataExcel(dataPath,fileName,beginDate,endDate)
    dfResampled = resampleSolarData(df,sampleRate,combinationFunction)
    dfResampledDirection = filterDirection(dfResampled,directions)
    dfResampledDirection.interpolate(method='time').plot()
    plt.xlabel('Tijd')
    plt.ylabel('Hoeveelheid zonlicht')
    #dfResampledDirection.plot()
    plt.title("Solar Data")
    plt.show(block=True)

if __name__ == "__main__":
    main()
