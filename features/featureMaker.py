__author__ = 'christiaanleysen'

import pandas as pd
import numpy as np



#parameters

#Householdconsumption paramters
dataPathConsumption = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/'
#fileNameConsumption = 'Measurements_Elec_other'+'.csv' given in method parameter
selectedHouseHolds = ['HH4']

#Temperature parameters
dataPathTemperature = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/'
fileNameTemperature = 'Temperature_ambient' #the words "date" and "temp" are added in head of rows
combinationFunctionTemp = 'mean'
roundNumberTemperature = 5

#Solar parameters
dataPathSolar = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/'
fileNameSolar = 'Data_SolarIrradiation'
directions = ['GHI [Wh/m^2]']#,'POA [Wh/m^2] 90/270','POA [Wh/m^2] 90/90','POA [Wh/m^2]90/180']
roundNumberSolar = 5

#General parameters
beginDate = '2013-09-01 00:00:00'
endDate = '2014-08-31 23:45:00'
combinationFunction = 'sum'
startDayTempIrr = '2013-09-01 00:00:00'

#dataframe save location
dataframeLocation = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/dataframes/'

#round temperature

#Read the input CSV file for given period
def readConsumptionDataCSV(dataPath,fileName,beginDate,endDate):
    df = pd.read_csv(dataPath+fileName+'.csv', sep=',', encoding='latin1',
    parse_dates=True, index_col=0)
    if 'dummy' in df:
        del df['dummy'] #delete the dummy column
    return df[beginDate:endDate]

#Read the input CSV file for given period
def readTemperatureDataCSV(dataPath,fileName,beginDate,endDate):
    df = pd.read_csv(dataPath+fileName+'.csv', sep=',', encoding='latin1',
    parse_dates=True, index_col=0)
    return df[beginDate:endDate]

#Read the input Excel file for given period
def readSolarDataExcel(dataPath,fileName,beginDate,endDate):
    xl = pd.ExcelFile(dataPath+fileName+'.xlsx')
    df = xl.parse(sheetname=0, header=0)
    return df[beginDate:endDate]


#Resample the data with the given sampleRate above
def resampleData(df,sampleRate,combinationFunction):
    dfResampled = df.resample(sampleRate, how=combinationFunction,fill_method='bfill')
    return dfResampled


def filterDirection(dfResampled,directions):
    dfResampledDirection = dfResampled[directions]
    return dfResampledDirection




#dfResampled.drop(dfResampled.columns[showHH], axis=1, inplace=True)

#Select houseHold that are included in the dataFrame
def filterhouseHolds(dfResampled,selectedHouseHolds):
    dfResampledSelection = dfResampled[selectedHouseHolds]
    return dfResampledSelection


#standardization ofthe features
def feature_standardization(dataset):
    """
    Returns a stardazied dataset with each column having zero mean and unit variance.
    mean_scaled and std_scaled is needed information for scaling back to normal scale
    """
    means_scaled = dataset.mean()
    std_scaled = dataset.std()
    dataset = (dataset-means_scaled)/std_scaled
    return dataset, means_scaled, std_scaled



def makeFeatureVectorForHouseHold(dfConsumption,dfTemperature,dfSolar,beginDate,endDate,sampleRate,selectedHouseHolds):
    shiftNr=2

    sampleRateString = str(sampleRate)
    if ("D" in sampleRateString):
        shiftNr = shiftNr
    elif ("H" in sampleRateString):
        shiftNr = shiftNr*24
    elif ("Min" in sampleRateString):
        shiftNr = shiftNr#*(24*4) #quater of hour
    else:
        print('STEPHANIE-ERROR: '
              'makeFeatureVector function in featureMaker FAILED')
    #Read and resample all the data

    #ConsumtionData
    dfConsumptionNormal1 = dfConsumption[beginDate:endDate].dropna()
    #dfConsumption = readConsumptionDataCSV(dataPathConsumption,str(fileNameConsumption),beginDate,endDate).dropna()
    dfConsumptionNormal = resampleData(dfConsumptionNormal1,sampleRate,combinationFunction)
    dfConsumptionNormal = filterhouseHolds(dfConsumptionNormal,selectedHouseHolds)
    #print(dfConsumption.head())

    #RecencyConsumptionData
    dfConsumptionLagged = dfConsumption.dropna()
    dfConsumptionLagged.rename(columns={selectedHouseHolds: 'HHRecency'}, inplace=True)
    dfConsumptionLagged = filterhouseHolds(dfConsumptionLagged,'HHRecency')
    dfConsumptionLagged = resampleData(dfConsumptionLagged,sampleRate,combinationFunction)
    dfConsumptionLagged = dfConsumptionLagged.shift(shiftNr)
    dfConsumptionLagged=dfConsumptionLagged.round(roundNumberTemperature)
    dfConsumptionLagged = dfConsumptionLagged[beginDate:endDate]

    #TemperatureData
    dfTemperature = dfTemperature[startDayTempIrr:endDate].dropna()
    #dfTemperature = readTemperatureDataCSV(dataPathTemperature,fileNameTemperature,startDayTempIrr,endDate).dropna()
    dfTemperature = resampleData(dfTemperature,sampleRate,combinationFunctionTemp)
    dfTemperatureShifted = dfTemperature.shift(shiftNr)
    dfTemperatureShifted=dfTemperatureShifted.round(roundNumberTemperature)
    dfTemperature = dfTemperatureShifted[beginDate:endDate]
    #print(dfTemperature)

    #SolarData
    dfSolar = dfSolar[startDayTempIrr:endDate].dropna()
    #dfSolar = readSolarDataExcel(dataPathSolar,fileNameSolar,startDayTempIrr,endDate).dropna()
    dfSolar = resampleData(dfSolar,sampleRate,combinationFunction)
    dfSolar = filterDirection(dfSolar,directions)
    dfSolarShifted = dfSolar.shift(shiftNr)
    dfSolarShifted=dfSolarShifted.round(roundNumberSolar)
    dfSolar = dfSolarShifted[beginDate:endDate]




    #make the feature vector itself
    #dfFeatures = pd.concat([dfConsumptionNormal,dfTemperature,dfSolar],axis=1) WITHOUT THE LAGGED VARIABLE

    dfFeatures = pd.concat([dfConsumptionNormal,dfConsumptionLagged,dfTemperature,dfSolar],axis=1)


    #drop the rows which contains NaN value because of the shifted temp and irradiation data
    dfFeatures = dfFeatures[np.isfinite(dfFeatures['Temp'])]
    return dfFeatures


def makeClusterFeatureVectorForHouseHold(dfConsumption,dfTemperature,dfSolar,beginDate,endDate,sampleRate,selectedHouseHolds):
    shiftNr=2

    sampleRateString = str(sampleRate)
    if ("D" in sampleRateString):
        shiftNr = shiftNr
    elif ("H" in sampleRateString):
        shiftNr = shiftNr*24
    elif ("Min" in sampleRateString):
        shiftNr = shiftNr#*(24*4) #quater of hour
    else:
        print('STEPHANIE-ERROR: '
              'makeFeatureVector function in featureMaker FAILED')
    #Read and resample all the data

    #ConsumtionData
    dfConsumptionNormal1 = dfConsumption[beginDate:endDate].dropna()
    #dfConsumption = readConsumptionDataCSV(dataPathConsumption,str(fileNameConsumption),beginDate,endDate).dropna()
    dfConsumptionNormal = resampleData(dfConsumptionNormal1,sampleRate,combinationFunction)
    dfConsumptionNormal = filterhouseHolds(dfConsumptionNormal,selectedHouseHolds)



    #make the feature vector itself
    dfFeatures = pd.concat([dfConsumptionNormal],axis=1)


    return dfFeatures




def addTimeFeatures(df,sampleRate):
    '''
    Add the time features (weekday,day of the year and weeknumber) to the featureset
    '''
    sampleRateString = str(sampleRate)
    if ("D" in sampleRateString):
        #df['year'] = df.index.year
        df['dayOfYear'] = df.index.dayofyear
        df['dayOfWeek'] = df.index.weekday
        df['weekNr'] = df.index.weekofyear
    elif ("H" in sampleRateString):
        #df['year'] = df.index.year
        df['dayOfYear'] = df.index.dayofyear
        df['dayOfWeek'] = df.index.weekday
        df['weekNr'] = df.index.weekofyear
        df['hour'] = df.index.hour
    elif ("Min" in sampleRateString):
        #df['year'] = df.index.year
        df['dayOfYear'] = df.index.dayofyear
        df['dayOfWeek'] = df.index.weekday
        df['weekNr'] = df.index.weekofyear
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute

    else:
        print('STEPHANIE-ERROR: '
              'addTimeFeatures function in featureMaker FAILED')
    return df


def addClusterTimeFeatures(df,sampleRate):
    '''
    Add the time features (weekday,day of the year and weeknumber) to the featureset
    '''
    sampleRateString = str(sampleRate)
    if ("D" in sampleRateString):
        #df['year'] = df.index.year
        #df['dayOfYear'] = df.index.dayofyear
        df['dayOfWeek'] = df.index.weekday
        #df['weekNr'] = df.index.weekofyear
    elif ("H" in sampleRateString):
        #df['year'] = df.index.year
        #df['dayOfYear'] = df.index.dayofyear
        df['dayOfWeek'] = df.index.weekday
        #df['weekNr'] = df.index.weekofyear
        df['hour'] = df.index.hour
    elif ("Min" in sampleRateString):
        #df['year'] = df.index.year
        #df['dayOfYear'] = df.index.dayofyear
        df['dayOfWeek'] = df.index.weekday
        #df['weekNr'] = df.index.weekofyear
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute

    else:
        print('STEPHANIE-ERROR: '
              'addTimeFeatures function in featureMaker FAILED')
    return df



def addRecencyfeature(df,nrOfDays):
    '''
    Add recency features to the featureset (for testpurposes only)
    '''
    dfRecency = df.shift(nrOfDays)
    dfRecencyColumn = dfRecency[dfRecency.columns[0]]
    dfExtended = pd.concat([df,dfRecencyColumn],axis=1)
    #still need to drop first nrOfDaysColumns
    return dfExtended.tail(len(dfExtended)-nrOfDays)

def addHolidays(dataset):
    """
    Marks holidays in given dataset as sundays (weekday=6)

    """
    holidays = [


        #'2013-09-02', #test
        '2013-11-01',
        '2013-11-11',
        '2013-12-25',


        '2014-01-01',
        '2014-04-21',
        '2014-05-01',
        '2014-05-29',
        '2014-06-09',
        '2014-07-21',
        '2014-08-15',
        '2014-11-01',
        '2014-11-11',
        '2014-12-25',

    ]
    for h in holidays:
        dataset.ix[h:h, 'dayOfWeek'] = 6
    return dataset

def detect_outliers(dataset, column_name, allow_negatives=False, verbose=False):
    """
    Detects values further away from mean than 3 standard deviations as outlier
    and marks them as NaN.

    Parameters
    ----------
    allow_negatives: if false, negative values are considered as outliers

    """
    column = dataset[column_name]
    if (allow_negatives):
        mean = column.mean()
        std = column.std()
        lowerBound = mean-3*std
        upperBound = mean+3*std
    else:
        mean = column[column>0].mean()
        std = column[column>0].std()
        lowerBound = max(0, mean-3*std)
        upperBound = mean+3*std

    if (verbose):
        print("========== VARIABLE: {} ==========".format(column_name))
        print("Mean: {0}, StdDev: {1}, Lower: {2}, Upper: {3}".format(mean, std, lowerBound, upperBound))

    count = 0
    for idx,val in column.iteritems():
        if (val < lowerBound or val > upperBound):
            print('------------------------NAN-------------------------------------')
            dataset.loc[idx,column_name] = np.nan
            count = count+1

def makeFeatureVectorAll(dfConsumption,dfTemperature,dfSolar,beginDate,endDate,sampleRate,DataframeFileName):
    featureVector = makeFeatureVector(dfConsumption,dfTemperature,dfSolar,beginDate,endDate,sampleRate)
    #detect_outliers(featureVector, 'HH4', allow_negatives=True, verbose=True)
    #featureVector, mean, std = feature_standardization(featureVector)
    featureVector = addTimeFeatures(featureVector,sampleRate)
    featureVector = addHolidays(featureVector)
    featureVector.to_pickle(dataframeLocation+DataframeFileName)
    print('Feature vector '+str(DataframeFileName) +' saved to'+str(dataframeLocation))
    return featureVector

def loadFeatureVector(DataFrameFileName):
    print('load feature vector '+str(DataFrameFileName)+' ...')
    return pd.read_pickle(dataframeLocation+DataFrameFileName)

def makeFeatureVectorAllSelectedHousehold(dfConsumption,dfTemperature,dfSolar,beginDate,endDate,sampleRate,selectedHouseHold,DataframeFileName):
    featureVector = makeFeatureVectorForHouseHold(dfConsumption,dfTemperature,dfSolar,beginDate,endDate,sampleRate,selectedHouseHold)
    #detect_outliers(featureVector, 'HH4', allow_negatives=True, verbose=True)
    #featureVector, mean, std = feature_standardization(featureVector)

    featureVector = addTimeFeatures(featureVector,sampleRate)
    featureVector = addHolidays(featureVector)
    featureVector.to_pickle(dataframeLocation+DataframeFileName)
    print('Feature vector '+str(DataframeFileName) +' saved to:'+str(dataframeLocation))
    return featureVector

def makeClusterFeatureVectorAllSelectedHousehold(dfConsumption,dfTemperature,dfSolar,beginDate,endDate,sampleRate,selectedHouseHold,DataframeFileName):
    featureVector = makeClusterFeatureVectorForHouseHold(dfConsumption,dfTemperature,dfSolar,beginDate,endDate,sampleRate,selectedHouseHold)
    #detect_outliers(featureVector, 'HH4', allow_negatives=True, verbose=True)
    #featureVector, mean, std = feature_standardization(featureVector)



    featureVector = addClusterTimeFeatures(featureVector,sampleRate)
    featureVector = addHolidays(featureVector)
    featureVector.to_pickle(dataframeLocation+DataframeFileName)
    print('Feature vector '+str(DataframeFileName) +' saved to:'+str(dataframeLocation))
    return featureVector

if __name__ == "__main__":
    makeFeatureVector()

'''
1) makeFeatureVector: read data and drop NaN values
2) feature_standardization: possible feature standardization
3) addTimeFeatures: add weekday, day of the year and week number to the featureset (possibly also hour of the day, depending on the sampleRate)
4) addHolidays: add specific hollidays to the featureset by setting the week number to 6 (= sunday)
'''
