__author__ = 'christiaanleysen'

import features.featureMaker as fm
import pandas as pd
import numpy as np

# location of the data
dataPathConsumption = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/'
# dataframe save location
dataframeLocation = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/dataframes/'


def makeClassificationFeatureVector(beginDate, endDate, sampleRate, fileNameConsumption, dataFrameFileName):
    df = fm.readConsumptionDataCSV(dataPathConsumption, fileNameConsumption, beginDate, endDate)

    classificationVector = pd.DataFrame()

    for household in list(df.columns.values):
        print('Making ClassificationFeatureVector...')
        singleHouseHoldFeatureVector = fm.makeFeatureVectorAllSelectedHousehold(beginDate, endDate, sampleRate,
                                                                                fileNameConsumption, household,
                                                                                'classificationFeatureVector')
        singleHouseHoldFeatureVector = singleHouseHoldFeatureVector.rename(
            columns={household: 'HHX'})  # rename the consumptionColumn to HHX so that all the data will be seen
        # as records of the same household because we want to combine all the data
        classificationVector = classificationVector.append(singleHouseHoldFeatureVector)
        print('featureVector ' + str(household) + ' ready')

    print(classificationVector)
    classificationVector.to_pickle(dataframeLocation + dataFrameFileName)
    print('classification feature vector Saved to' + str(dataframeLocation))
    return classificationVector


def updateYearIndex(df):
    df['year'] = df.index.year
    return df


def makeClassificationFeatureVectorShiftedByYear(beginDate, endDate, sampleRate, fileNameConsumption,
                                                 dataFrameFileName):
    df = fm.readConsumptionDataCSV(dataPathConsumption, fileNameConsumption, beginDate, endDate)

    classificationVector = pd.DataFrame()
    counter = 0
    for household in list(df.columns.values)[:4]: #TODO
        print('Making ClassificationFeatureVector...')
        singleHouseHoldFeatureVector = fm.makeFeatureVectorAllSelectedHousehold(beginDate, endDate, sampleRate,
                                                                                fileNameConsumption, household,
                                                                                'classificationFeatureVector')
        singleHouseHoldFeatureVector = singleHouseHoldFeatureVector.rename(
            columns={household: 'HHX'})  # rename the consumptionColumn to HHX so that all the data will be seen
        # as records of the same household because we want to combine all the data
        singleHouseHoldFeatureVector.index = singleHouseHoldFeatureVector.index + np.timedelta64(counter,
                                                                                                 'Y')  # data will be shifted by a year to act as data from different years
        singleHouseHoldFeatureVector = updateYearIndex(singleHouseHoldFeatureVector)
        classificationVector = classificationVector.append(singleHouseHoldFeatureVector)

        print('featureVector ' + str(household) + ' ready')
        counter = counter + 1  # +1 can give problems
        print('counter', counter)

    print(classificationVector)
    classificationVector.to_pickle(dataframeLocation + dataFrameFileName)
    print('classification feature vector Saved to' + str(dataframeLocation))

    return classificationVector


def addUniqueID(df, id):
    df['id'] = pd.Series([id for x in range(len(df.index))], index=df.index)
    return df


def makeClassificationFeatureVectorWithUniqueID(dfConsumption,dfTemperature,dfSolar,beginDate, endDate, sampleRate, dataFrameFileName):

    df=dfConsumption
    classificationVector = pd.DataFrame()
    counter = 10
    for household in list(df.columns.values):
        print('Making ClassificationFeatureVector...')
        singleHouseHoldFeatureVector = fm.makeFeatureVectorAllSelectedHousehold(dfConsumption,dfTemperature,dfSolar,beginDate, endDate, sampleRate,
                                                                                 household,
                                                                                'classificationFeatureVector')
        # print('singleHHFV:',singleHouseHoldFeatureVector)
        singleHouseHoldFeatureVector = singleHouseHoldFeatureVector.rename(
            columns={household: 'HHX'})  # rename the consumptionColumn to HHX so that all the data will be seen
        # as records of the same household because we want to combine all the data

        singleHouseHoldFeatureVector = addUniqueID(singleHouseHoldFeatureVector, counter)
        classificationVector = classificationVector.append(singleHouseHoldFeatureVector)

        print('featureVector ' + str(household) + ' ready')
        counter = counter + 1
        print('counter', counter)


    # print(classificationVector)
    classificationVector.to_pickle(dataframeLocation + dataFrameFileName)
    print('classification feature vector '+dataFrameFileName+' Saved to' + str(dataframeLocation))

    return classificationVector


def makeSingleClassificationFeatureVectorWithUniqueID(dfConsumption,dfTemperature,dfSolar,beginDate, endDate, sampleRate,
                                                      houseHold,dataFrameFileName):
    print('Making ClassificationFeatureVector...')
    singleHouseHoldFeatureVector = fm.makeFeatureVectorAllSelectedHousehold(dfConsumption,dfTemperature,dfSolar,beginDate, endDate, sampleRate,
                                                                             houseHold,
                                                                            dataFrameFileName)
    singleHouseHoldFeatureVector = addUniqueID(singleHouseHoldFeatureVector, 0.000005) #0.000005
    print('featureVector ' + str(houseHold) + ' ready')

    singleHouseHoldFeatureVector.to_pickle(dataframeLocation + dataFrameFileName)
    print('Classification Feature vector '+str(dataFrameFileName) +' saved to'+str(dataframeLocation))
    return singleHouseHoldFeatureVector


def makeClassificationFeatureVectorTweakedIrradiation(beginDate, endDate, sampleRate, fileNameConsumption,
                                                      dataFrameFileName):
    return 0


def loadClassificationFeatureVector(DataFrameFileName):
    print('load classification feature vector '+str(DataFrameFileName)+' ...')
    return pd.read_pickle(dataframeLocation + DataFrameFileName)


def test(beginDate, endDate, fileNameConsumption):
    df = fm.readConsumptionDataCSV(dataPathConsumption, fileNameConsumption, beginDate, endDate)
    print('df', df)
    df.index = df.index + np.timedelta64(0, 'Y')
    print('dfY', df)


#if __name__ == "__main__":
    #print(pd.__version__)
    # makeClassificationFeatureVectorShiftedByYear('2013-09-01 00:00:00', '2014-08-31 23:45:00',"D",'Measurements_Elec_other_HP_PV','classificationfeatureVector')
    #makeClassificationFeatureVectorWithUniqueID('2013-09-01 00:00:00', '2014-08-31 23:45:00', "D",
     #                                           'Measurements_Elec_other_HP_PV', 'classificationfeatureVector')

    # test('2013-09-01 00:00:00', '2014-08-31 23:45:00','Measurements_Elec_other_HP_PV')
