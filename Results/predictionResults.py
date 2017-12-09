# coding=utf-8

__author__ = 'christiaanleysen'



import Methods.GaussianProcesses as GP
import numpy as np
import pandas as pd
import time
import datetime
import locale
import Results.plotEngine as plotEngine
import Methods.BaselineMethod as BM
import Methods.SVMPrediction as SVR
import Methods.LinRegPrediction as OLS
locale.setlocale(locale.LC_ALL, 'en_US')
from sklearn import preprocessing




resultsLocation = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/'

inputDataPath = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/'
fileNameIrr = 'Data_SolarIrradiation'
fileNameTemp = 'Temperature_ambient'
theta = 0.0009
nugget = 0.001 #0.0009, 0.001



#Read the input CSV file for given period
def readConsumptionDataCSV(dataPath,fileName):
    df = pd.read_csv(dataPath+fileName+'.csv', sep=',', encoding='latin1',
    parse_dates=True, index_col=0)
    if 'dummy' in df:
        del df['dummy'] #delete the dummy column
    return df

#Read the input CSV file for given period
def readTemperatureDataCSV(dataPath,fileName):
    df = pd.read_csv(dataPath+fileName+'.csv', sep=',', encoding='latin1',
    parse_dates=True, index_col=0)
    return df

#Read the input Excel file for given period
def readSolarDataExcel(dataPath,fileName):
    xl = pd.ExcelFile(dataPath+fileName+'.xlsx')
    df = xl.parse(sheetname=0, header=0)
    return df

dfTemp = readTemperatureDataCSV(inputDataPath,fileNameTemp)
dfIrr = readSolarDataExcel(inputDataPath,fileNameIrr)

#if inside confidence interval return 0 else return 1
def CountOutsideConfidenceInterval(predictedValueList,realValueList,sigmaList):
    count = 0
    for i in range(0,len(predictedValueList)):
        minBound = predictedValueList[i]-1.96*sigmaList[i]
        maxBound = predictedValueList[i]+1.96*sigmaList[i]
        if (not(minBound<=realValueList[i]<=maxBound)):
            count += 1
    return count


def predictionResults(dfConsumption,beginDateTraining,endDateTraining,beginDateTest,endDateTest,nrOfHH,sampleRate,featureVectorsAreCreated,optimizeHyperParameters,kernelfunction,plotBoolean,csvFileName,handleNotHome=True):
    notHomeError=1-0.15 # the error which ia used for when the HH is not home at the time of the prediction .15/(1-.15) = HH error


    startFeatures = time.time()
    NumberOfHouseholds = nrOfHH
    if (featureVectorsAreCreated == False):
        for nr in range(0, NumberOfHouseholds+1):
            GP.makefeatureVectors(dfConsumption,dfTemp,dfIrr,beginDateTraining,endDateTraining,beginDateTest,endDateTest,sampleRate,'HH'+str(nr),str(csvFileName)+'_DATAFRAME_training_HH'+str(nr),str(csvFileName)+'_DATAFRAME_test_HH'+str(nr))
    totalresultsDataFrame = pd.DataFrame()
    endFeatures = time.time()
    elapsedTimeFeatureVector = round((endFeatures - startFeatures),2)

    startPrediction = time.time()
    global theta
    global nugget
    numberOfTestSamples = 0
    for nr in range(0, NumberOfHouseholds+1):
        featureTrainingSet, featureTestSet = GP.loadFeatureVectors(str(csvFileName)+'_DATAFRAME_training_HH'+str(nr),str(csvFileName)+'_DATAFRAME_test_HH'+str(nr))
        numberOfTestSamples = len(featureTestSet)
        #print('lengte:',numberOfTestSamples )
        if optimizeHyperParameters == True:
            thetaRange = np.arange(0.00001, 1, 0.001)
            nuggetRange = np.arange(0.001, 0.01, 0.005)
            nfold = 10
            theta, nugget = GP.CalculateOptimalHyperParameters(featureTrainingSet,kernelfunction,thetaRange,0.001,nfold)
        EECprediction, sigma,MSE,MAPE,MAE,MRE,R2TrainingScore, R2TestScore = GP.predict(featureTrainingSet,featureTestSet,theta, nugget,kernelfunction)
        sumrealValue = sum(featureTestSet.values[:, 0])
        sumpredictedValue =sum(EECprediction)
        meanSigma = np.mean(sigma)
        CountOutOFConf = CountOutsideConfidenceInterval(EECprediction,featureTestSet.values[:, 0],sigma)
        HHResultdataframe = pd.DataFrame( {'HHnr' : [nr], 'Reële som EEC' : [sumrealValue],'som voorspelde EEC' : [sumpredictedValue],'Gemiddelde Sigma' : [meanSigma],
                                            'MSE' : [MSE], 'MAE' : [MAE], 'MAPE' : [MAPE], 'MRE' : [MRE], 'R2Training' : [R2TrainingScore], 'R2Test': [R2TestScore], '#BHI' :[CountOutOFConf] },
                                          columns=['HHnr','Reële som EEC','som voorspelde EEC','Gemiddelde Sigma','R2Training','R2Test','MSE','MAPE','#BHI','MAE','MRE'])
                                        #data= [nr, sumrealValue,sumpredictedValue,meanSigma,MSE,MAE,MAPE,MRE],
                                        #columns=['HH nr','Reële som EEC','Voorspelling som EEC','Gemiddelde Sigma','MSE','MAE','MAPE','MRE'])
        totalresultsDataFrame = totalresultsDataFrame.append(HHResultdataframe)
        if(plotBoolean ==True):
            GP.plotGPResult(featureTestSet,EECprediction,sigma)

        #TODO pas theta en nugget aan

    if(handleNotHome):
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100
        totalresultsDataFrame = totalresultsDataFrame.set_index([range(0,len(totalresultsDataFrame),1)])
        print(totalresultsDataFrame)
        print('length', len(totalresultsDataFrame))
        for i in range(0,len(totalresultsDataFrame),1):

            print('totI',totalresultsDataFrame['Reële som EEC'][i])
            if(totalresultsDataFrame['Reële som EEC'][i]==0):
                print("notHome Altered")
                test = notHomeError*totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                #((100-18)/100)*100
                totalresultsDataFrame['Reële som EEC'][i] = test #totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])


        sumRowDf = totalresultsDataFrame.sum(0)
        sumRowDf[['MAE']] = (sumRowDf[['MAE']]/(nrOfHH)) #TODO
        sumRowDf[['MRE']] = (sumRowDf[['MRE']]/(nrOfHH))
        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        end = len(totalresultsDataFrame)-1
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']))*100


        print(totalresultsDataFrame)



    else:
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        sumRowDf = totalresultsDataFrame.sum(0)
        #sumRowDf[['MAE']] = (sumRowDf[['MAE']]/(nrOfHH+1))
        #sumRowDf[['MRE']] = (sumRowDf[['MRE']]/(nrOfHH+1))

        sumRowDf[['MAE']] = (sumRowDf[['MAE']]/(nrOfHH)) #TODO
        sumRowDf[['MRE']] = (sumRowDf[['MRE']]/(nrOfHH))
        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100


    totalresultsDataFrame = np.around(totalresultsDataFrame.astype(np.double),2)
    totalresultsDataFrame['HHnr'][NumberOfHouseholds+1]='Total'
    endPrediction = time.time()

    elapsedTimePrediction = round((endPrediction - startPrediction),2)
    elapsedTimestring = '[ET featureVector= '+ str(elapsedTimeFeatureVector)+'s, ET voorspelling= '+str(elapsedTimePrediction)+'s]'
    timeDf = pd.DataFrame({'HHnr':['Tijd'],'Reële som EEC' : [elapsedTimestring]})
    totalresultsDataFrame = totalresultsDataFrame.append(timeDf,ignore_index=True)
    totalresultsDataFrame = totalresultsDataFrame[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','Gemiddelde Sigma','R2Training','R2Test','MSE','MAPE','#BHI','MAE','MRE']]


    totalresultsDataFrame.to_csv(str(resultsLocation)+str(csvFileName)+'.csv', "\t",index=False,thousands=',')
    print('results are saved to '+str(csvFileName)+'.csv'+' at location: '+str(resultsLocation))
    return totalresultsDataFrame

def predictionResultspyGP(dfConsumption,beginDateTraining,endDateTraining,beginDateTest,endDateTest,nrOfHH,sampleRate,featureVectorsAreCreated,optimizeHyperParameters,kernelfunction,plotBoolean,csvFileName,handleNotHome=False):
    notHomeError=1-0.10 # the error which ia used for when the HH is not home at the time of the prediction .15/(1-.15) = HH error

    startFeatures = time.time()
    NumberOfHouseholds = nrOfHH
    if (featureVectorsAreCreated == False):
        for nr in range(0, NumberOfHouseholds+1):
            GP.makefeatureVectors(dfConsumption,dfTemp,dfIrr,beginDateTraining,endDateTraining,beginDateTest,endDateTest,sampleRate,'HH'+str(nr),str(csvFileName)+'_DATAFRAME_training_HH'+str(nr),str(csvFileName)+'_DATAFRAME_test_HH'+str(nr))
    totalresultsDataFrame = pd.DataFrame()
    endFeatures = time.time()
    elapsedTimeFeatureVector = round((endFeatures - startFeatures),2)

    startPrediction = time.time()

    for nr in range(0, NumberOfHouseholds+1):
        print('predicting HH'+str(nr))
        featureTrainingSet, featureTestSet = GP.loadFeatureVectors(str(csvFileName)+'_DATAFRAME_training_HH'+str(nr),str(csvFileName)+'_DATAFRAME_test_HH'+str(nr))
        numberOfTestSamples = len(featureTestSet)

        EECprediction,sigma,MAE,MRE  = GP.predictpyGP(featureTrainingSet,featureTestSet)
        print('MRE',MRE)
        sumrealValue = sum(featureTestSet.values[:, 0])
        sumpredictedValue =sum(EECprediction)
        meanSigma = np.mean(sigma)
        CountOutOFConf = CountOutsideConfidenceInterval(EECprediction,featureTestSet.values[:, 0],sigma)
        HHResultdataframe = pd.DataFrame( {'HHnr' : [nr], 'Reële som EEC' : [sumrealValue],'som voorspelde EEC' : [sumpredictedValue],
                                            'MAE' : [MAE],'MRE': [MRE]},
                                          columns=['HHnr','Reële som EEC','som voorspelde EEC','MAE','MRE'])
                     #data= [nr, sumrealValue,sumpredictedValue,meanSigma,MSE,MAE,MAPE,MRE],
                                        #columns=['HH nr','Reële som EEC','Voorspelling som EEC','Gemiddelde Sigma','MSE','MAE','MAPE','MRE'])
        totalresultsDataFrame = totalresultsDataFrame.append(HHResultdataframe)
        if(plotBoolean ==True):
            GP.plotGPResultpyGP(featureTestSet,EECprediction,sigma)



    if(handleNotHome):
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100
        totalresultsDataFrame = totalresultsDataFrame.set_index([range(0,len(totalresultsDataFrame),1)])
        print(totalresultsDataFrame)
        print('length', len(totalresultsDataFrame))
        for i in range(0,len(totalresultsDataFrame),1):

            print('totI',totalresultsDataFrame['Reële som EEC'][i])
            if(totalresultsDataFrame['Reële som EEC'][i]==0):
                print("notHome Altered")
                test = notHomeError*totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                #((100-18)/100)*100
                totalresultsDataFrame['Reële som EEC'][i] = test #totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])


        sumRowDf = totalresultsDataFrame.sum(0)
        sumRowDf[['MAE']] = (sumRowDf[['MAE']]/(nrOfHH)) #TODO
        sumRowDf[['MRE']] = (sumRowDf[['MRE']]/(nrOfHH))
        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        end = len(totalresultsDataFrame)-1
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']))*100





    else:
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        sumRowDf = totalresultsDataFrame.sum(0)
        sumRowDf[['MAE']] = (sumRowDf[['MAE']]/(nrOfHH)) #TODO
        sumRowDf[['MRE']] = (sumRowDf[['MRE']]/(nrOfHH))
        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100



    totalresultsDataFrame = np.around(totalresultsDataFrame.astype(np.double),2)


    totalresultsDataFrame['HHnr'][NumberOfHouseholds+1]='Total'
    endPrediction = time.time()

    elapsedTimePrediction = round((endPrediction - startPrediction),2)
    elapsedTimestring = '[ET featureVector= '+ str(elapsedTimeFeatureVector)+'s, ET voorspelling= '+str(elapsedTimePrediction)+'s]'
    timeDf = pd.DataFrame({'HHnr':['Tijd'],'Reële som EEC' : [elapsedTimestring]})
    totalresultsDataFrame = totalresultsDataFrame.append(timeDf,ignore_index=True)
    totalresultsDataFrame = totalresultsDataFrame[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','MAE','MRE']]


    totalresultsDataFrame.to_csv(str(resultsLocation)+str(csvFileName)+'.csv', "\t",index=False,thousands=',')
    print('results are saved to '+str(csvFileName)+'.csv'+' at location: '+str(resultsLocation))
    return totalresultsDataFrame

def predictionResultspyGPScaled(dfConsumption,beginDateTraining,endDateTraining,beginDateTest,endDateTest,nrOfHH,sampleRate,featureVectorsAreCreated,optimizeHyperParameters,kernelfunction,plotBoolean,csvFileName,handleNotHome=False):
    notHomeError=1-0.15 # the error which is used for when the HH is not home at the time of the prediction .15/(1-.15) = HH error

    startFeatures = time.time()
    NumberOfHouseholds = nrOfHH
    if (featureVectorsAreCreated == False):
        for nr in range(0, NumberOfHouseholds+1):
            GP.makefeatureVectors(dfConsumption,dfTemp,dfIrr,beginDateTraining,endDateTraining,beginDateTest,endDateTest,sampleRate,'HH'+str(nr),str(csvFileName)+'_DATAFRAME_training_HH'+str(nr),str(csvFileName)+'_DATAFRAME_test_HH'+str(nr))
    totalresultsDataFrame = pd.DataFrame()
    endFeatures = time.time()
    elapsedTimeFeatureVector = round((endFeatures - startFeatures),2)

    startPrediction = time.time()

    for nr in range(0, NumberOfHouseholds+1):
        print('predicting HH'+str(nr))
        featureTrainingSet, featureTestSet = GP.loadFeatureVectors(str(csvFileName)+'_DATAFRAME_training_HH'+str(nr),str(csvFileName)+'_DATAFRAME_test_HH'+str(nr))
        numberOfTestSamples = len(featureTestSet)




        EECprediction,sigma,MSE  = GP.predictpyGP(featureTrainingSet,featureTestSet,scaled=True)

        sumrealValue = sum(featureTestSet.values[:, 0])

        sumpredictedValue =sum(EECprediction)
        meanSigma = np.mean(sigma)
        CountOutOFConf = CountOutsideConfidenceInterval(EECprediction,featureTestSet.values[:, 0],sigma)
        HHResultdataframe = pd.DataFrame( {'HHnr' : [nr], 'Reële som EEC' : [sumrealValue],'som voorspelde EEC' : [sumpredictedValue],'Gemiddelde Sigma' : [meanSigma],
                                            'MSE' : [MSE], '#BHI' :[CountOutOFConf] },
                                          columns=['HHnr','Reële som EEC','som voorspelde EEC','Gemiddelde Sigma','MSE','#BHI'])
                                        #data= [nr, sumrealValue,sumpredictedValue,meanSigma,MSE,MAE,MAPE,MRE],
                                        #columns=['HH nr','Reële som EEC','Voorspelling som EEC','Gemiddelde Sigma','MSE','MAE','MAPE','MRE'])
        totalresultsDataFrame = totalresultsDataFrame.append(HHResultdataframe)
        if(plotBoolean ==True):
            GP.plotGPResultpyGP(featureTestSet,EECprediction,sigma)



    if(handleNotHome):
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100
        totalresultsDataFrame = totalresultsDataFrame.set_index([range(0,len(totalresultsDataFrame),1)])
        print(totalresultsDataFrame)
        print('length', len(totalresultsDataFrame))
        for i in range(0,len(totalresultsDataFrame),1):

            print('totI',totalresultsDataFrame['Reële som EEC'][i])
            if(totalresultsDataFrame['Reële som EEC'][i]==0):
                print("notHome Altered")
                test = notHomeError*totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                #((100-18)/100)*100
                totalresultsDataFrame['Reële som EEC'][i] = test #totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])


        sumRowDf = totalresultsDataFrame.sum(0)
        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        end = len(totalresultsDataFrame)-1
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']))*100





    else:
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        sumRowDf = totalresultsDataFrame.sum(0)
        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100



    totalresultsDataFrame = np.around(totalresultsDataFrame.astype(np.double),2)


    totalresultsDataFrame['HHnr'][NumberOfHouseholds+1]='Total'
    endPrediction = time.time()

    elapsedTimePrediction = round((endPrediction - startPrediction),2)
    elapsedTimestring = '[ET featureVector= '+ str(elapsedTimeFeatureVector)+'s, ET voorspelling= '+str(elapsedTimePrediction)+'s]'
    timeDf = pd.DataFrame({'HHnr':['Tijd'],'Reële som EEC' : [elapsedTimestring]})
    totalresultsDataFrame = totalresultsDataFrame.append(timeDf,ignore_index=True)
    totalresultsDataFrame = totalresultsDataFrame[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','Gemiddelde Sigma','MSE','#BHI']]


    totalresultsDataFrame.to_csv(str(resultsLocation)+str(csvFileName)+'.csv', "\t",index=False,thousands=',')
    print('results are saved to '+str(csvFileName)+'.csv'+' at location: '+str(resultsLocation))
    return totalresultsDataFrame

def predictionResultsBaseline(dfConsumption,beginDateTraining,endDateTraining,beginDateTest,endDateTest,nrOfHH,sampleRate,featureVectorsAreCreated,optimizeHyperParameters,kernelfunction,plotBoolean,csvFileName,handleNotHome=False,nrOfPredictionDays=7):
    notHomeError=1-0.15 # the error which ia used for when the HH is not home at the time of the prediction .15/(1-.15) = HH error

    startFeatures = time.time()
    NumberOfHouseholds = nrOfHH

    endFeatures = time.time()
    elapsedTimeFeatureVector = round((endFeatures - startFeatures),2)

    startPrediction = time.time()
    totalresultsDataFrame = pd.DataFrame()
    for nr in range(0, NumberOfHouseholds+1):
        print('predicting HH'+str(nr))

        EECprediction,testSet,MAE,MRE = BM.predictConsumption(dfConsumption, beginDateTest,endDateTest,sampleRate,nrOfPredictionDays,'HH'+str(nr))



        sumrealValue = sum(testSet.values[:, 0])
        sumpredictedValue =sum(EECprediction)


        HHResultdataframe = pd.DataFrame( {'HHnr' : [nr], 'Reële som EEC' : [sumrealValue],'som voorspelde EEC' : [sumpredictedValue],
                                            'MAE' : [MAE],'MRE': [MRE]},
                                          columns=['HHnr','Reële som EEC','som voorspelde EEC','MAE','MRE'])

        totalresultsDataFrame = totalresultsDataFrame.append(HHResultdataframe)
        if(plotBoolean ==True):
            BM.plot(testSet.values[:,0],EECprediction,"consumptie",0,70000)



    if(handleNotHome):
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100
        totalresultsDataFrame = totalresultsDataFrame.set_index([range(0,len(totalresultsDataFrame),1)])
        print(totalresultsDataFrame)
        print('length', len(totalresultsDataFrame))
        for i in range(0,len(totalresultsDataFrame),1):

            print('totI',totalresultsDataFrame['Reële som EEC'][i])
            if(totalresultsDataFrame['Reële som EEC'][i]==0):
                print("notHome Altered")
                test = notHomeError*totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                #((100-18)/100)*100
                totalresultsDataFrame['Reële som EEC'][i] = test #totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])


        sumRowDf = totalresultsDataFrame.sum(0)
        print('NRHH',nrOfHH+1)
        sumRowDf[['MAE']] = (sumRowDf[['MAE']]/(nrOfHH+1))
        sumRowDf[['MRE']] = (sumRowDf[['MRE']]/(nrOfHH+1))
        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        end = len(totalresultsDataFrame)-1
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']))*100





    else:
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        sumRowDf = totalresultsDataFrame.sum(0)
        sumRowDf[['MAE']] = (sumRowDf[['MAE']]/(nrOfHH))
        sumRowDf[['MRE']] = (sumRowDf[['MRE']]/(nrOfHH))
        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100



    totalresultsDataFrame = np.around(totalresultsDataFrame.astype(np.double),2)


    totalresultsDataFrame['HHnr'][NumberOfHouseholds+1]='Total'
    endPrediction = time.time()

    elapsedTimePrediction = round((endPrediction - startPrediction),2)
    elapsedTimestring = '[ET featureVector= '+ str(elapsedTimeFeatureVector)+'s, ET voorspelling= '+str(elapsedTimePrediction)+'s]'
    timeDf = pd.DataFrame({'HHnr':['Tijd'],'Reële som EEC' : [elapsedTimestring]})
    totalresultsDataFrame = totalresultsDataFrame.append(timeDf,ignore_index=True)
    totalresultsDataFrame = totalresultsDataFrame[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','MAE','MRE']]


    totalresultsDataFrame.to_csv(str(resultsLocation)+str(csvFileName)+'.csv', "\t",index=False,thousands=',')
    print('results are saved to '+str(csvFileName)+'.csv'+' at location: '+str(resultsLocation))
    return totalresultsDataFrame


def predictionResultsSVR(dfConsumption,beginDateTraining,endDateTraining,beginDateTest,endDateTest,nrOfHH,sampleRate,featureVectorsAreCreated,optimizeHyperParameters,kernelfunction,plotBoolean,csvFileName,handleNotHome=False):
    notHomeError=1-0.15 # the error which ia used for when the HH is not home at the time of the prediction .15/(1-.15) = HH error

    startFeatures = time.time()
    NumberOfHouseholds = nrOfHH
    if (featureVectorsAreCreated == False):
        for nr in range(0, NumberOfHouseholds+1):
            GP.makefeatureVectors(dfConsumption,dfTemp,dfIrr,beginDateTraining,endDateTraining,beginDateTest,endDateTest,sampleRate,'HH'+str(nr),str(csvFileName)+'_DATAFRAME_training_HH'+str(nr),str(csvFileName)+'_DATAFRAME_test_HH'+str(nr))
    #totalresultsDataFrame = pd.DataFrame()
    endFeatures = time.time()
    elapsedTimeFeatureVector = round((endFeatures - startFeatures),2)

    startPrediction = time.time()
    totalresultsDataFrame = pd.DataFrame()
    for nr in range(0, NumberOfHouseholds+1):
        print('predicting HH'+str(nr))
        featureTrainingSet, featureTestSet = GP.loadFeatureVectors(str(csvFileName)+'_DATAFRAME_training_HH'+str(nr),str(csvFileName)+'_DATAFRAME_test_HH'+str(nr))
        numberOfTestSamples = len(featureTestSet)

        trainSetX = featureTrainingSet.values[:, 1:20]
        trainSetY = featureTrainingSet.values[:, 0]
        testSetX = featureTestSet.values[:, 1:20]
        testSetY = featureTestSet.values[:, 0]

        EECprediction,testSet,MAE,MRE  = SVR.predictConsumption(trainSetX,trainSetY,testSetX,testSetY)


        sumrealValue = sum(testSetY)
        sumpredictedValue =sum(EECprediction)

        HHResultdataframe = pd.DataFrame( {'HHnr' : [nr], 'Reële som EEC' : [sumrealValue],'som voorspelde EEC' : [sumpredictedValue],
                                            'MAE' : [MAE],'MRE':[MRE]},
                                          columns=['HHnr','Reële som EEC','som voorspelde EEC','MAE','MRE'])

        totalresultsDataFrame = totalresultsDataFrame.append(HHResultdataframe)
        if(plotBoolean ==True):
            SVR.plot(testSet.values[:,0],EECprediction,"consumptie",0,70000)






    if(handleNotHome):
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100
        totalresultsDataFrame = totalresultsDataFrame.set_index([range(0,len(totalresultsDataFrame),1)])
        print(totalresultsDataFrame)
        print('length', len(totalresultsDataFrame))
        for i in range(0,len(totalresultsDataFrame),1):

            print('totI',totalresultsDataFrame['Reële som EEC'][i])
            if(totalresultsDataFrame['Reële som EEC'][i]==0):
                print("notHome Altered")
                test = notHomeError*totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                #((100-18)/100)*100
                totalresultsDataFrame['Reële som EEC'][i] = test #totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])


        sumRowDf = totalresultsDataFrame.sum(0)
        sumRowDf[['MAE']] = (sumRowDf[['MAE']]/(nrOfHH+1))
        sumRowDf[['MRE']] = (sumRowDf[['MRE']]/(nrOfHH+1))



        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        end = len(totalresultsDataFrame)-1
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']))*100





    else:
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        sumRowDf = totalresultsDataFrame.sum(0)
        sumRowDf[['MAE']] = (sumRowDf[['MAE']]/(nrOfHH+1))
        sumRowDf[['MRE']] = (sumRowDf[['MRE']]/(nrOfHH+1))
        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100



    totalresultsDataFrame = np.around(totalresultsDataFrame.astype(np.double),2)


    totalresultsDataFrame['HHnr'][NumberOfHouseholds+1]='Total'
    endPrediction = time.time()

    elapsedTimePrediction = round((endPrediction - startPrediction),2)
    elapsedTimestring = '[ET featureVector= '+ str(elapsedTimeFeatureVector)+'s, ET voorspelling= '+str(elapsedTimePrediction)+'s]'
    timeDf = pd.DataFrame({'HHnr':['Tijd'],'Reële som EEC' : [elapsedTimestring]})
    totalresultsDataFrame = totalresultsDataFrame.append(timeDf,ignore_index=True)
    totalresultsDataFrame = totalresultsDataFrame[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','MAE','MRE']]


    totalresultsDataFrame.to_csv(str(resultsLocation)+str(csvFileName)+'.csv', "\t",index=False,thousands=',')
    print('results are saved to '+str(csvFileName)+'.csv'+' at location: '+str(resultsLocation))
    return totalresultsDataFrame


def predictionResultsOLS(dfConsumption,beginDateTraining,endDateTraining,beginDateTest,endDateTest,nrOfHH,sampleRate,featureVectorsAreCreated,optimizeHyperParameters,kernelfunction,plotBoolean,csvFileName,handleNotHome=False):
    notHomeError=1-0.15 # the error which ia used for when the HH is not home at the time of the prediction .15/(1-.15) = HH error

    startFeatures = time.time()
    NumberOfHouseholds = nrOfHH
    if (featureVectorsAreCreated == False):
        for nr in range(0, NumberOfHouseholds+1):
            GP.makefeatureVectors(dfConsumption,dfTemp,dfIrr,beginDateTraining,endDateTraining,beginDateTest,endDateTest,sampleRate,'HH'+str(nr),str(csvFileName)+'_DATAFRAME_training_HH'+str(nr),str(csvFileName)+'_DATAFRAME_test_HH'+str(nr))
    #totalresultsDataFrame = pd.DataFrame()
    endFeatures = time.time()
    elapsedTimeFeatureVector = round((endFeatures - startFeatures),2)

    startPrediction = time.time()
    totalresultsDataFrame = pd.DataFrame()
    for nr in range(0, NumberOfHouseholds+1):
        print('predicting HH'+str(nr))
        featureTrainingSet, featureTestSet = GP.loadFeatureVectors(str(csvFileName)+'_DATAFRAME_training_HH'+str(nr),str(csvFileName)+'_DATAFRAME_test_HH'+str(nr))
        numberOfTestSamples = len(featureTestSet)

        trainSetX = featureTrainingSet.values[:, 1:20]
        trainSetY = featureTrainingSet.values[:, 0]
        testSetX = featureTestSet.values[:, 1:20]
        testSetY = featureTestSet.values[:, 0]

        EECprediction,testSet,MAE,MRE  = OLS.predictConsumption(trainSetX,trainSetY,testSetX,testSetY)


        sumrealValue = sum(testSetY)
        sumpredictedValue =sum(EECprediction)

        HHResultdataframe = pd.DataFrame( {'HHnr' : [nr], 'Reële som EEC' : [sumrealValue],'som voorspelde EEC' : [sumpredictedValue],
                                            'MAE' : [MAE],'MRE':[MRE]},
                                          columns=['HHnr','Reële som EEC','som voorspelde EEC','MAE','MRE'])

        totalresultsDataFrame = totalresultsDataFrame.append(HHResultdataframe)
        if(plotBoolean ==True):
            SVR.plot(testSet.values[:,0],EECprediction,"consumptie",0,70000)






    if(handleNotHome):
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100
        totalresultsDataFrame = totalresultsDataFrame.set_index([range(0,len(totalresultsDataFrame),1)])
        print(totalresultsDataFrame)
        print('length', len(totalresultsDataFrame))
        for i in range(0,len(totalresultsDataFrame),1):

            print('totI',totalresultsDataFrame['Reële som EEC'][i])
            if(totalresultsDataFrame['Reële som EEC'][i]==0):
                print("notHome Altered")
                test = notHomeError*totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                #((100-18)/100)*100
                totalresultsDataFrame['Reële som EEC'][i] = test #totalresultsDataFrame['som voorspelde EEC'].iloc[i]
                totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])


        sumRowDf = totalresultsDataFrame.sum(0)
        sumRowDf[['MAE']] = (sumRowDf[['MAE']]/(nrOfHH+1))
        sumRowDf[['MRE']] = (sumRowDf[['MRE']]/(nrOfHH+1))



        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        end = len(totalresultsDataFrame)-1
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']))*100





    else:
        totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)'] =abs(totalresultsDataFrame['som voorspelde EEC']-totalresultsDataFrame['Reële som EEC'])
        sumRowDf = totalresultsDataFrame.sum(0)
        sumRowDf[['MAE']] = (sumRowDf[['MAE']]/(nrOfHH+1))
        sumRowDf[['MRE']] = (sumRowDf[['MRE']]/(nrOfHH+1))
        totalresultsDataFrame = totalresultsDataFrame.append(sumRowDf,ignore_index=True)
        totalresultsDataFrame['% fout'] = (totalresultsDataFrame['abs(voorsp. EEC - Re. EEC)']/(totalresultsDataFrame['Reële som EEC']+1))*100



    totalresultsDataFrame = np.around(totalresultsDataFrame.astype(np.double),2)


    totalresultsDataFrame['HHnr'][NumberOfHouseholds+1]='Total'
    endPrediction = time.time()

    elapsedTimePrediction = round((endPrediction - startPrediction),2)
    elapsedTimestring = '[ET featureVector= '+ str(elapsedTimeFeatureVector)+'s, ET voorspelling= '+str(elapsedTimePrediction)+'s]'
    timeDf = pd.DataFrame({'HHnr':['Tijd'],'Reële som EEC' : [elapsedTimestring]})
    totalresultsDataFrame = totalresultsDataFrame.append(timeDf,ignore_index=True)
    totalresultsDataFrame = totalresultsDataFrame[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','MAE','MRE']]


    totalresultsDataFrame.to_csv(str(resultsLocation)+str(csvFileName)+'.csv', "\t",index=False,thousands=',')
    print('results are saved to '+str(csvFileName)+'.csv'+' at location: '+str(resultsLocation))
    return totalresultsDataFrame



def returnAndSaveTotalList(dfList,csvFileName,thousandSeperator):
    pd.options.mode.chained_assignment = None  # default='warn'

    totalDf = pd.DataFrame()

    for i in range (0,len(dfList)):
         df = dfList[i].tail(2).head(1)
         df['HHnr']=i
         totalDf = totalDf.append(df)
    sumRowDf = totalDf.sum(0)
    print('aantal files:',len(dfList))
    #sumRowDf[['MAE']]=(sumRowDf[['MAE']]/len(dfList))
    #sumRowDf[['MRE']]=(sumRowDf[['MRE']]/len(dfList))
    sumRowDf['% fout'] = (sumRowDf['abs(voorsp. EEC - Re. EEC)']/sumRowDf['Reële som EEC'])*100
    sumRowDf[['Gemiddelde Sigma','R2Training','R2Test','MSE','MAE','MAPE','MRE']]=(sumRowDf[['Gemiddelde Sigma','R2Training','R2Test','MSE','MAE','MAPE','MRE']]/len(dfList))
    totalDf = totalDf.append(sumRowDf,ignore_index=True)
    totalDf = totalDf[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','Gemiddelde Sigma','R2Training','R2Test','MSE','MAE','MAPE','MRE','#BHI']]

    totalDf['HHnr'][len(dfList)]='Total'

    timeDf = pd.DataFrame({'HHnr':['Dummy']})
    totalDf = totalDf.append(timeDf,ignore_index=True)
    totalDf = totalDf[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','Gemiddelde Sigma','R2Training','R2Test','MSE','MAPE','#BHI','MAE','MRE']]
    totalDf = totalDf[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','#BHI','MAE','MRE']]

    #pd.set_option('display.float_format',
    #  lambda x: '{:,.4f}'.format(x) if abs(x) < 10000 else '{:,.0f}'.format(x))
    if(thousandSeperator):
        totalDf=totalDf.applymap(lambda x: "{:,}".format(x) if (isinstance(x,float))else x)

    totalDf.to_csv(str(resultsLocation)+str(csvFileName)+'.csv', sep= "\t",index=False)
    print('results are saved to '+str(csvFileName)+'.csv'+' at location: '+str(resultsLocation))
    return totalDf


def returnAndSaveTotalListpyGP(dfList,csvFileName,thousandSeperator):
    pd.options.mode.chained_assignment = None  # default='warn'

    totalDf = pd.DataFrame()

    for i in range (0,len(dfList)):
         df = dfList[i].tail(2).head(1)
         df['HHnr']=i
         totalDf = totalDf.append(df)
    sumRowDf = totalDf.sum(0)
    sumRowDf['% fout'] = (sumRowDf['abs(voorsp. EEC - Re. EEC)']/sumRowDf['Reële som EEC'])*100
    sumRowDf[['Gemiddelde Sigma','MSE']]=(sumRowDf[['Gemiddelde Sigma','MSE']]/len(dfList))
    totalDf = totalDf.append(sumRowDf,ignore_index=True)
    totalDf = totalDf[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','Gemiddelde Sigma','MSE','#BHI']]

    totalDf['HHnr'][len(dfList)]='Total'

    timeDf = pd.DataFrame({'HHnr':['Dummy']})
    totalDf = totalDf.append(timeDf,ignore_index=True)
    totalDf = totalDf[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','Gemiddelde Sigma','MSE','#BHI']]
    #totalDf = totalDf[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','#BHI']]

    #pd.set_option('display.float_format',
    #  lambda x: '{:,.4f}'.format(x) if abs(x) < 10000 else '{:,.0f}'.format(x))
    if(thousandSeperator):
        totalDf=totalDf.applymap(lambda x: "{:,}".format(x) if (isinstance(x,float))else x)

    totalDf.to_csv(str(resultsLocation)+str(csvFileName)+'.csv', sep= "\t",index=False)
    print('results are saved to '+str(csvFileName)+'.csv'+' at location: '+str(resultsLocation))
    return totalDf


def returnAndSaveTotalListBaseline(dfList,csvFileName,thousandSeperator):
    pd.options.mode.chained_assignment = None  # default='warn'

    totalDf = pd.DataFrame()

    for i in range (0,len(dfList)):
         df = dfList[i].tail(2).head(1)
         df['HHnr']=i
         totalDf = totalDf.append(df)
    sumRowDf = totalDf.sum(0)
    sumRowDf['% fout'] = (sumRowDf['abs(voorsp. EEC - Re. EEC)']/sumRowDf['Reële som EEC'])*100
    sumRowDf[['MAE']]=(sumRowDf[['MAE']]/len(dfList))
    sumRowDf[['MRE']]=(sumRowDf[['MRE']]/len(dfList))
    totalDf = totalDf.append(sumRowDf,ignore_index=True)
    totalDf = totalDf[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','MAE','MRE']]

    totalDf['HHnr'][len(dfList)]='Total'

    timeDf = pd.DataFrame({'HHnr':['Dummy']})
    totalDf = totalDf.append(timeDf,ignore_index=True)
    totalDf = totalDf[['HHnr','som voorspelde EEC','Reële som EEC','abs(voorsp. EEC - Re. EEC)','% fout','MAE','MRE']]


    if(thousandSeperator):
        totalDf=totalDf.applymap(lambda x: "{:,}".format(x) if (isinstance(x,float))else x)

    totalDf.to_csv(str(resultsLocation)+str(csvFileName)+'.csv', sep= "\t",index=False)
    print('results are saved to '+str(csvFileName)+'.csv'+' at location: '+str(resultsLocation))
    return totalDf



'''
RESULTS:
'''

def testTotalYearMonthlyDaily2DayspyGP():

        start = time.time()
        loaded2 =False
        loaded4 = False
        loaded5 = False
        loaded6= False


        dfC4 = readConsumptionDataCSV(inputDataPath,'Measurements_HP')
        #resultDFMeteoSpringDaily_HP = predictionResultspyGP(dfC4,'2014-03-01 00:00:00','2014-05-28 23:45:00','2014-05-29 00:00:00','2014-05-31 23:45:00',2,'H',loaded4,False,'linear',True,'season_HP_Met_springDaily')


        Jan3Daily_HP = predictionResultspyGP(dfC4,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-27 00:00:00','2014-01-28 23:45:00',10,'H',loaded4,False,'linear',False,'Monthly_HP_JanHpyGP')
        Feb3Daily_HP = predictionResultspyGP(dfC4,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_FebHpyGP')
        Maa3Daily_HP = predictionResultspyGP(dfC4,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_MaaHpyGP')
        Apr3Daily_HP = predictionResultspyGP(dfC4,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_AprHpyGP')
        Mei3Daily_HP = predictionResultspyGP(dfC4,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_MeiHpyGP')
        Jun3Daily_HP = predictionResultspyGP(dfC4,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JunHpyGP')
        Jul3Daily_HP = predictionResultspyGP(dfC4,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JulHpyGP')
        Aug3Daily_HP = predictionResultspyGP(dfC4,'2014-08-01 00:00:00','2014-08-24 23:45:00','2014-08-27 00:00:00','2014-08-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_AugHpyGP')
        Sep3Daily_HP = predictionResultspyGP(dfC4,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_SepHpyGP')
        Okt3Daily_HP = predictionResultspyGP(dfC4,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_OktHpyGP')
        Nov3Daily_HP = predictionResultspyGP(dfC4,'2013-11-01 00:00:00','2013-11-24 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_NovHpyGP')
        Dec3Daily_HP = predictionResultspyGP(dfC4,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_DecHpyGP')

        totalHP = returnAndSaveTotalListBaseline([Jan3Daily_HP,Feb3Daily_HP,Maa3Daily_HP,Apr3Daily_HP,Mei3Daily_HP,Jun3Daily_HP,Jul3Daily_HP,Aug3Daily_HP,Sep3Daily_HP,Okt3Daily_HP,Nov3Daily_HP,Dec3Daily_HP],'TotalHP_pyGP',False)



        dfC2 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other_HP')



        Jan3Daily_Ot_HP = predictionResultspyGP(dfC2,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-27 00:00:00','2014-01-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JanHpyGP')
        Feb3Daily_Ot_HP = predictionResultspyGP(dfC2,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_FebHpyGP')
        Maa3Daily_Ot_HP = predictionResultspyGP(dfC2,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MaaHpyGP')
        Apr3Daily_Ot_HP = predictionResultspyGP(dfC2,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AprHpyGP')
        Mei3Daily_Ot_HP = predictionResultspyGP(dfC2,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MeiHpyGP')
        Jun3Daily_Ot_HP = predictionResultspyGP(dfC2,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JunHpyGP')
        Jul3Daily_Ot_HP = predictionResultspyGP(dfC2,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JulHpyGP')
        Aug3Daily_Ot_HP = predictionResultspyGP(dfC2,'2014-08-01 00:00:00','2014-08-26 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AugHpyGP')
        Sep3Daily_Ot_HP = predictionResultspyGP(dfC2,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_SepHpyGP')
        Okt3Daily_Ot_HP = predictionResultspyGP(dfC2,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_OktHpyGP')
        Nov3Daily_Ot_HP = predictionResultspyGP(dfC2,'2013-11-01 00:00:00','2013-11-24 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_NovHpyGP')
        Dec3Daily_Ot_HP = predictionResultspyGP(dfC2,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_DecHpyGP')


        total_ot_HP = returnAndSaveTotalListBaseline([Jan3Daily_Ot_HP,Feb3Daily_Ot_HP,Maa3Daily_Ot_HP,Apr3Daily_Ot_HP,Mei3Daily_Ot_HP,Jun3Daily_Ot_HP,Jul3Daily_Ot_HP,Aug3Daily_Ot_HP,Sep3Daily_Ot_HP,Okt3Daily_Ot_HP,Nov3Daily_Ot_HP,Dec3Daily_Ot_HP],'Total_Ot_HP_pyGP',False)

        '''
        dfC5 = readConsumptionDataCSV(inputDataPath,'Measurements_PV')
        Jan3Daily_PV = predictionResultspyGP(dfC5,'2014-01-01 00:00:00','2014-01-27 23:45:00','2014-01-29 00:00:00','2014-01-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_JanHpygP')
        Feb3Daily_PV = predictionResultspyGP(dfC5,'2014-02-01 00:00:00','2014-02-27 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_FebHpygP')
        Maa3Daily_PV = predictionResultspyGP(dfC5,'2014-03-01 00:00:00','2014-03-27 23:45:00','2014-03-29 00:00:00','2014-03-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_MaaHpygP')
        Apr3Daily_PV = predictionResultspyGP(dfC5,'2014-04-01 00:00:00','2014-04-27 23:45:00','2014-04-29 00:00:00','2014-04-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_AprHpygP')
        Mei3Daily_PV = predictionResultspyGP(dfC5,'2014-05-01 00:00:00','2014-05-27 23:45:00','2014-05-29 00:00:00','2014-05-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_MeiHpygP')
        Jun3Daily_PV = predictionResultspyGP(dfC5,'2014-06-01 00:00:00','2014-06-27 23:45:00','2014-06-29 00:00:00','2014-06-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_JunHpygP')
        Jul3Daily_PV = predictionResultspyGP(dfC5,'2014-07-01 00:00:00','2014-07-27 23:45:00','2014-07-29 00:00:00','2014-07-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_JulHpygP')
        Aug3Daily_PV = predictionResultspyGP(dfC5,'2014-08-01 00:00:00','2014-08-27 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_AugHpygP')
        Sep3Daily_PV = predictionResultspyGP(dfC5,'2013-09-01 00:00:00','2013-09-27 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_SepHpygP')
        Okt3Daily_PV = predictionResultspyGP(dfC5,'2013-10-01 00:00:00','2013-10-27 23:45:00','2013-10-29 00:00:00','2013-10-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_OktHpygP')
        Nov3Daily_PV = predictionResultspyGP(dfC5,'2013-11-01 00:00:00','2013-11-27 23:45:00','2013-11-29 00:00:00','2013-11-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_NovHpygP')
        Dec3Daily_PV = predictionResultspyGP(dfC5,'2013-12-01 00:00:00','2013-12-27 23:45:00','2013-12-29 00:00:00','2013-12-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_DecHpygP')

        totalPV = returnAndSaveTotalListBaseline([Jan3Daily_PV,Feb3Daily_PV,Maa3Daily_PV,Apr3Daily_PV,Mei3Daily_PV,Jun3Daily_PV,Jul3Daily_PV,Aug3Daily_PV,Sep3Daily_PV,Okt3Daily_PV,Nov3Daily_PV,Dec3Daily_PV],'TotalPV_pyGP',False)
        '''

        dfC6 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other')

        Jan3Daily_Ot = predictionResultspyGP(dfC6,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-30 00:00:00','2014-01-31 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JanHpyGP')
        Feb3Daily_Ot = predictionResultspyGP(dfC6,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_FebHpyGP')
        Maa3Daily_Ot = predictionResultspyGP(dfC6,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_MaaHpyGP')
        Apr3Daily_Ot = predictionResultspyGP(dfC6,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_AprHpyGP')
        Mei3Daily_Ot = predictionResultspyGP(dfC6,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_MeiHpyGP')
        Jun3Daily_Ot = predictionResultspyGP(dfC6,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JunHpyGP')
        Jul3Daily_Ot = predictionResultspyGP(dfC6,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JulHpyGP')
        Aug3Daily_Ot = predictionResultspyGP(dfC6,'2014-08-01 00:00:00','2014-08-26 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_AugHpyGP')
        Sep3Daily_Ot = predictionResultspyGP(dfC6,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_SepHpyGP')
        Okt3Daily_Ot = predictionResultspyGP(dfC6,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_OktHpyGP')
        Nov3Daily_Ot = predictionResultspyGP(dfC6,'2013-11-01 00:00:00','2013-11-26 23:45:00','2013-11-29 00:00:00','2013-11-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_NovHpyGP')
        Dec3Daily_Ot = predictionResultspyGP(dfC6,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_DecHpyGP')

        totalOt = returnAndSaveTotalListBaseline([Jan3Daily_Ot,Feb3Daily_Ot,Maa3Daily_Ot,Apr3Daily_Ot,Mei3Daily_Ot,Jun3Daily_Ot,Jul3Daily_Ot,Aug3Daily_Ot,Sep3Daily_Ot,Okt3Daily_Ot,Nov3Daily_Ot,Dec3Daily_Ot],'TotalOt_pyGP',False)


        #Conclusion = returnAndSaveTotalListpyGP([totalHP,totalOt],'conclusionConsumption',False)


        end = time.time()
        print('monthlyPredictionResults2DaysHourly Elapsed Time: '+str(end - start))


def testScaled():
        start = time.time()
        loaded2 =False
        loaded4 = False
        loaded5 = False
        loaded6= False

        dfC4 = readConsumptionDataCSV(inputDataPath,'Measurements_HP')
        #resultDFMeteoSpringDaily_HP = predictionResultspyGP(dfC4,'2014-03-01 00:00:00','2014-05-28 23:45:00','2014-05-29 00:00:00','2014-05-31 23:45:00',2,'H',loaded4,False,'linear',True,'season_HP_Met_springDaily')


        Jan3Daily_HP = predictionResultspyGPScaled(dfC4,'2014-01-01 00:00:00','2014-01-27 23:45:00','2014-01-30 00:00:00','2014-01-31 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JanHpyGP_SC')
        Feb3Daily_HP = predictionResultspyGPScaled(dfC4,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_FebHpyGP_SC')
        Maa3Daily_HP = predictionResultspyGPScaled(dfC4,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_MaaHpyGP_SC')
        Apr3Daily_HP = predictionResultspyGPScaled(dfC4,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_AprHpyGP_SC')
        Mei3Daily_HP = predictionResultspyGPScaled(dfC4,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_MeiHpyGP_SC')
        Jun3Daily_HP = predictionResultspyGPScaled(dfC4,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JunHpyGP_SC')
        Jul3Daily_HP = predictionResultspyGPScaled(dfC4,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JulHpyGP_SC')
        Aug3Daily_HP = predictionResultspyGPScaled(dfC4,'2014-08-01 00:00:00','2014-08-24 23:45:00','2014-08-27 00:00:00','2014-08-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_AugHpyGP_SC')
        Sep3Daily_HP = predictionResultspyGPScaled(dfC4,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_SepHpyGP_SC')
        Okt3Daily_HP = predictionResultspyGPScaled(dfC4,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_OktHpyGP_SC')
        Nov3Daily_HP = predictionResultspyGPScaled(dfC4,'2013-11-01 00:00:00','2013-11-24 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_NovHpyGP_SC')
        Dec3Daily_HP = predictionResultspyGPScaled(dfC4,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_DecHpyGP_SC')

        totalHP = returnAndSaveTotalListpyGP([Jan3Daily_HP,Feb3Daily_HP,Maa3Daily_HP,Apr3Daily_HP,Mei3Daily_HP,Jun3Daily_HP,Jul3Daily_HP,Aug3Daily_HP,Sep3Daily_HP,Okt3Daily_HP,Nov3Daily_HP,Dec3Daily_HP],'TotalHP_SC',False)



        dfC2 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other_HP')


        Jan3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-27 00:00:00','2014-01-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JanHpygP_SC')
        Feb3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_FebHpygP_SC')
        Maa3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MaaHpygP_SC')
        Apr3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AprHpygP_SC')
        Mei3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2014-05-01 00:00:00','2014-05-26 23:45:00','2014-05-29 00:00:00','2014-05-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MeiHpygP_SC')
        Jun3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JunHpygP_SC')
        Jul3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JulHpygP_SC')
        Aug3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2014-08-01 00:00:00','2014-08-26 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AugHpygP_SC')
        Sep3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_SepHpygP_SC')
        Okt3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_OktHpygP_SC')
        Nov3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2013-11-01 00:00:00','2013-11-24 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_NovHpygP_SC')
        Dec3Daily_Ot_HP = predictionResultspyGPScaled(dfC2,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_DecHpygP_SC')


        total_ot_HP = returnAndSaveTotalListpyGP([Jan3Daily_Ot_HP,Feb3Daily_Ot_HP,Maa3Daily_Ot_HP,Apr3Daily_Ot_HP,Mei3Daily_Ot_HP,Jun3Daily_Ot_HP,Jul3Daily_Ot_HP,Aug3Daily_Ot_HP,Sep3Daily_Ot_HP,Okt3Daily_Ot_HP,Nov3Daily_Ot_HP,Dec3Daily_Ot_HP],'Total_Ot_HP_SC',False)

        '''
        dfC5 = readConsumptionDataCSV(inputDataPath,'Measurements_PV')
        Jan3Daily_PV = predictionResultspyGP(dfC5,'2014-01-01 00:00:00','2014-01-27 23:45:00','2014-01-29 00:00:00','2014-01-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_JanHpygP')
        Feb3Daily_PV = predictionResultspyGP(dfC5,'2014-02-01 00:00:00','2014-02-27 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_FebHpygP')
        Maa3Daily_PV = predictionResultspyGP(dfC5,'2014-03-01 00:00:00','2014-03-27 23:45:00','2014-03-29 00:00:00','2014-03-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_MaaHpygP')
        Apr3Daily_PV = predictionResultspyGP(dfC5,'2014-04-01 00:00:00','2014-04-27 23:45:00','2014-04-29 00:00:00','2014-04-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_AprHpygP')
        Mei3Daily_PV = predictionResultspyGP(dfC5,'2014-05-01 00:00:00','2014-05-27 23:45:00','2014-05-29 00:00:00','2014-05-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_MeiHpygP')
        Jun3Daily_PV = predictionResultspyGP(dfC5,'2014-06-01 00:00:00','2014-06-27 23:45:00','2014-06-29 00:00:00','2014-06-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_JunHpygP')
        Jul3Daily_PV = predictionResultspyGP(dfC5,'2014-07-01 00:00:00','2014-07-27 23:45:00','2014-07-29 00:00:00','2014-07-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_JulHpygP')
        Aug3Daily_PV = predictionResultspyGP(dfC5,'2014-08-01 00:00:00','2014-08-27 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_AugHpygP')
        Sep3Daily_PV = predictionResultspyGP(dfC5,'2013-09-01 00:00:00','2013-09-27 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_SepHpygP')
        Okt3Daily_PV = predictionResultspyGP(dfC5,'2013-10-01 00:00:00','2013-10-27 23:45:00','2013-10-29 00:00:00','2013-10-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_OktHpygP')
        Nov3Daily_PV = predictionResultspyGP(dfC5,'2013-11-01 00:00:00','2013-11-27 23:45:00','2013-11-29 00:00:00','2013-11-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_NovHpygP')
        Dec3Daily_PV = predictionResultspyGP(dfC5,'2013-12-01 00:00:00','2013-12-27 23:45:00','2013-12-29 00:00:00','2013-12-30 23:45:00',35,'H',loaded5,False,'linear',False,'Monthly_PV_DecHpygP')

        totalPV = returnAndSaveTotalListpyGP([Jan3Daily_PV,Feb3Daily_PV,Maa3Daily_PV,Apr3Daily_PV,Mei3Daily_PV,Jun3Daily_PV,Jul3Daily_PV,Aug3Daily_PV,Sep3Daily_PV,Okt3Daily_PV,Nov3Daily_PV,Dec3Daily_PV],'TotalPV',False)
        '''


        dfC6 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other')

        Jan3Daily_Ot = predictionResultspyGPScaled(dfC6,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-30 00:00:00','2014-01-31 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JanHpyGP_SC')
        Feb3Daily_Ot = predictionResultspyGPScaled(dfC6,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_FebHpyGP_SC')
        Maa3Daily_Ot = predictionResultspyGPScaled(dfC6,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_MaaHpyGP_SC')
        Apr3Daily_Ot = predictionResultspyGPScaled(dfC6,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_AprHpyGP_SC')
        Mei3Daily_Ot = predictionResultspyGPScaled(dfC6,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_MeiHpyGP_SC')
        Jun3Daily_Ot = predictionResultspyGPScaled(dfC6,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JunHpyGP_SC')
        Jul3Daily_Ot = predictionResultspyGPScaled(dfC6,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JulHpyGP_SC')
        Aug3Daily_Ot = predictionResultspyGPScaled(dfC6,'2014-08-01 00:00:00','2014-08-26 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_AugHpyGP_SC')
        Sep3Daily_Ot = predictionResultspyGPScaled(dfC6,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_SepHpyGP_SC')
        Okt3Daily_Ot = predictionResultspyGPScaled(dfC6,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_OktHpyGP_SC')
        Nov3Daily_Ot = predictionResultspyGPScaled(dfC6,'2013-11-01 00:00:00','2013-11-26 23:45:00','2013-11-29 00:00:00','2013-11-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_NovHpyGP_SC')
        Dec3Daily_Ot = predictionResultspyGPScaled(dfC6,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_DecHpyGP_SC')

        totalOt = returnAndSaveTotalListpyGP([Jan3Daily_Ot,Feb3Daily_Ot,Maa3Daily_Ot,Apr3Daily_Ot,Mei3Daily_Ot,Jun3Daily_Ot,Jul3Daily_Ot,Aug3Daily_Ot,Sep3Daily_Ot,Okt3Daily_Ot,Nov3Daily_Ot,Dec3Daily_Ot],'TotalOt_SC',False)


        #Conclusion = returnAndSaveTotalListpyGP([totalHP,totalOt],'conclusionConsumption',False)

def BaselineMethodResults():
        loaded4= False
        loaded6= False

        dfC4 = readConsumptionDataCSV(inputDataPath,'Measurements_HP')


        Jan3Daily_HP = predictionResultsBaseline(dfC4,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-27 00:00:00','2014-01-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Jan_H_BL')
        Feb3Daily_HP = predictionResultsBaseline(dfC4,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Feb_H_BL')
        Maa3Daily_HP = predictionResultsBaseline(dfC4,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Maa_H_BL')
        Apr3Daily_HP = predictionResultsBaseline(dfC4,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Apr_H_BL')
        Mei3Daily_HP = predictionResultsBaseline(dfC4,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Mei_H_BL')
        Jun3Daily_HP = predictionResultsBaseline(dfC4,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Jun_H_BL')
        Jul3Daily_HP = predictionResultsBaseline(dfC4,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Jul_H_BL')
        Aug3Daily_HP = predictionResultsBaseline(dfC4,'2014-08-01 00:00:00','2014-08-24 23:45:00','2014-08-27 00:00:00','2014-08-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Aug_H_BL')
        Sep3Daily_HP = predictionResultsBaseline(dfC4,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Sep_H_BL')
        Okt3Daily_HP = predictionResultsBaseline(dfC4,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Okt_H_BL')
        Nov3Daily_HP = predictionResultsBaseline(dfC4,'2013-11-01 00:00:00','2013-11-24 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Nov_H_BL')
        Dec3Daily_HP = predictionResultsBaseline(dfC4,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_Dec_H_BL')

        totalHP = returnAndSaveTotalListBaseline([Jan3Daily_HP,Feb3Daily_HP,Maa3Daily_HP,Apr3Daily_HP,Mei3Daily_HP,Jun3Daily_HP,Jul3Daily_HP,Aug3Daily_HP,Sep3Daily_HP,Okt3Daily_HP,Nov3Daily_HP,Dec3Daily_HP],'TotalHP_BL',False)


        dfC6 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other')

        Jan3Daily_Ot = predictionResultsBaseline(dfC6,'2014-01-01 00:00:00','2014-01-25 23:45:00','2014-01-26 00:00:00','2014-01-27 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Jan_H_BL')
        Feb3Daily_Ot = predictionResultsBaseline(dfC6,'2014-02-01 00:00:00','2014-02-25 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Feb_H_BL')
        Maa3Daily_Ot = predictionResultsBaseline(dfC6,'2014-03-01 00:00:00','2014-03-25 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Maa_H_BL')
        Apr3Daily_Ot = predictionResultsBaseline(dfC6,'2014-04-01 00:00:00','2014-04-25 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Apr_H_BL')
        Mei3Daily_Ot = predictionResultsBaseline(dfC6,'2014-05-01 00:00:00','2014-05-25 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Mei_H_BL')
        Jun3Daily_Ot = predictionResultsBaseline(dfC6,'2014-06-01 00:00:00','2014-06-25 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Jun_H_BL')
        Jul3Daily_Ot = predictionResultsBaseline(dfC6,'2014-07-01 00:00:00','2014-07-25 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Jul_H_BL')
        Aug3Daily_Ot = predictionResultsBaseline(dfC6,'2014-08-01 00:00:00','2014-08-26 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Aug_H_BL')
        Sep3Daily_Ot = predictionResultsBaseline(dfC6,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Sep_H_BL')
        Okt3Daily_Ot = predictionResultsBaseline(dfC6,'2013-10-01 00:00:00','2013-10-25 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Okt_H_BL')
        Nov3Daily_Ot = predictionResultsBaseline(dfC6,'2013-11-01 00:00:00','2013-11-26 23:45:00','2013-11-29 00:00:00','2013-11-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Nov_H_BL')
        Dec3Daily_Ot = predictionResultsBaseline(dfC6,'2013-12-01 00:00:00','2013-12-25 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_Dec_H_BL')

        totalOt = returnAndSaveTotalListBaseline([Jan3Daily_Ot,Feb3Daily_Ot,Maa3Daily_Ot,Apr3Daily_Ot,Mei3Daily_Ot,Jun3Daily_Ot,Jul3Daily_Ot,Aug3Daily_Ot,Sep3Daily_Ot,Okt3Daily_Ot,Nov3Daily_Ot,Dec3Daily_Ot],'Total_Ot_BL',False)


        dfC2 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other_HP')
        loaded2 = False
        Jan3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2014-01-01 00:00:00','2014-01-25 23:45:00','2014-01-27 00:00:00','2014-01-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JanH_baseline')
        #total_ot_HP = returnAndSaveTotalListpyGP([Jan3Daily_Ot_HP],'TotalOtHHpygP',False)

        Feb3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2014-02-01 00:00:00','2014-02-25 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_FebH_baseline')
        Maa3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2014-03-01 00:00:00','2014-03-25 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MaaH_baseline')
        Apr3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2014-04-01 00:00:00','2014-04-25 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AprH_baseline')
        Mei3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2014-05-01 00:00:00','2014-05-25 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MeiH_baseline')
        Jun3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2014-06-01 00:00:00','2014-06-25 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JunH_baseline')
        Jul3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2014-07-01 00:00:00','2014-07-25 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JulH_baseline')
        Aug3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2014-08-01 00:00:00','2014-08-27 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AugH_baseline')
        Sep3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2013-09-01 00:00:00','2013-09-27 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_SepH_baseline')
        Okt3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2013-10-01 00:00:00','2013-10-25 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_OktH_baseline')
        Nov3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2013-11-01 00:00:00','2013-11-25 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_NovH_baseline')
        Dec3Daily_Ot_HP = predictionResultsBaseline(dfC2,'2013-12-01 00:00:00','2013-12-25 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_DecH_baseline')


        total_ot_HP = returnAndSaveTotalListBaseline([Jan3Daily_Ot_HP,Feb3Daily_Ot_HP,Maa3Daily_Ot_HP,Apr3Daily_Ot_HP,Mei3Daily_Ot_HP,Jun3Daily_Ot_HP,Jul3Daily_Ot_HP,Aug3Daily_Ot_HP,Sep3Daily_Ot_HP,Okt3Daily_Ot_HP,Nov3Daily_Ot_HP,Dec3Daily_Ot_HP],'Total_Ot_HP_BL',False)


def SVRResults():
        start = time.time()
        loaded2 =False
        loaded4 = False
        loaded5 = False
        loaded6= False

        dfC2 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other_HP')
        Jan3Daily_Ot_HP = predictionResultsSVR(dfC2,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-27 00:00:00','2014-01-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JanH_SVR')
        Feb3Daily_Ot_HP = predictionResultsSVR(dfC2,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_FebH_SVR')
        Maa3Daily_Ot_HP = predictionResultsSVR(dfC2,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MaaH_SVR')
        Apr3Daily_Ot_HP = predictionResultsSVR(dfC2,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AprH_SVR')
        Mei3Daily_Ot_HP = predictionResultsSVR(dfC2,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MeiH_SVR')
        Jun3Daily_Ot_HP = predictionResultsSVR(dfC2,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JunH_SVR')
        Jul3Daily_Ot_HP = predictionResultsSVR(dfC2,'2014-07-01 00:00:00','2014-07-25 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JulH_SVR')
        Aug3Daily_Ot_HP = predictionResultsSVR(dfC2,'2014-08-01 00:00:00','2014-08-27 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AugH_SVR')
        Sep3Daily_Ot_HP = predictionResultsSVR(dfC2,'2013-09-01 00:00:00','2013-09-27 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_SepH_SVR')
        Okt3Daily_Ot_HP = predictionResultsSVR(dfC2,'2013-10-01 00:00:00','2013-10-25 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_OktH_SVR')
        Nov3Daily_Ot_HP = predictionResultsSVR(dfC2,'2013-11-01 00:00:00','2013-11-25 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_NovH_SVR')
        Dec3Daily_Ot_HP = predictionResultsSVR(dfC2,'2013-12-01 00:00:00','2013-12-25 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_DecH_SVR')


        total_ot_HP = returnAndSaveTotalListBaseline([Jan3Daily_Ot_HP,Feb3Daily_Ot_HP,Maa3Daily_Ot_HP,Apr3Daily_Ot_HP,Mei3Daily_Ot_HP,Jun3Daily_Ot_HP,Jul3Daily_Ot_HP,Aug3Daily_Ot_HP,Sep3Daily_Ot_HP,Okt3Daily_Ot_HP,Nov3Daily_Ot_HP,Dec3Daily_Ot_HP],'Total_Ot_HP_SVR',False)


        dfC6 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other')

        Jan3Daily_Ot = predictionResultsSVR(dfC6,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-30 00:00:00','2014-01-31 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JanH_SVR')
        Feb3Daily_Ot = predictionResultsSVR(dfC6,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_FebH_SVR')
        Maa3Daily_Ot = predictionResultsSVR(dfC6,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_MaaH_SVR')
        Apr3Daily_Ot = predictionResultsSVR(dfC6,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_AprH_SVR')
        Mei3Daily_Ot = predictionResultsSVR(dfC6,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_MeiH_SVR')
        Jun3Daily_Ot = predictionResultsSVR(dfC6,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JunH_SVR')
        Jul3Daily_Ot = predictionResultsSVR(dfC6,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JulH_SVR')
        Aug3Daily_Ot = predictionResultsSVR(dfC6,'2014-08-01 00:00:00','2014-08-26 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_AugH_SVR')
        Sep3Daily_Ot = predictionResultsSVR(dfC6,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_SepH_SVR')
        Okt3Daily_Ot = predictionResultsSVR(dfC6,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_OktH_SVR')
        Nov3Daily_Ot = predictionResultsSVR(dfC6,'2013-11-01 00:00:00','2013-11-26 23:45:00','2013-11-29 00:00:00','2013-11-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_NovH_SVR')
        Dec3Daily_Ot = predictionResultsSVR(dfC6,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_DecH_SVR')

        totalOt = returnAndSaveTotalListBaseline([Jan3Daily_Ot,Feb3Daily_Ot,Maa3Daily_Ot,Apr3Daily_Ot,Mei3Daily_Ot,Jun3Daily_Ot,Jul3Daily_Ot,Aug3Daily_Ot,Sep3Daily_Ot,Okt3Daily_Ot,Nov3Daily_Ot,Dec3Daily_Ot],'TotalOt_SVR',False)

        dfC4 = readConsumptionDataCSV(inputDataPath,'Measurements_HP')
        #resultDFMeteoSpringDaily_HP = predictionResultspyGP(dfC4,'2014-03-01 00:00:00','2014-05-28 23:45:00','2014-05-29 00:00:00','2014-05-31 23:45:00',2,'H',loaded4,False,'linear',True,'season_HP_Met_springDaily')


        Jan3Daily_HP = predictionResultsSVR(dfC4,'2014-01-01 00:00:00','2014-01-27 23:45:00','2014-01-30 00:00:00','2014-01-31 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JanH_SVR')
        Feb3Daily_HP = predictionResultsSVR(dfC4,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_FebH_SVR')
        Maa3Daily_HP = predictionResultsSVR(dfC4,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_MaaH_SVR')
        Apr3Daily_HP = predictionResultsSVR(dfC4,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_AprH_SVR')
        Mei3Daily_HP = predictionResultsSVR(dfC4,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_MeiH_SVR')
        Jun3Daily_HP = predictionResultsSVR(dfC4,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JunH_SVR')
        Jul3Daily_HP = predictionResultsSVR(dfC4,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JulH_SVR')
        Aug3Daily_HP = predictionResultsSVR(dfC4,'2014-08-01 00:00:00','2014-08-24 23:45:00','2014-08-27 00:00:00','2014-08-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_AugH_SVR')
        Sep3Daily_HP = predictionResultsSVR(dfC4,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_SepH_SVR')
        Okt3Daily_HP = predictionResultsSVR(dfC4,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_OktH_SVR')
        Nov3Daily_HP = predictionResultsSVR(dfC4,'2013-11-01 00:00:00','2013-11-24 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_NovH_SVR')
        Dec3Daily_HP = predictionResultsSVR(dfC4,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_DecH_SVR')

        totalHP = returnAndSaveTotalListBaseline([Jan3Daily_HP,Feb3Daily_HP,Maa3Daily_HP,Apr3Daily_HP,Mei3Daily_HP,Jun3Daily_HP,Jul3Daily_HP,Aug3Daily_HP,Sep3Daily_HP,Okt3Daily_HP,Nov3Daily_HP,Dec3Daily_HP],'TotalHP_SVR',False)


def OLSResults():
        start = time.time()
        loaded2 =False
        loaded4 = False
        loaded5 = False
        loaded6= False

        dfC2 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other_HP')
        Jan3Daily_Ot_HP = predictionResultsOLS(dfC2,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-27 00:00:00','2014-01-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JanH_OLS')
        Feb3Daily_Ot_HP = predictionResultsOLS(dfC2,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_FebH_OLS')
        Maa3Daily_Ot_HP = predictionResultsOLS(dfC2,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MaaH_OLS')
        Apr3Daily_Ot_HP = predictionResultsOLS(dfC2,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AprH_OLS')
        Mei3Daily_Ot_HP = predictionResultsOLS(dfC2,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MeiH_OLS')
        Jun3Daily_Ot_HP = predictionResultsOLS(dfC2,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JunH_OLS')
        Jul3Daily_Ot_HP = predictionResultsOLS(dfC2,'2014-07-01 00:00:00','2014-07-25 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JulH_OLS')
        Aug3Daily_Ot_HP = predictionResultsOLS(dfC2,'2014-08-01 00:00:00','2014-08-27 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AugH_OLS')
        Sep3Daily_Ot_HP = predictionResultsOLS(dfC2,'2013-09-01 00:00:00','2013-09-27 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_SepH_OLS')
        Okt3Daily_Ot_HP = predictionResultsOLS(dfC2,'2013-10-01 00:00:00','2013-10-25 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_OktH_OLS')
        Nov3Daily_Ot_HP = predictionResultsOLS(dfC2,'2013-11-01 00:00:00','2013-11-25 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_NovH_OLS')
        Dec3Daily_Ot_HP = predictionResultsOLS(dfC2,'2013-12-01 00:00:00','2013-12-25 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_DecH_OLS')


        total_ot_HP = returnAndSaveTotalListBaseline([Jan3Daily_Ot_HP,Feb3Daily_Ot_HP,Maa3Daily_Ot_HP,Apr3Daily_Ot_HP,Mei3Daily_Ot_HP,Jun3Daily_Ot_HP,Jul3Daily_Ot_HP,Aug3Daily_Ot_HP,Sep3Daily_Ot_HP,Okt3Daily_Ot_HP,Nov3Daily_Ot_HP,Dec3Daily_Ot_HP],'Total_Ot_HP_OLS',False)


        dfC6 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other')

        Jan3Daily_Ot = predictionResultsOLS(dfC6,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-30 00:00:00','2014-01-31 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JanH_OLS')
        Feb3Daily_Ot = predictionResultsOLS(dfC6,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_FebH_OLS')
        Maa3Daily_Ot = predictionResultsOLS(dfC6,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_MaaH_OLS')
        Apr3Daily_Ot = predictionResultsOLS(dfC6,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_AprH_OLS')
        Mei3Daily_Ot = predictionResultsOLS(dfC6,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_MeiH_OLS')
        Jun3Daily_Ot = predictionResultsOLS(dfC6,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JunH_OLS')
        Jul3Daily_Ot = predictionResultsOLS(dfC6,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JulH_OLS')
        Aug3Daily_Ot = predictionResultsOLS(dfC6,'2014-08-01 00:00:00','2014-08-26 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_AugH_OLS')
        Sep3Daily_Ot = predictionResultsOLS(dfC6,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_SepH_OLS')
        Okt3Daily_Ot = predictionResultsOLS(dfC6,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_OktH_OLS')
        Nov3Daily_Ot = predictionResultsOLS(dfC6,'2013-11-01 00:00:00','2013-11-26 23:45:00','2013-11-29 00:00:00','2013-11-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_NovH_OLS')
        Dec3Daily_Ot = predictionResultsOLS(dfC6,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_DecH_OLS')

        totalOt = returnAndSaveTotalListBaseline([Jan3Daily_Ot,Feb3Daily_Ot,Maa3Daily_Ot,Apr3Daily_Ot,Mei3Daily_Ot,Jun3Daily_Ot,Jul3Daily_Ot,Aug3Daily_Ot,Sep3Daily_Ot,Okt3Daily_Ot,Nov3Daily_Ot,Dec3Daily_Ot],'TotalOt_OLS',False)

        dfC4 = readConsumptionDataCSV(inputDataPath,'Measurements_HP')
        #resultDFMeteoSpringDaily_HP = predictionResultspyGP(dfC4,'2014-03-01 00:00:00','2014-05-28 23:45:00','2014-05-29 00:00:00','2014-05-31 23:45:00',2,'H',loaded4,False,'linear',True,'season_HP_Met_springDaily')


        Jan3Daily_HP = predictionResultsOLS(dfC4,'2014-01-01 00:00:00','2014-01-27 23:45:00','2014-01-30 00:00:00','2014-01-31 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JanH_OLS')
        Feb3Daily_HP = predictionResultsOLS(dfC4,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_FebH_OLS')
        Maa3Daily_HP = predictionResultsOLS(dfC4,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_MaaH_OLS')
        Apr3Daily_HP = predictionResultsOLS(dfC4,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_AprH_OLS')
        Mei3Daily_HP = predictionResultsOLS(dfC4,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_MeiH_OLS')
        Jun3Daily_HP = predictionResultsOLS(dfC4,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JunH_OLS')
        Jul3Daily_HP = predictionResultsOLS(dfC4,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JulH_OLS')
        Aug3Daily_HP = predictionResultsOLS(dfC4,'2014-08-01 00:00:00','2014-08-24 23:45:00','2014-08-27 00:00:00','2014-08-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_AugH_OLS')
        Sep3Daily_HP = predictionResultsOLS(dfC4,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_SepH_OLS')
        Okt3Daily_HP = predictionResultsOLS(dfC4,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_OktH_OLS')
        Nov3Daily_HP = predictionResultsOLS(dfC4,'2013-11-01 00:00:00','2013-11-24 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_NovH_OLS')
        Dec3Daily_HP = predictionResultsOLS(dfC4,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_DecH_OLS')

        totalHP = returnAndSaveTotalListBaseline([Jan3Daily_HP,Feb3Daily_HP,Maa3Daily_HP,Apr3Daily_HP,Mei3Daily_HP,Jun3Daily_HP,Jul3Daily_HP,Aug3Daily_HP,Sep3Daily_HP,Okt3Daily_HP,Nov3Daily_HP,Dec3Daily_HP],'TotalHP_OLS',False)



def sklearnGPResults():
        global theta
        loaded2 =False
        loaded4 = False
        loaded5 = False
        loaded6= False


        dfC2 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other_HP')
        Jan3Daily_Ot_HP = predictionResults(dfC2,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-27 00:00:00','2014-01-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JanH_SKL')
        Feb3Daily_Ot_HP = predictionResults(dfC2,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_FebH_SKL')
        Maa3Daily_Ot_HP = predictionResults(dfC2,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MaaH_SKL')
        Apr3Daily_Ot_HP = predictionResults(dfC2,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AprH_SKL')
        Mei3Daily_Ot_HP = predictionResults(dfC2,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_MeiH_SKL')
        Jun3Daily_Ot_HP = predictionResults(dfC2,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JunH_SKL')
        Jul3Daily_Ot_HP = predictionResults(dfC2,'2014-07-01 00:00:00','2014-07-25 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_JulH_SKL')
        Aug3Daily_Ot_HP = predictionResults(dfC2,'2014-08-01 00:00:00','2014-08-27 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_AugH_SKL')
        Sep3Daily_Ot_HP = predictionResults(dfC2,'2013-09-01 00:00:00','2013-09-27 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_SepH_SKL')
        Okt3Daily_Ot_HP = predictionResults(dfC2,'2013-10-01 00:00:00','2013-10-25 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_OktH_SKL')
        Nov3Daily_Ot_HP = predictionResults(dfC2,'2013-11-01 00:00:00','2013-11-25 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_NovH_SKL')
        Dec3Daily_Ot_HP = predictionResults(dfC2,'2013-12-01 00:00:00','2013-12-25 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded2,False,'linear',False,'Monthly_Ot_HP_DecH_SKL')


        total_ot_HP = returnAndSaveTotalList([Jan3Daily_Ot_HP,Feb3Daily_Ot_HP,Maa3Daily_Ot_HP,Apr3Daily_Ot_HP,Mei3Daily_Ot_HP,Jun3Daily_Ot_HP,Jul3Daily_Ot_HP,Aug3Daily_Ot_HP,Sep3Daily_Ot_HP,Okt3Daily_Ot_HP,Nov3Daily_Ot_HP,Dec3Daily_Ot_HP],'Total_Ot_HP_SKL',False)


        dfC6 = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other')

        Jan3Daily_Ot = predictionResults(dfC6,'2014-01-01 00:00:00','2014-01-24 23:45:00','2014-01-30 00:00:00','2014-01-31 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JanHpyGP_SKL')
        Feb3Daily_Ot = predictionResults(dfC6,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_FebHpyGP_SKL')
        Maa3Daily_Ot = predictionResults(dfC6,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_MaaHpyGP_SKL')
        Apr3Daily_Ot = predictionResults(dfC6,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_AprHpyGP_SKL')
        Mei3Daily_Ot = predictionResults(dfC6,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_MeiHpyGP_SKL')
        Jun3Daily_Ot = predictionResults(dfC6,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JunHpyGP_SKL')
        Jul3Daily_Ot = predictionResults(dfC6,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_JulHpyGP_SKL')
        Aug3Daily_Ot = predictionResults(dfC6,'2014-08-01 00:00:00','2014-08-26 23:45:00','2014-08-29 00:00:00','2014-08-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_AugHpyGP_SKL')
        Sep3Daily_Ot = predictionResults(dfC6,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_SepHpyGP_SKL')
        Okt3Daily_Ot = predictionResults(dfC6,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_OktHpyGP_SKL')
        Nov3Daily_Ot = predictionResults(dfC6,'2013-11-01 00:00:00','2013-11-26 23:45:00','2013-11-29 00:00:00','2013-11-30 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_NovHpyGP_SKL')
        Dec3Daily_Ot = predictionResults(dfC6,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded6,False,'linear',False,'Monthly_Ot_DecHpyGP_SKL')

        totalOt = returnAndSaveTotalList([Jan3Daily_Ot,Feb3Daily_Ot,Maa3Daily_Ot,Apr3Daily_Ot,Mei3Daily_Ot,Jun3Daily_Ot,Jul3Daily_Ot,Aug3Daily_Ot,Sep3Daily_Ot,Okt3Daily_Ot,Nov3Daily_Ot,Dec3Daily_Ot],'TotalOt_SKL',False)

        dfC4 = readConsumptionDataCSV(inputDataPath,'Measurements_HP')
        #resultDFMeteoSpringDaily_HP = predictionResultspyGP(dfC4,'2014-03-01 00:00:00','2014-05-28 23:45:00','2014-05-29 00:00:00','2014-05-31 23:45:00',2,'H',loaded4,False,'linear',True,'season_HP_Met_springDaily')


        Jan3Daily_HP = predictionResults(dfC4,'2014-01-01 00:00:00','2014-01-27 23:45:00','2014-01-30 00:00:00','2014-01-31 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JanHpyGP_SKL')
        Feb3Daily_HP = predictionResults(dfC4,'2014-02-01 00:00:00','2014-02-24 23:45:00','2014-02-27 00:00:00','2014-02-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_FebHpyGP_SKL')
        Maa3Daily_HP = predictionResults(dfC4,'2014-03-01 00:00:00','2014-03-24 23:45:00','2014-03-27 00:00:00','2014-03-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_MaaHpyGP_SKL')
        Apr3Daily_HP = predictionResults(dfC4,'2014-04-01 00:00:00','2014-04-24 23:45:00','2014-04-27 00:00:00','2014-04-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_AprHpyGP_SKL')
        Mei3Daily_HP = predictionResults(dfC4,'2014-05-01 00:00:00','2014-05-24 23:45:00','2014-05-27 00:00:00','2014-05-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_MeiHpyGP_SKL')
        Jun3Daily_HP = predictionResults(dfC4,'2014-06-01 00:00:00','2014-06-24 23:45:00','2014-06-27 00:00:00','2014-06-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JunHpyGP_SKL')
        Jul3Daily_HP = predictionResults(dfC4,'2014-07-01 00:00:00','2014-07-24 23:45:00','2014-07-27 00:00:00','2014-07-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_JulHpyGP_SKL')
        Aug3Daily_HP = predictionResults(dfC4,'2014-08-01 00:00:00','2014-08-24 23:45:00','2014-08-27 00:00:00','2014-08-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_AugHpyGP_SKL')
        Sep3Daily_HP = predictionResults(dfC4,'2013-09-01 00:00:00','2013-09-26 23:45:00','2013-09-29 00:00:00','2013-09-30 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_SepHpyGP_SKL')
        Okt3Daily_HP = predictionResults(dfC4,'2013-10-01 00:00:00','2013-10-24 23:45:00','2013-10-27 00:00:00','2013-10-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_OktHpyGP_SKL')
        Nov3Daily_HP = predictionResults(dfC4,'2013-11-01 00:00:00','2013-11-24 23:45:00','2013-11-27 00:00:00','2013-11-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_NovHpyGP_SKL')
        Dec3Daily_HP = predictionResults(dfC4,'2013-12-01 00:00:00','2013-12-24 23:45:00','2013-12-27 00:00:00','2013-12-28 23:45:00',70,'H',loaded4,False,'linear',False,'Monthly_HP_DecHpyGP_SKL')

        totalHP = returnAndSaveTotalList([Jan3Daily_HP,Feb3Daily_HP,Maa3Daily_HP,Apr3Daily_HP,Mei3Daily_HP,Jun3Daily_HP,Jul3Daily_HP,Aug3Daily_HP,Sep3Daily_HP,Okt3Daily_HP,Nov3Daily_HP,Dec3Daily_HP],'TotalHP_SKL',False)



if __name__ == "__main__":
        #SVRResults()
        #BaselineMethodResults()
        #SVRResults()
        #testTotalYearMonthlyDaily2DayspyGP()
        sklearnGPResults()
        #OLSResults()




        listOtHP = ['Monthly_Ot_HP_SepHpyGP','Monthly_Ot_HP_OktHpyGP','Monthly_Ot_HP_NovHpyGP','Monthly_Ot_HP_DecHpyGP','Monthly_Ot_HP_JanHpyGP','Monthly_Ot_HP_FebHpyGP','Monthly_Ot_HP_MaaHpyGP','Monthly_Ot_HP_AprHpyGP','Monthly_Ot_HP_MeiHpyGP','Monthly_Ot_HP_JunHpyGP','Monthly_Ot_HP_JulHpyGP','Monthly_Ot_HP_AugHpyGP']
        #testTotalYearMonthlyDaily2DayspyGP()
        listOtpyGP = ['Monthly_Ot_SepHpyGP','Monthly_Ot_OktHpyGP','Monthly_Ot_NovHpyGP','Monthly_Ot_DecHpyGP','Monthly_Ot_JanHpyGP','Monthly_Ot_FebHpyGP','Monthly_Ot_MaaHpyGP','Monthly_Ot_AprHpyGP','Monthly_Ot_MeiHpyGP','Monthly_Ot_JunHpyGP','Monthly_Ot_JulHpyGP','Monthly_Ot_AugHpyGP']
        listHPpyGP = ['Monthly_HP_SepHpyGP','Monthly_HP_OktHpygP','Monthly_HP_NovHpygP','Monthly_HP_DecHpygP','Monthly_HP_JanHpygP','Monthly_HP_FebHpygP','Monthly_HP_MaaHpygP','Monthly_HP_AprHpygP','Monthly_HP_MeiHpygP','Monthly_HP_JunHpygP','Monthly_HP_JulHpygP','Monthly_HP_AugHpygP']

        listHP_BL =['Monthly_HP_Sep_H_BL','Monthly_HP_Okt_H_BL','Monthly_HP_Nov_H_BL','Monthly_HP_Dec_H_BL','Monthly_HP_Jan_H_BL','Monthly_HP_Feb_H_BL','Monthly_HP_Maa_H_BL','Monthly_HP_Apr_H_BL','Monthly_HP_Mei_H_BL','Monthly_HP_Jun_H_BL','Monthly_HP_Jul_H_BL','Monthly_HP_Aug_H_BL']
        listOT_BL =['Monthly_Ot_Sep_H_BL','Monthly_Ot_Okt_H_BL','Monthly_Ot_Nov_H_BL','Monthly_Ot_Dec_H_BL','Monthly_Ot_Jan_H_BL','Monthly_Ot_Feb_H_BL','Monthly_Ot_Maa_H_BL','Monthly_Ot_Apr_H_BL','Monthly_Ot_Mei_H_BL','Monthly_Ot_Jun_H_BL','Monthly_Ot_Jul_H_BL','Monthly_Ot_Aug_H_BL']
        listOT_HP_BL =['Monthly_Ot_HP_SepH_baseline','Monthly_Ot_HP_OktH_baseline','Monthly_Ot_HP_NovH_baseline','Monthly_Ot_HP_DecH_baseline','Monthly_Ot_HP_JanH_baseline','Monthly_Ot_HP_FebH_baseline','Monthly_Ot_HP_MaaH_baseline','Monthly_Ot_HP_AprH_baseline','Monthly_Ot_HP_MeiH_baseline','Monthly_Ot_HP_JunH_baseline','Monthly_Ot_HP_JulH_baseline','Monthly_Ot_HP_AugH_baseline']

        listOT_SVR = ['Monthly_Ot_SepH_SVR','Monthly_Ot_OktH_SVR','Monthly_Ot_NovH_SVR','Monthly_Ot_DecH_SVR','Monthly_Ot_JanH_SVR','Monthly_Ot_FebH_SVR','Monthly_Ot_MaaH_SVR','Monthly_Ot_AprH_SVR','Monthly_Ot_MeiH_SVR','Monthly_Ot_JunH_SVR','Monthly_Ot_JulH_SVR','Monthly_Ot_AugH_SVR']
        listHP_SVR = ['Monthly_HP_SepH_SVR','Monthly_HP_OktH_SVR','Monthly_HP_NovH_SVR','Monthly_HP_DecH_SVR','Monthly_HP_JanH_SVR','Monthly_HP_FebH_SVR','Monthly_HP_MaaH_SVR','Monthly_HP_AprH_SVR','Monthly_HP_MeiH_SVR','Monthly_HP_JunH_SVR','Monthly_HP_JulH_SVR','Monthly_HP_AugH_SVR']
        listOT_HP_SVR =['Monthly_Ot_HP_SepH_SVR','Monthly_Ot_HP_OktH_SVR','Monthly_Ot_HP_NovH_SVR','Monthly_Ot_HP_DecH_SVR','Monthly_Ot_HP_JanH_SVR','Monthly_Ot_HP_FebH_SVR','Monthly_Ot_HP_MaaH_SVR','Monthly_Ot_HP_AprH_SVR','Monthly_Ot_HP_MeiH_SVR','Monthly_Ot_HP_JunH_SVR','Monthly_Ot_HP_JulH_SVR','Monthly_Ot_HP_AugH_SVR']

        listOT_SKL = ['Monthly_Ot_SepHpyGP_SKL','Monthly_Ot_OktHpyGP_SKL','Monthly_Ot_NovHpyGP_SKL','Monthly_Ot_DecHpyGP_SKL','Monthly_Ot_JanHpyGP_SKL','Monthly_Ot_FebHpyGP_SKL','Monthly_Ot_MaaHpyGP_SKL','Monthly_Ot_AprHpyGP_SKL','Monthly_Ot_MeiHpyGP_SKL','Monthly_Ot_JunHpyGP_SKL','Monthly_Ot_JulHpyGP_SKL','Monthly_Ot_AugHpyGP_SKL']
        listHP_SKL = ['Monthly_HP_SepHpyGP_SKL','Monthly_HP_OktHpyGP_SKL','Monthly_HP_NovHpyGP_SKL','Monthly_HP_DecHpyGP_SKL','Monthly_HP_JanHpyGP_SKL','Monthly_HP_FebHpyGP_SKL','Monthly_HP_MaaHpyGP_SKL','Monthly_HP_AprHpyGP_SKL','Monthly_HP_MeiHpyGP_SKL','Monthly_HP_JunHpyGP_SKL','Monthly_HP_JulHpyGP_SKL','Monthly_HP_AugHpyGP_SKL']
        listOT_HP_SKL = ['Monthly_Ot_HP_SepH_SKL','Monthly_Ot_HP_OktH_SKL','Monthly_Ot_HP_NovH_SKL','Monthly_Ot_HP_DecH_SKL','Monthly_Ot_HP_JanH_SKL','Monthly_Ot_HP_FebH_SKL','Monthly_Ot_HP_MaaH_SKL','Monthly_Ot_HP_AprH_SKL','Monthly_Ot_HP_MeiH_SKL','Monthly_Ot_HP_JunH_SKL','Monthly_Ot_HP_JulH_SKL','Monthly_Ot_HP_AugH_SKL']


        listOT_OLS = ['Monthly_Ot_SepH_OLS','Monthly_Ot_OktH_OLS','Monthly_Ot_NovH_OLS','Monthly_Ot_DecH_OLS','Monthly_Ot_JanH_OLS','Monthly_Ot_FebH_OLS','Monthly_Ot_MaaH_OLS','Monthly_Ot_AprH_OLS','Monthly_Ot_MeiH_OLS','Monthly_Ot_JunH_OLS','Monthly_Ot_JulH_OLS','Monthly_Ot_AugH_OLS']
        listHP_OLS = ['Monthly_HP_SepH_OLS','Monthly_HP_OktH_OLS','Monthly_HP_NovH_OLS','Monthly_HP_DecH_OLS','Monthly_HP_JanH_OLS','Monthly_HP_FebH_OLS','Monthly_HP_MaaH_OLS','Monthly_HP_AprH_OLS','Monthly_HP_MeiH_OLS','Monthly_HP_JunH_OLS','Monthly_HP_JulH_OLS','Monthly_HP_AugH_OLS']
        listOT_HP_OLS = ['Monthly_Ot_HP_SepH_OLS','Monthly_Ot_HP_OktH_OLS','Monthly_Ot_HP_NovH_OLS','Monthly_Ot_HP_DecH_OLS','Monthly_Ot_HP_JanH_OLS','Monthly_Ot_HP_FebH_OLS','Monthly_Ot_HP_MaaH_OLS','Monthly_Ot_HP_AprH_OLS','Monthly_Ot_HP_MeiH_OLS','Monthly_Ot_HP_JunH_OLS','Monthly_Ot_HP_JulH_OLS','Monthly_Ot_HP_AugH_OLS']



        #plotEngine.plotErrorPerMonth('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/resultsHpOt/',listOtHP,1)
        #plotEngine.plotErrorPerHH('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/resultsHpOt/',listOtHP)
        #plotEngine.plotErrorPerHH('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/resultsHp/',listHP)
        #plotEngine.plotErrorPerHH('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/resultsOt/',listOt)


        #seasonalPredictionResults2Days()
        #testTotalYearMonthlyDaily2DayspyGP()
        #testTotalYearMonthlyDaily2DayspyGP()
        #testTotalYearMonthlyDaily2DayspyGP()
        #testScaled()
        #testTotalYearMonthlyDaily2DayspyGP()
        #BaselineMethodResults()

        #testTotalYearMonthlyDaily2DayspyGP()
        #testScaled()
        '''
        plotEngine.plotErrorPerHH('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/BL results19032016/7dagen/totalHP/',listHP_BL)
        plotEngine.plotErrorPerMonth('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/BL results19032016/7dagen/totalHP/',listHP_BL,1)

        plotEngine.plotErrorPerHH('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/BL results19032016/7dagen/totalOt/',listOT_BL)
        plotEngine.plotErrorPerMonth('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/BL results19032016/7dagen/totalOt/',listOT_BL,1)

        plotEngine.plotErrorPerHH('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/BL results19032016/7dagen/totalHPot/',listOT_HP_BL)
        plotEngine.plotErrorPerMonth('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/BL results19032016/7dagen/totalHPot/',listOT_HP_BL,1)
        '''
        '''
        plotEngine.plotErrorPerHH('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/GPR results19032016/totalHP/',listHPpyGP)
        plotEngine.plotErrorPerMonth('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/GPR results19032016/totalHP/',listHPpyGP,1)

        plotEngine.plotErrorPerHH('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/GPR results19032016/totalOt/',listOtpyGP)
        plotEngine.plotErrorPerMonth('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/GPR results19032016/totalOt/',listOtpyGP,1)

        plotEngine.plotErrorPerHH('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/GPR results19032016/totalHPot/',listOtHP)
        plotEngine.plotErrorPerMonth('/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/GPR results19032016/totalHPot/',listOtHP,1)
        '''






        '''Gebruikt
        dataFramesLocation = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/OldResults/BL results19032016/7dagen/totalOt/'
        fileNames=listOT_BL
        dataFramesLocation2 = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/pygp/rbf+lin with recency/'
        fileNames2 = listOtpyGP
        dataFramesLocation3 = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/OldResults/GPR2Rec results19032016/totalOt/'
        fileNames3 = listOtpyGP

        plotEngine.plotErrorPerMonth3(dataFramesLocation,fileNames,dataFramesLocation2,fileNames2,dataFramesLocation3,fileNames3)
        plotEngine.plotErrorPerHH3(dataFramesLocation,fileNames,dataFramesLocation2,fileNames2,dataFramesLocation3,fileNames3)
        '''




        '''
        dataFramesLocation = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/BL results19032016/7dagen/totalHP/'
        fileNames=listHP_BL
        dataFramesLocation2 = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/SVR_lin_kernel/'
        fileNames2 = listHP_SVR
        dataFramesLocation3 = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/GPR2Rec results19032016/totalHP/'
        fileNames3 = listHPpyGP


        plotEngine.plotErrorPerMonth3(dataFramesLocation,fileNames,dataFramesLocation2,fileNames2,dataFramesLocation3,fileNames3)
        plotEngine.plotErrorPerHH3(dataFramesLocation,fileNames,dataFramesLocation2,fileNames2,dataFramesLocation3,fileNames3)


        dataFramesLocation = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/BL results19032016/7dagen/totalHPot/'
        fileNames=listOT_HP_BL
        #dataFramesLocation2 = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/GPR results19032016/totalHPot/'
        #fileNames2 = listOtHP
        dataFramesLocation2 = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/SVR_lin_kernel/'
        fileNames2 = listOT_HP_SVR
        dataFramesLocation3 = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/GPR2Rec results19032016/totalHPot/'
        fileNames3 = listOtHP




        plotEngine.plotErrorPerMonth3(dataFramesLocation,fileNames,dataFramesLocation2,fileNames2,dataFramesLocation3,fileNames3)
        plotEngine.plotErrorPerHH3(dataFramesLocation,fileNames,dataFramesLocation2,fileNames2,dataFramesLocation3,fileNames3)
        '''

        #with recency:
        dataFramesLocation = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/baseline methode resulaten/juiste -7/'
        fileNames=listOT_BL
        dataFramesLocation2 ='/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/pygp/lin + rbf kernel without receny new/'
        fileNames2 = listOtpyGP
        dataFramesLocation3 = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/SVR/lin kernel no recency/'
        fileNames3 = listOT_SVR
        dataFramesLocation4 = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/sklearnGP/lin kernel no recency new/'
        fileNames4 = listOT_SKL
        dataFramesLocation5 = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/OLS/without recency/'
        fileNames5 = listOT_OLS

        #plotEngine.plotErrorPerMonth3(dataFramesLocation,fileNames,dataFramesLocation2,fileNames2,dataFramesLocation3,fileNames3)
        plotEngine.plotErrorPerHH_List([dataFramesLocation,dataFramesLocation3,dataFramesLocation4,dataFramesLocation5],[fileNames,fileNames3,fileNames4,fileNames5],['Baseline','SVR','SKL_GP','OLS'],'All_rec_HH')
        #plotEngine.plotErrorPerMonth_List([dataFramesLocation,dataFramesLocation3,dataFramesLocation4,dataFramesLocation5],[fileNames,fileNames3,fileNames4,fileNames5],['Baseline','SVR','SKL_GP','OLS'],'All_rec_month')



