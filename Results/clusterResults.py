__author__ = 'christiaanleysen'
import pandas as pd
import features.featureMaker as fm
import Results.plotEngine as plotEng
import Methods.GaussianprocessClustering as GPC
import Results.tree as TREE
import timeit

resultsLocation = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/'

inputDataPath = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/'
fileNameIrr = 'Data_SolarIrradiation'
fileNameTemp = 'Temperature_ambient'



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
dfC = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other_HP')

def clusterResults(): #DEPRECATED: use "clusterResultsMultipleHHDates()" instead
    featureSet = []
    numberOfHH = 71
    for i in range(0,numberOfHH,1):
        features=fm.makeFeatureVectorAllSelectedHousehold(dfC,dfTemp,dfIrr,'2014-02-01 00:00:00','2014-02-07 23:45:00','H','HH'+str(i),'dateFrameTrainingPyGP1')
        featureSet.append(features)
        #vectorX.append(features.values[:, 1:20])
        #vectorY.append(features.values[:, 0])


    initialClusterlist = []
    for i in range(0,numberOfHH,1):
        initialClusterlist.append((i,None))

    #totallistRMSE = GPC.calculateRMSEPyGP(vectorX,vectorY,intialClusterlist)
    print('initialList',initialClusterlist)

    #clusterResults,resultTree = GPC.clusterMain(featureSet,intialClusterlist,6.53,6)
    threshold =0.996
    minimumClusterSize = 1
    splitRatio = 0.2

    clusterResults,resultTree = GPC.clusterMain(featureSet,initialClusterlist,threshold,minimumClusterSize,splitRatio)

    tree = plotEng.makeTree(range(0,numberOfHH+1,1),resultTree)
    newick = []
    print(tree)
    TREE.convertTreeAux(tree, newick)
    convertedTree = (''.join(newick)+';')
    plotEng.plotTree(convertedTree,featureSet,threshold,minimumClusterSize,splitRatio)


def clusterResultsMultipleHHDates():
    featureSet = []
    numberOfHH = 10
    #HHDates=[('2014-02-01 00:00:00','2014-02-07 23:45:00'),('2014-05-03 00:00:00','2014-05-09 23:45:00'),('2014-08-02 00:00:00','2014-08-08 23:45:00'),('2013-11-02 00:00:00','2013-11-08 23:45:00')]
    #HHDates=[('2014-01-04 00:00:00','2014-01-10 23:45:00')]#,('2014-01-18 00:00:00','2014-01-24 23:45:00'),('2014-01-25 00:00:00','2014-01-31 23:45:00')]
    #HHDates=[('2014-02-01 00:00:00','2014-02-07 23:45:00'),('2014-02-08 00:00:00','2014-02-14 23:45:00'),('2014-02-15 00:00:00','2014-02-21 23:45:00'),('2014-02-22 00:00:00','2014-02-28 23:45:00')]
    #HHDates = [('2014-03-01 08:00:00','2014-03-01 16:45:00')]
    HHDates=[('2014-07-12 00:00:00','2014-07-18 23:45:00')]
    for i in range(0,numberOfHH,1):
        j=1
        for date in HHDates:
            features=fm.makeClusterFeatureVectorAllSelectedHousehold(dfC,dfTemp,dfIrr,date[0],date[1],'H','HH'+str(i),'dateFrameTrainingPyGPCluster') #dateFrameTrainingPyGP1
            if (len(HHDates) == 1):
                features = features.rename(columns={'HH'+str(i): 'HH'+str(i+1)})
                featureSet.append(('HH'+str(i+1),features))
            else:
                features = features.rename(columns={'HH'+str(i): 'HH'+str(i+1)+'.'+str(j)})
                featureSet.append(('HH'+str(i+1)+'.'+str(j),features))

            j+=1

    initialClusterlist = []
    for i in range(0,numberOfHH,1):
        for j in range(1,len(HHDates)+1,1):
            if (len(HHDates) == 1):
                initialClusterlist.append((str(i+1),None))
            else:
                initialClusterlist.append((str(i+1)+'.'+str(j),None))

    print('lenfeatureSet',len(featureSet))
    print('initialClusterlist',initialClusterlist)


    #cluster parameters
    threshold =0.995#0.989#0.995#0.991
    minimumClusterSize = 1
    splitRatio =0.25#0.2 #0.35#0.35



    start = timeit.default_timer()
    clusterResults,resultTree = GPC.clusterMain(featureSet,initialClusterlist,threshold,minimumClusterSize,splitRatio)
    print('clustertree',resultTree)
    stop = timeit.default_timer()
    print('calculation time: ', stop - start) #196.59 s

    HHlist = [item[0] for item in initialClusterlist]
    tree = plotEng.makeTree(HHlist,resultTree)


    newick = []
    print(tree)
    TREE.convertTreeAux(tree, newick)


    convertedTree = (''.join(newick)+';')
    print(convertedTree)

    plotEng.plotTree(convertedTree,featureSet,threshold,minimumClusterSize,splitRatio)
    return clusterResults




if __name__ == "__main__":
    clusterResults = clusterResultsMultipleHHDates()
    #10, 30.92974090576172; 29.4299578666687
    #20, 51.856143951416016; 48.55599093437195
    #30, 91.647784948349
    #50, 132.19911098480225; 120.74244213104248; 111.93636202812195
    #71, 199.85740613937378;
    #100, 322.8784739971161; 273.7597019672394
    #120, 293.92700004577637; 307.9657869338989
    #140, 397.9198110103607

