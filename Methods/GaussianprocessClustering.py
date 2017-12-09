__author__ = 'christiaanleysen'
import features.featureMaker as fm
import pandas as pd
import matplotlib.pyplot as plt
import pyGPs
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import mean_squared_error
import Results.plotEngine as plotEng
import Methods.GaussianprocessClusteringPyGP as GPCP
import math
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
dfC = readConsumptionDataCSV(inputDataPath,'Measurements_Elec_other_HP')


def plotGaussianProcess(testSetY, predictedSetY, sigma,title,min,max):
    """
    makes plot of the gaussian process output
    Parameters:
    -----------

    testSetY: test value set
    predictedSetY: predicted value set
    sigma: variance of the prediction
    title: string with name of plot
    min: minimum value of the Y-axis
    max: maximum value of the Y-axis

    Returns:
    --------
    a plot of the predicted values and their variance and the observed values
    """


    fig = plt.figure(figsize=(20, 6))

    plt.ylabel('Consumptie (kWh)', fontsize=13)
    plt.title('Gaussian Process Regressie: '+str(title))
    plt.legend(loc='upper right')
    plt.xlim([0, len(testSetY)])
    plt.ylim([min, max])

    #xTickLabels = pd.DataFrame(predictedSetY.index[np.arange(0,len(predictedSetY),1)])
    #xTickLabels['date'] = xTickLabels['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    xTickLabels = np.arange(0,len(predictedSetY),1)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, len(predictedSetY), 1))
    ax.set_xticklabels(labels=xTickLabels, fontsize=9, rotation=90)

    plt.plot(predictedSetY, 'b-', label=u'Voorspelling',color='green')
    plt.plot(testSetY, 'r.', markersize=10, label=u'Observaties',color='red')
    x = range(len(testSetY))
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([predictedSetY - 1.9600 * sigma, (predictedSetY + 1.9600 * sigma)[::-1]]),
             alpha=.3, fc='b', ec='None', label='95% betrouwbaarheidsinterval',facecolor='green')

def getHouseholdsXandYs (featuresSet,listHHNr,normalize=True):
    """
    find the X=featureset and Y=valueset of the given households
    Parameters:
    -----------

    featureset: entire featur and value eset of all households (form: (Y,x1,...,xn) with Y = value and x are features)
    listHHNr: list of string repr. of the houselholds form: 'HH1'
    Returns:
    --------
    a plot of the predicted values and their variance and the observed values
    """

    vectorX=[]
    vectorY=[]
    for (i,rmse) in listHHNr:
        HHTuples = dict(featuresSet)

        vectorX.append(HHTuples['HH'+i].values[:,1:20])
        vectorY.append(HHTuples['HH'+i].values[:,0])
    if(normalize):
        vectorX = [preprocessing.scale(element )for element in vectorX]
        vectorY=preprocessing.scale(vectorY,axis=1)
    return vectorX,vectorY

def calculateRMSE(vectorX,vectorY,clusterList):
    """
    calculate the root mean squared error
    Parameters:
    -----------

    vectorX: featureSet
    vectorY: valueSet
    clusterlist: cluster of households
    Returns:
    --------
    list of (household,rmse) tuples
    """

    setX = [preprocessing.scale(element )for element in vectorX]
    setY=preprocessing.scale(vectorY,axis=1)#np.asarray(vectorY)
    model = pyGPs.GPR()      # specify model (GP regression)
    k = pyGPs.cov.Linear() + pyGPs.cov.RBF() # product of kernels
    model.setPrior(kernel=k)

    model.getPosteriorIndependent(setX,setY,True)# fit default model (mean zero & rbf kernel) with data
    y_pred, ys2, fm, fs2, lp = model.predict(setX[0])

    rmseData = []
    for i in range(0,len(setY),1):
        rmse = mean_squared_error(setY[i], y_pred)**0.5
        HH = clusterList[i][0]
        rmseData.append((HH,rmse))
    return rmseData

def calculateRMSEPyGP(vectorX,vectorY,clusterList):
    """
    calculate the root mean squared error
    Parameters:
    -----------

    vectorX: featureSet
    vectorY: valueSet
    clusterlist: cluster of households
    Returns:
    --------
    list of (household,rmse) tuples
    """

    setX = [preprocessing.scale(element )for element in vectorX]
    setY=preprocessing.scale(vectorY,axis=1)#np.asarray(vectorY)
    model = pyGPs.GPR()      # specify model (GP regression)
    k =  pyGPs.cov.Linear() + pyGPs.cov.RBF() #hyperparamse will be set with optimizeHyperparameters method
    m = pyGPs.mean.Linear(D=vectorX[0].shape[1])
    model.setPrior(kernel=k,mean=m)

    #model.setPrior(kernel=k)


    hyperparams, model2 = GPCP.optimizeHyperparameters_deprecated([0.0000001,0.0000001,0.0000001],model,setX,setY,bounds=[(None,5),(None,5),(None,5)],method = 'L-BFGS-B')
    print('hyerparameters used:',hyperparams)

    y_pred, ys2, fm, fs2, lp = model2.predict(vectorX[0])


    '''
    plt.plot(y_pred, color='red')
    for i in vectorY:
        plt.plot(i,color='blue')
    plt.show(block=True)
    '''

    rmseData = []
    for i in range(0,len(vectorY),1):
        rmse = mean_squared_error(vectorY[i], y_pred)**0.5
        HH = clusterList[i][0]
        rmseData.append((HH,rmse))
    return rmseData



def divideInClusters(featureSet,clusterlist,threshold,clusterSize,splitRatio):
    """
    aux method for the clustering which devides the clusterlist further into clusters using a certain threshold
    Parameters:
    -----------

    featureSet: featureSet
    clusterlist: cluster of households
    threshold to divide the clusters
    Returns:
    --------
    list of clusters of (household,rmse) tuples
    """


    vectorX,vectorY = getHouseholdsXandYs(featureSet,clusterlist)
    #listRMSE = calculateRMSE(vectorX,vectorY,clusterlist)
    listRMSE = calculateRMSEPyGP(vectorX,vectorY,clusterlist) #switch between RMSE(mean) and RMSEPyGP


    #SPLIT HIER AL EN CHECK EERST OF DE GROOTTE VAN DE CLUSTERS OKE IS, ZOJA splits en check treshhold
    #zo nee return originele cluster of verhoog de splitrate?

    sortedListRMSE = sorted(listRMSE, key=lambda x: x[1])
    print("Threshold",threshold)

    NormalizeValue = sortedListRMSE[-1][1]
    sortedListRMSE_normalized = [(x[0],x[1] / NormalizeValue) for x in sortedListRMSE][::-1]
    print("SortedRMSEList_normalized",sortedListRMSE_normalized)

    clusterSizeLength = int(math.ceil(splitRatio * len(sortedListRMSE_normalized)))
    print('clusterSize',clusterSizeLength)

    newClusterlist = sortedListRMSE_normalized[-clusterSizeLength:]#[::-1]
    newRemaininglist = sortedListRMSE_normalized[:len(sortedListRMSE_normalized)-clusterSizeLength]#[::-1]

    if (len(newClusterlist)>=clusterSize and len(newRemaininglist)>=clusterSize):
        print('mean',np.mean([item[1] for item in sortedListRMSE_normalized]))
        if(np.mean([item[1] for item in sortedListRMSE_normalized])<threshold): #check goodness of cluster

            '''
            if((len(newClusterlist)<clusterSize) or (len(newRemaininglist)<clusterSize)):
                print('inside')
                printClusterList = [item[0] for item in clusterlist]
                return clusterlist, [],printClusterList,[]
            else:
            '''
            printClusterList = [item[0] for item in newClusterlist]
            printRemainingList = [item[0] for item in newRemaininglist]
            return newClusterlist,newRemaininglist,printClusterList,printRemainingList


        else:
            print("ELSE")
            printClusterList = [item[0] for item in clusterlist][::-1]
            return clusterlist, [],printClusterList,[]
            '''
            clusterlist =[item for item in sortedListRMSE_normalized if item[1] >= threshold]
            remaininglist = [item for item in sortedListRMSE_normalized if item[1] < threshold]
            '''


    else:
        printClusterList = [item[0] for item in clusterlist]
        return clusterlist, [],printClusterList,[]



    '''
    if (len(clusterlist)>1):
        vectorX,vectorY = getHouseholdsXandYs(featureSet,clusterlist)
        listRMSE = calculateRMSEPyGP(vectorX,vectorY,clusterlist) #switch between RMSE(mean) and RMSEPyGP
        print("RMSELIST",listRMSE)
        print("Threshold",threshold)
        clusterlist =[item for item in listRMSE if item[1] >= threshold]
        remaininglist = [item for item in listRMSE if item[1] < threshold]
        printClusterList = [item[0] for item in clusterlist]
        printRemainingList = [item[0] for item in remaininglist]
        return clusterlist,remaininglist,printClusterList,printRemainingList
    else:
        printClusterList = [item[0] for item in clusterlist]
        return clusterlist, [],printClusterList,[]
    '''


'''
def cluster(featureSet,clusterlist,threshold):
    if (clusterlist ==[]):
        print('Stephanie ERROR: clusterlist is empty')
        return []
    cluster1,cluster2,printCluster1,printCluster2=divideInClusters(featureSet,clusterlist,threshold)
    if ((cluster1==clusterlist) or (cluster2==clusterlist)):

        return clusterlist
    else:
        return cluster(featureSet,cluster1,threshold).append(cluster(featureSet,cluster2,threshold))
'''
def containsSameHouseholds (tupleList1,tupleList2):
    #check wether two tuples of form (Household,RMSE) contain the same households (aux method for the clustering)
     HHs1 =[tuple[0] for tuple in tupleList1]
     HHs2 =[tuple[0] for tuple in tupleList2]
     HHs1S = sorted(HHs1)
     HHs2S= sorted(HHs2)
     return (HHs1S==HHs2S)




def makeClusters(featureSet,clusterlist,threshold,clusterSize,splitRatio,resultList,resultTree):
    """
    recursive clustering method
    Parameters:
    -----------

    featureSet: featureSet
    clusterlist: cluster of households
    threshold: threshold for the clustering
    thresholdDelta: delta to decrement or increment the threshold after every clustering
    resultist: list of clusters of tuples (Household,RMSE)
    resultTree: list of leafs (list of households) of the clustering

    Returns:
    --------
    list of (household,rmse) tuples
    list of leafs (list of households) of the clusting
    """
    if (clusterlist ==[]):
        print('Stephanie ERROR: clusterlist is empty')


    cluster1,cluster2,printCluster1,printCluster2=divideInClusters(featureSet,clusterlist,threshold,clusterSize,splitRatio)
    if((not len(printCluster1)==0)and (not len(printCluster2)== 0)):
        resultTree.append([printCluster1,printCluster2])
    if (containsSameHouseholds(cluster1, clusterlist) or containsSameHouseholds(cluster2,clusterlist)):
        resultList.append(clusterlist)



    else:
         '''
         makeClusters(featureSet,cluster2,threshold+thresholdDelta,thresholdDelta,clusterSize,resultList,resultTree)
         makeClusters(featureSet,cluster1,threshold+thresholdDelta,thresholdDelta,clusterSize,resultList,resultTree)
         '''

         makeClusters(featureSet,cluster2,threshold,clusterSize,splitRatio,resultList,resultTree)
         makeClusters(featureSet,cluster1,threshold,clusterSize,splitRatio,resultList,resultTree)


    return resultList,resultTree

def plotClusters(clusterResults,featuresSet):
    """
    plot the clustring results
    Parameters:
    -----------

    featureSet: featureSet list of lists of form (Y,x1,...,xn) with Y = value and x1,...,xn are features
    clusterresults: list of clusters which contains (household,RMSE) tuples


    Returns:
    --------
    plot of the cunsumption of the households per cluster
    """


    HHClusters=[]
    for i in clusterResults:
        HHClusters.append([ seq[0] for seq in i ])


    i=0
    for cluster in clusterResults:

        X,Y = getHouseholdsXandYs(featuresSet,cluster)


        for plotcluster in Y:
            plt.plot(plotcluster)
            plt.title("Cluster"+str(HHClusters[i]))
            fig = plt.gcf()
            fig.canvas.set_window_title('Cluster'+str(i))


        i=i+1
        plt.show(block=True)


def clusterMain(featureSet,initialClusterlist,threshold,clusterSize,splitRatio):
    """
    main clustering method to be called
    Parameters:
    -----------

    featureSet: featureSet list of lists of form (Y,x1,...,xn) with Y = value and x1,...,xn are features
    initialclusterlist: cluster of initial tuple(household,RMSE) list e.g. [(HH0,None)..(HH35,None)]
    threshold: threshold for the clustering
    thresholdDelta: delta to decrement or increment the threshold after every clustering


    Returns:
    --------
    resultist: list of clusters of tuples (Household,RMSE)
    resultTree: list of leafs (list of households) of the clustering
    """

    clusterResults,resultTree = makeClusters(featureSet,initialClusterlist,threshold,clusterSize,splitRatio,[],[])

    return clusterResults, resultTree




def test():
    '''
    vectorX =[]
    vectorY =[]
    featureSet = []
    numberOfHH = 35
    for i in range(0,numberOfHH+1,1):
        features=fm.makeFeatureVectorAllSelectedHousehold(dfC,dfTemp,dfIrr,'2014-02-01 00:00:00','2014-02-07 23:45:00','H','HH'+str(i),'dateFrameTrainingPyGP1')
        featureSet.append(features)
        vectorX.append(features.values[:, 1:6])
        vectorY.append(features.values[:, 0])

    intialClusterlist = []
    for i in range(0,numberOfHH+1,1):
        intialClusterlist.append((i,None))

    x,y = getHouseholdsXandYs(featureSet,intialClusterlist)
    totallistRMSE = calculateRMSEPyGP(x,y,intialClusterlist)
    print('initialList',totallistRMSE)
    clusterResults,resultTree = makeClusters(featureSet,totallistRMSE,10.5,-1,[],[])


    tree = plotEng.makeTree(range(0,numberOfHH+1,1),resultTree)

    print('aantal clusters',len(clusterResults))
    plotEng.plotTree(tree.converTree(),featureSet)
    '''




if __name__ == "__main__":
    test()
