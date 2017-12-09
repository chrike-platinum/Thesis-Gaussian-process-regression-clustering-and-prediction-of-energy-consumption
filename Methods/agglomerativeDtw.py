__author__ = 'christiaanleysen'
#!/usr/bin/env python
# -*- coding:utf-8 -*-


import timeit
import pandas as pd
import features.featureMaker as fm
import Methods.util as ut
import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'
import Results.plotEngine as plotEng

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

"""
Agglomerative Clustering Algorithm
Iteratively build hierarchical cluster between all data points.
O(n^2) complexity
Author: Ryan Flynn <parseerror+agglomerative-clustering@gmail.com>
References:
 1. "Hierarchical Clustering Algorithms", http://home.dei.polimi.it/matteucc/Clustering/tutorial_html/hierarchical.html
 2. "How to Explain Hierarchical Clustering", Stephen P. Borgatti, http://www.analytictech.com/networks/hiclus.htm, Retrieved May 25, 2011
 3. Johnson,S.C. 1967, "Hierarchical Clustering Schemes" Psychometrika, 2:241-254.
"""

class Cluster:
    def __init__(self):
        pass
    def __repr__(self):
        return '(%s,%s)' % (self.left, self.right)
    def add(self, clusters, grid, lefti, righti):
        self.left = clusters[lefti]
        self.right = clusters[righti]
        # merge columns grid[row][righti] and row grid[righti] into corresponding lefti
        for r in grid:
            r[lefti] = min(r[lefti], r.pop(righti))
        grid[lefti] = map(min, zip(grid[lefti], grid.pop(righti)))
        clusters.pop(righti)
        return (clusters, grid)

def agglomerate(labels, grid):
    """
    given a list of labels and a 2-D grid of distances, iteratively agglomerate
    hierarchical Cluster
    """
    clusters = labels
    while len(clusters) > 1:
        # find 2 closest clusters
        #print(clusters)
        distances = [(1, 0, grid[1][0])]
        for i,row in enumerate(grid[2:]):
            distances += [(i+2, j, c) for j,c in enumerate(row[:i+2])]
        j,i,_ = min(distances, key=lambda x:x[2])
        # merge i<-j
        c = Cluster()
        clusters, grid = c.add(clusters, grid, i, j)
        clusters[i] = c
    return clusters.pop()


def compute_similarity_matrix(data_slice):
    pairwise_dists2=[]
    for i in (data_slice):
        clist = []
        for j in data_slice:
            #i = np.asarray(i).reshape(-1, 1)
            #j = np.asarray(j).reshape(-1, 1)
            #dist, cost, acc, path = dtw(i, j,dist=scipy.linalg.norm)
            #clist.append(dist)
            clist.append(ut.DTWDistance(i,j,0))
        pairwise_dists2.append(clist)
    return pairwise_dists2


def test(nrOfHH,nrOfHH_lbls):
    '''
    featureSet = []
    numberOfHH = nrOfHH
    HHDates=[('2014-02-01 00:00:00','2014-02-07 23:45:00')]#,('2014-05-03 00:00:00','2014-05-09 23:45:00')]#,('2014-08-02 00:00:00','2014-08-08 23:45:00'),('2013-11-02 00:00:00','2013-11-08 23:45:00')]
    #HHDates=[('2014-07-12 00:00:00','2014-07-18 23:45:00')]
    for i in range(0,numberOfHH,1):
        j=0
        for date in HHDates:
            features=fm.makeClusterFeatureVectorAllSelectedHousehold(dfC,dfTemp,dfIrr,date[0],date[1],'H','HH'+str(i),'dateFrameTrainingPyGPCluster') #dateFrameTrainingPyGP1
            if (j == 0):
                features = features.rename(columns={'HH'+str(i): 'HH'+str(i)})
                #featureSet.append(('HH'+str(i),features['HH'+str(i)].values))
                featureSet.append(features['HH'+str(i)].values)
            else:
                features = features.rename(columns={'HH'+str(i): 'HH'+str(i)+'-'+str(j)})
                #featureSet.append(('HH'+str(i)+'-'+str(j),features))
                featureSet.append(features['HH'+str(i)+'-'+str(j)].values)

            j+=1

    '''
    labels=[]
    featureSetDTW=[]
    featureSet = []
    numberOfHH = nrOfHH
    #HHDates=[('2014-02-01 00:00:00','2014-02-07 23:45:00')]#,('2014-05-03 00:00:00','2014-05-09 23:45:00')]#,('2014-08-02 00:00:00','2014-08-08 23:45:00'),('2013-11-02 00:00:00','2013-11-08 23:45:00')]
    HHDates=[('2014-07-12 00:00:00','2014-07-18 23:45:00')]
    for i in range(0,numberOfHH,1):
        j=1
        for date in HHDates:
            features=fm.makeClusterFeatureVectorAllSelectedHousehold(dfC,dfTemp,dfIrr,date[0],date[1],'H','HH'+str(i),'dateFrameTrainingPyGPCluster') #dateFrameTrainingPyGP1
            if (len(HHDates) == 1):
                features = features.rename(columns={'HH'+str(i): 'HH'+str(i+1)})
                labels.append([str(i+1)])
                featureSetDTW.append(features['HH'+str(i+1)].values)
                featureSet.append(('HH'+str(i+1),features))

            else:
                features = features.rename(columns={'HH'+str(i): 'HH'+str(i+1)+'.'+str(j)}) #TODO streepje tussen HH labels laat geen sorteren toe
                labels.append([str(i+1)+'.'+str(j)])
                featureSetDTW.append(features['HH'+str(i+1)+'.'+str(j)].values)
                featureSet.append(('HH'+str(i+1)+'.'+str(j),features))

            j+=1

    initialClusterlist = []
    for i in range(0,numberOfHH,1):
        for j in range(1,len(HHDates)+1,1):
            if (len(HHDates) == 1):
                initialClusterlist.append((str(i+1),None))
            else:
                initialClusterlist.append((str(i+1)+'.'+str(j),None))

    start = timeit.default_timer()
    distances = compute_similarity_matrix(featureSetDTW)
    clustering = agglomerate(labels,distances)
    stop = timeit.default_timer()
    print('calculation time: ',nrOfHH, stop - start)
    print(clustering)

    convertedTree = (str(clustering)+';')

    plotEng.plotTree(convertedTree,featureSet,'#','#','#')



if __name__ == '__main__':
    test(71,71)
    '''
    test(10,10) #10, 4.992589950561523; 4.9124298095703125
    test(20,20) #20, 23.822389125823975; 19.553832054138184
    '''
    #test(10,10) #30 43.73
    '''
    test(50,50) #50, 139.81262183189392; 122.27435803413391
    test(71,71) #71,71, 267.0979390144348; 252.50046801567078
    test(50,100)#100,  550.2102599143982; 495.4987189769745
    test(60,120) #120   714.5087490081787
    test(70,140) #140, 1022.4158380031586;

    '''


    plotList = [(10,4.99),(20,23.82),(50,139.81),(71,267.10),(100,550.21),(140,1022.42)]
    plotListX = [item[0] for item in plotList]
    plotListY = [item[1] for item in plotList]
    plt.plot(plotListX,plotListY,'-o')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of time series (households)')
    plt.show(block=True)
