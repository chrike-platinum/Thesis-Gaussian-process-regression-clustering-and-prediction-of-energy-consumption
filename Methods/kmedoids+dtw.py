# pam.py
# PAM implemenation for pyCluster
# Aug 17, 2013
# Added timing
# May 27, 2013
# daveti@cs.uoregon.edu
# http://daveti.blog.com


'''
This python file is used to calculate the k-medoids+DTW. Use the test method the cluster.
'''
from Methods.util import DTWDistance
import random
import time
import pandas as pd
import numpy as np
import features.featureMaker as fm
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

# Global variables
initMedoidsFixed = False # Fix the init value of medoids for performance comparison
debugEnabled = True # Debug
distances_cache = {}	# Save the tmp distance for acceleration (idxMedoid, idxData) -> distance


def totalCost(data, costF_idx, medoids_idx, windowBound,cacheOn=True):
    '''
    Compute the total cost and do the clustering based on certain cost function
    '''
    # Init the cluster
    size = len(data)
    total_cost = 0.0
    medoids = {}
    for idx in medoids_idx:
        medoids[idx] = []

    # Compute the distance and do the clustering
    for i in range(size):
        choice = -1
        # Make a big number
        min_cost = float('inf')
        for m in medoids:
            if cacheOn == True:
                # Check for cache
                tmp = distances_cache.get((m,i), None)
            if cacheOn == False or tmp == None:
                if costF_idx == 0:
                    # euclidean_distance
                    tmp = DTWDistance(data[m], data[i],windowBound)
                elif costF_idx == 1:
                    # manhattan_distance
                    tmp = manhattan_distance(data[m], data[i])
                elif costF_idx == 2:
                    # pearson_distance
                    tmp = pearson_distance(data[m], data[i])
                else:
                    print('Error: unknown cost function idx: ' % (costF_idx))
            if cacheOn == True:
                # Save the distance for acceleration
                distances_cache[(m,i)] = tmp
            # Clustering
            if tmp < min_cost:
                choice = m
                min_cost = tmp
        # Done the clustering
        medoids[choice].append(i)
        total_cost += min_cost

    # Return the total cost and clustering
    return(total_cost, medoids)


def kmedoids(data, k,windowBound):
    '''
    kMedoids - PAM implemenation
    See more : http://en.wikipedia.org/wiki/K-medoids
    The most common realisation of k-medoid clustering is the Partitioning Around Medoids (PAM) algorithm and is as follows:[2]
    1. Initialize: randomly select k of the n data points as the medoids
    2. Associate each data point to the closest medoid. ("closest" here is defined using any valid distance metric, most commonly Euclidean distance, Manhattan distance or Minkowski distance)
    3. For each medoid m
        For each non-medoid data point o
            Swap m and o and compute the total cost of the configuration
    4. Select the configuration with the lowest cost.
    5. repeat steps 2 to 4 until there is no change in the medoid.
    '''
    size = len(data)
    medoids_idx = []
    if initMedoidsFixed == False:
        medoids_idx = random.sample([i for i in range(size)], k)
    else:
        medoids_idx = [i for i in range(k)]
    pre_cost, medoids = totalCost(data, 0, medoids_idx,windowBound)
    if debugEnabled == True:
        print('pre_cost: ', pre_cost)
    #print('medioids: ', medoids)
    current_cost = pre_cost
    best_choice = []
    best_res = {}
    iter_count = 0

    while True:
        for m in medoids:
            for item in medoids[m]:
                # NOTE: both m and item are idx!
                if item != m:
                    # Swap m and o - save the idx
                    idx = medoids_idx.index(m)
                    # This is m actually...
                    swap_temp = medoids_idx[idx]
                    medoids_idx[idx] = item
                    tmp_cost, tmp_medoids = totalCost(data, 0, medoids_idx,windowBound)
                    # Find the lowest cost
                    if tmp_cost < current_cost:
                        best_choice = list(medoids_idx) # Make a copy
                        best_res = dict(tmp_medoids) 	# Make a copy
                        current_cost = tmp_cost
                    # Re-swap the m and o
                    medoids_idx[idx] = swap_temp
        # Increment the counter
        iter_count += 1
        if debugEnabled == True:
            print('current_cost: ', current_cost)
            print('iter_count: ', iter_count)

        if best_choice == medoids_idx:
            # Done the clustering
            break

        # Update the cost and medoids
        if current_cost <= pre_cost:
            pre_cost = current_cost
            medoids = best_res
            medoids_idx = best_choice

    return(current_cost, best_choice, best_res)




def test(numberOfHH,numberOfHH_lbls):
    '''
    :param numberOfHH: number of households
    :param numberOfHH_lbls: number of labels of the households
    :return: a k-medoids clustering
    '''
    featureSetDTW =[]
    featureSet = []
    numberOfHH = numberOfHH
    #HHDates=[('2014-02-01 00:00:00','2014-02-07 23:45:00')]#,('2014-05-03 00:00:00','2014-05-09 23:45:00')]#,('2014-08-02 00:00:00','2014-08-08 23:45:00'),('2013-11-02 00:00:00','2013-11-08 23:45:00')]
    HHDates=[('2014-07-12 00:00:00','2014-07-18 23:45:00')]
    for i in range(0,numberOfHH,1):
        j=1
        for date in HHDates:
            features=fm.makeClusterFeatureVectorAllSelectedHousehold(dfC,dfTemp,dfIrr,date[0],date[1],'H','HH'+str(i),'dateFrameTrainingPyGPCluster') #dateFrameTrainingPyGP1
            if (len(HHDates) == 1):
                features = features.rename(columns={'HH'+str(i): 'HH'+str(i+1)})
                featureSetDTW.append(features['HH'+str(i+1)].values)
                featureSet.append(('HH'+str(i+1),features))
            else:
                features = features.rename(columns={'HH'+str(i): 'HH'+str(i+1)+'.'+str(j)}) #TODO streepje tussen HH labels laat geen sorteren toe
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

    startTime = time.time()
    best_cost, best_choice, best_medoids = kmedoids(featureSetDTW, 15,0)
    endTime = time.time()

    print('best_time: ',numberOfHH_lbls, endTime - startTime)
    print('best_cost: ', best_cost)
    print('best_choice: ', best_choice)
    print('best_medoids:')
    clustering = ''
    for i in best_choice:
        addlist = [str(item+1) for item in best_medoids[i] ]
        clustering = clustering+','+(str(addlist).replace(',','|'))

    clustering = '('+clustering[1:]+');'
    plotEng.plotTree(clustering,featureSet,'#','#','#')





if __name__ == '__main__':

    test(20,20)
    #test(10,10) #10, 5.257373094558716; 4.828974962234497
    #test(20,20) #20, 19.797210931777954; 20.36758303642273
    #test(30,30)  #30 43.815181016922
    #test(50,50) #50, 106.97584915161133; 121.81246399879456
    #test(71,71) #71, 269.99076414108276; 281.7977159023285
    #test(50,100)#100, 629.1293041706085; 563.7125210762024
    #test(60,120),#120, 668.871062040329; 752.9195749759674
    #test(71,142) #142, 1156.2243220806122






    #plotListPAM = [(10,5.26),(20,19.79),(50,106.97),(71,269.99),(100,629.13),(120,752.91),(140,1156.22)]
    plotListPAM = [(10,4.83),(30,43.82),(50,121.81),(71,281.80),(100,563.13),(120,752.91),(140,1156.22)]
    plotListXPAM = [item[0] for item in plotListPAM]
    plotListYPAM = [item[1] for item in plotListPAM]

    #plotListAglo = [(10,4.99),(20,23.82),(50,139.81),(71,267.10),(100,550.21),(140,1022.42)]
    plotListAglo = [(10,4.91),(30,43.73),(50,122.27),(71,252.50),(100,495.50),(120,714.51),(140,1022.42)]
    plotListXAglo = [item[0] for item in plotListAglo]
    plotListYAglo = [item[1] for item in plotListAglo]


    plotListGPR = [(10,29.43),(30,91.65),(50,132.20),(71,199.86),(100,273.76),(120,317.9657869338989),(140,397.92)]
    plotListXGPR = [item[0] for item in plotListGPR]
    plotListYGPR = [item[1] for item in plotListGPR]

    plt.plot(plotListXGPR,plotListYGPR,'-o',label='GPRC')
    plt.plot(plotListXPAM,plotListYPAM,'-^',label='K-medoids+DTW')
    plt.plot(plotListXAglo,plotListYAglo,'-s',label='HAC+DTW')
    plt.xlabel('Aantal tijdreeksen (huishoudens)')
    plt.ylabel('Tijd (s)')
    plt.legend(loc=0)
    plt.show(block=True)


