__author__ = 'christiaanleysen'

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import math
from ete3 import Tree, TreeStyle,faces
import Results.tree as TB
import ast
from PIL import Image,ImageDraw,ImageFont
import Results.tree as TREE
import plotly.plotly as py
py.sign_in('christiaanleysen','r0b7k9csea')
import plotly.graph_objs as go



# Turn interactive plotting off
plt.ioff()

pd.options.display.mpl_style = 'default'
# coding=utf-8
dataFrameLocation = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/OldResults/'

def loadDataCSV(dataFrameLocation,dataFrameFileName):
    df = pd.read_csv(dataFrameLocation+dataFrameFileName+'.csv', sep='\t', encoding='latin1',
    index_col=0)
    return df

def loadMultipleCSVDataframes(dataFrameLocation,dataFrameFileNameList):
    resultlist = []
    for name in dataFrameFileNameList:
        resultlist.append(loadDataCSV(dataFrameLocation,name))

    return resultlist


def plotErrorPerMonth(dataFramesLocation,fileNames,numberOfFilesPerMonth):

    dates = []
    for year in range(2013, 2015):
            for month in range(1, 13):
                dates.append(dt.datetime(year=year, month=month, day=1))

    valuesPlotList = []
    if (numberOfFilesPerMonth == 1): #data is included in one file for example HP and Ot together
        dataFrameList = loadMultipleCSVDataframes(dataFramesLocation,fileNames)

        for dfi in dataFrameList:
            valuesPlotList.append(dfi['% fout'].get('Total'))


        print(valuesPlotList)
        print(dates)
        plt.figure(figsize=(10,4))
        plt.plot(dates[8:20], valuesPlotList)
        x1,x2,y1,y2 = plt.axis()

        plt.axis((x1,x2,0,100))
        plt.rcParams.update({'font.size': 9})
        plt.title("RMSE per maand")
        plt.xlabel('Maand')
        plt.ylabel('RMSE')
        plt.savefig(dataFramesLocation+'RMSE per month.pdf')
        plt.show(block=True)


def plotErrorPerMonth_List(dataFramesLocations,fileNames,labels,saveName):

    dataFrameLists = []
    for i in range(len(dataFramesLocations)):
            dataFrameList = loadMultipleCSVDataframes(dataFramesLocations[i],fileNames[i])
            dataFrameLists.append(dataFrameList)

    dates = []
    for year in range(2013, 2015):
            for month in range(1, 13):
                dates.append(dt.datetime(year=year, month=month, day=1))



    valuesPlotLists = []
    for dataFrameList2 in dataFrameLists:
            valuesPlotList = []
            for dfi in dataFrameList2:
                #print(dfi["MRE"][0:len(dfi)-2])
                valuesPlotList.append(dfi['MRE'].get('Total'))
                valuesPlotList2 = [(i*0.9859) for i in valuesPlotList]
            #valuesPlotList = [(x / 12)  for x in valuesPlotList]
            valuesPlotLists.append(valuesPlotList2)



    plt.figure(figsize=(10,4))
    makers = ['o','^','s','x','*','>','<']
    i=0
    for plotData in valuesPlotLists:
            plt.plot(dates[8:20],plotData,label=labels[i],marker=makers[i])
            i+=1


    plt.legend(loc=9)
    x1,x2,y1,y2 = plt.axis()

    plt.axis((x1,x2,20,140))
    plt.rcParams.update({'font.size': 9})
    plt.title("MRE per maand")
    plt.xlabel('Maand')
    plt.ylabel('MRE')
    dataFrameLocation = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/'
    plt.savefig(dataFrameLocation+saveName)
    plt.show(block=True)





def plotErrorPerHH(dataFramesLocation,fileNames,ShowNotHomeValues=False):
        dataFrameList = loadMultipleCSVDataframes(dataFramesLocation,fileNames)
        nrOfHH = len(dataFrameList[0])-2
        valuesPlotList = [float(0)] * nrOfHH

        for dfi in dataFrameList:
            valuesPlotList = [float(x) + float(y) if (not math.isnan(y)) else float(x) + float(0.97) for x, y in zip(valuesPlotList, dfi["% fout"][0:len(dfi)-2])]
            #the check for Nan is because of a strange parse bug when the value is 0.97 in the dataframes

        valuesPlotList = [x / 12  for x in valuesPlotList]
        #valuesPlotList = [x / 12 if x!=0 else x for x in valuesPlotList] #meter was probabily broken for this HH because whole te



        plt.figure(figsize=(8,5))
        plt.plot(valuesPlotList)

        plt.axis([0, nrOfHH, 0, 100])
        plt.title("RMSE per Huishouden")
        plt.xlabel('Huishouden')
        plt.ylabel('RMSE')
        plt.savefig(dataFramesLocation+'RMSE per HH.pdf')
        plt.show(block=True)


def plotErrorPerHH_List(dataFramesLocations,fileNames,labels,saveName,ShowNotHomeValues=False,nrOfHH=71):

        dataFrameLists = []
        for i in range(len(dataFramesLocations)):
            dataFrameList = loadMultipleCSVDataframes(dataFramesLocations[i],fileNames[i])
            dataFrameLists.append(dataFrameList)




        valuesPlotLists = []
        for dataFrameList2 in dataFrameLists:
            valuesPlotList = [float(0)] * nrOfHH
            for dfi in dataFrameList2:
                #print(dfi["MRE"][0:len(dfi)-2])
                valuesPlotList = [float(x) + float(y*0.9859) if (not math.isnan(y)) else float(x) + float(0.5) for x, y in zip(valuesPlotList, dfi["MRE"][0:len(dfi)-2])]
                print(valuesPlotList)
            valuesPlotList = [(x / 12)  for x in valuesPlotList]
            valuesPlotLists.append(valuesPlotList)

            #the check for Nan is because of a strange parse bug when the value is 0.97 in the dataframes

        '''
        plt.figure(figsize=(8,5))
        print('length',len(valuesPlotLists))
        i =0
        for plotData in valuesPlotLists:
            plt.plot(plotData,label=labels[i])
            i+=1




        plt.legend(loc=9)
        plt.axis([0, nrOfHH, 0, 100])
        plt.title("Gemiddelde MSE per Huishouden")
        plt.xlabel('Huishouden')
        plt.ylabel('MSE')

        dataFrameLocation = '/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/'
        plt.savefig(dataFrameLocation+saveName)
        plt.show(block=True)
        '''
        valuesPlotList



        data = []
        i = 0
        for plotData in valuesPlotLists:
            print('plotData',plotData)
            data.append(go.Bar( x=range(1,72), y=plotData,name=labels[i]))
            i+=1

        layout = go.Layout(
            xaxis=dict(
                title='Huishoudens',
                titlefont=dict(
                family='Arial, sans-serif',
                size=22),
                range=[0, 13.5],
                tickfont = dict(size=20)


            ),
            yaxis=dict(
                title='MRE',
                titlefont=dict(
                family='Arial, sans-serif',
                size=22),
                range=[0, 200],
                tickfont = dict(size=20)

            ),
            legend=dict(
                    font=dict(
                        family='sans-serif',
                        size=20,
                        color='#000')
            ),



        barmode='group')

        fig = go.Figure(data=data, layout=layout)
        plot_url = py.plot(fig, filename='grouped-bar')



def makeTree(HHlist,inputlist):
    '''
    tree = TB.binarynode(HHlist,None,None)
    for lists in inputlist:
        orderedlist = reversed(sorted(lists, key=len))
        for list in orderedlist:
            tree.insertBF(list)
    return tree
    '''
    tree = TREE.Node(None)
    tree.value = HHlist
    for lists in inputlist:
        orderedlist1 = [sorted(item,key=lambda x: float(x)) for item in lists] #sort on integers
        orderedlist = reversed(sorted(orderedlist1, key=len)) #sort on length of list
        for list in orderedlist:
            TREE.insertBF(tree,list)

    return tree



def getHouseholdsXandYs (featuresSet,listHHNr):
    vectorX=[]
    vectorY=[]
    for i in listHHNr:
        HHTuples = dict(featuresSet)
        vectorX.append(HHTuples['HH'+str(i)].values[:,1:20]) #todo stond vroeger op 6
        vectorY.append(HHTuples['HH'+str(i)].values[:,0])
    #.loc[features[searchKeyDataFrame].isin(listHHNr)]
    return vectorX,vectorY


def getplotClustersImage(cluster,featuresSet,saveLocation):
        plt.ioff()
        X,Y = getHouseholdsXandYs(featuresSet,cluster)

        vectorY = []
        for i in range(0,len(featuresSet),1):
            vectorY.append(featuresSet[i][1].values[:,0])


        maximum = max([item for sublist in vectorY for item in sublist])
        Ymax = 1.05*maximum
        fig = plt.figure(figsize=(8, 6))
        for plotcluster in Y:
            plt.plot(plotcluster)
            for i in range(0,len(plotcluster),24):
                plt.axvline(x=i)
            x1,x2,y1,y2 = plt.axis()
            plt.axis((x1,x2,0,Ymax))
            plt.title("Cluster "+str(cluster),fontsize=12)

        #fig.savefig(saveLocation+"Cluster "+str(cluster)+'.png')
        fig.savefig(saveLocation+"TempCluster.png")
        plt.close(fig)

def getclusterTextImage(cluster,saveLocation):
    print('cluster',cluster)
    clusterText = str(cluster)
    img = Image.new('RGB', (3000, 100), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("ArialHB.ttc", 80)
    d.text((0, 0), clusterText,font=font,fill=(0, 0, 0))
    img.save(saveLocation+"Cluster "+str(cluster)+'.png')


def mylayout(node):
     saveLocation = "/Users/christiaanleysen/Dropbox/thesis1516/3E-building_energy_consumption/trydata/Results/OldResults/clusterImages/"
     featureSet = getGlobalFeatures()
     threshold = getglobalThreshold()
     minimumClusterSize = getglobalMinimumclusterSize()
     splitRatio = getglobalSplitRatio()
     if node.is_leaf():
        leafname = ast.literal_eval(node.name.replace('|',','))
        node.name = leafname
        node.add_feature("# cluster",str(leafname))
        node.add_feature("# clusterSize",len(leafname))
        node.add_feature("# threshold",threshold)
        node.add_feature("# minimum cluster size",minimumClusterSize)
        node.add_feature("# split ratio",splitRatio)
        #featureSet = [tuple[1] for tuple in featureSet]
        getplotClustersImage(leafname,featureSet,saveLocation)
        #getclusterTextImage(leafname,saveLocation)
        #plot = faces.ImgFace(saveLocation+"Cluster "+str(leafname)+'.png')
        plot = faces.ImgFace(saveLocation+"TempCluster.png")
        plot.rotation= -90
        faces.add_face_to_node(plot, node, 0)

     else:
            leafnames = node.get_leaf_names()
            HHlist = []
            for i in leafnames:
                HHs = ast.literal_eval(i.replace('|',','))
                HHlist.append(HHs)

            HHlist = sorted([item for sublist in HHlist for item in sublist],key=lambda x: float(x))
            #node.del_feature('bgcolor')
            node.add_feature("name",str(HHlist))
            node.add_feature("# cluster",str(HHlist))
            node.add_feature("# clusterSize",len(HHlist))
            node.add_feature("# threshold",threshold)
            node.add_feature("# minimum cluster size",minimumClusterSize)
            node.add_feature("# split ratio",splitRatio)
            getplotClustersImage(HHlist,featureSet,saveLocation)
            #getclusterTextImage(HHlist,saveLocation)
            #plot = faces.ImgFace(saveLocation+"Cluster "+str(HHlist)+'.png')
            plot = faces.ImgFace(saveLocation+"TempCluster.png")
            plot.rotation= -90
            faces.add_face_to_node(plot, node, 0)




def plotTree(tree,featuresSet,threshold,minimumClusterSize,splitPercentage,images=True):
    setglobalMinimumClusterSize(minimumClusterSize)
    setglobalSplitRatio(splitPercentage)
    setglobalThreshold(threshold)
    setGlobalFeatures(featuresSet)


    t=Tree(tree)
    ts = TreeStyle()
    if images:
        ts.layout_fn = mylayout
        ts.show_leaf_name = False
    else:
         ts.show_leaf_name = True
    ts.rotation = 90


    print(t)
    t.show(tree_style=ts)



globalFeatures = None

def setGlobalFeatures(features):
    global globalFeatures    # Needed to modify global copy of globvar
    globalFeatures = features

def getGlobalFeatures():
    global globalFeatures
    return globalFeatures

globalThreshold = None

def setglobalThreshold(TH):
    global globalThreshold    # Needed to modify global copy of globvar
    globalThreshold = TH

def getglobalThreshold():
    global globalThreshold
    return globalThreshold

minimumClusterSize = None

def setglobalMinimumClusterSize(D):
    global minimumClusterSize    # Needed to modify global copy of globvar
    minimumClusterSize = D

def getglobalMinimumclusterSize():
    global minimumClusterSize
    return minimumClusterSize

splitRatio = None

def setglobalSplitRatio(D):
    global splitRatio    # Needed to modify global copy of globvar
    splitRatio = D

def getglobalSplitRatio():
    global splitRatio
    return splitRatio













