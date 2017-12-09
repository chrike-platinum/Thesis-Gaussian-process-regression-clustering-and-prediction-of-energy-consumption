"""
Module for providing cross validation strategies for time series data.

"""

from datetime import timedelta
import random


class FoldWithGap():
    """
    Splits the dataset into one training and one test set while keeping a gap
    between these two sets to avoid dependencies between them.
    
    Parameters
    ----------
    dataset: the dataset for which the split is to be computed
    testsize_days: size of the desired test set in days
    gapsize_days: size if the desired gap in days
    
    """    
    
    def __init__(self, dataset, testsize_days, gapsize_days):
        self.dataset = dataset
        self.testsize_days = testsize_days
        self.gapsize_days = gapsize_days

        delta_testset = timedelta(days=testsize_days)
        delta_gapset = timedelta(days=gapsize_days)

        testset_endtime = dataset.tail(1).index[0]
        testset_starttime = testset_endtime - delta_testset + timedelta(seconds=1)
        # slicing based on times assures that we find the next available date
        testset_starttime = dataset[testset_starttime : testset_endtime].head().index      
        
        testset_startindex = dataset.index.get_indexer_for(testset_starttime)[0]
        
        trainset = dataset[ : testset_starttime[0] - delta_gapset]
        trainset_endtime = trainset.tail(1).index        
        
        trainset_endindex = dataset.index.get_indexer_for(trainset_endtime)[0]       
        
        self.training_set_indices = range(0, trainset_endindex)
        self.test_set_indices = range(testset_startindex, len(dataset))


    def __iter__(self):
        yield self.training_set_indices, self.test_set_indices


    def __repr__(self):
        return '%s.%s(testsize_days=%s, gapsize_days=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.testsize_days,
            self.gapsize_days)
    
    
    def __len__(self):
        return 1


class KFoldWithGap():
    """
    Splits the dataset k-times into training and test set while keeping a gap
    between these two sets to avoid dependencies between them.
    
    This will create the following splits:
    
    |------------------------------------|
    
    |          Train        |Gap|  Test  |
    |    Train     |Gap|  Test  | Train  |
    ...
    |Gap|  Test  |        Train          |
    
    Parameters
    ----------
    dataset: the dataset for which the split is to be computed
    gapsize_days: size if the desired gap in days
    n_folds: number of folds to be created
    
    """    
    
    def __init__(self, dataset, gapsize_days, n_folds):
        self.dataset = dataset
        self.gapsize_days = gapsize_days
        self.n_folds = n_folds
        self.folds = []

        firstdate = self.dataset.head(1).index[0]
        lastdate = self.dataset.tail(1).index[0]
        testsize_days = ((lastdate-firstdate).days - self.gapsize_days) / self.n_folds
        
        for n in range(self.n_folds):      
        
            testset_endtime = self.dataset.tail(1).index[0] - n*timedelta(days=testsize_days)
            testset_starttime = testset_endtime - timedelta(days=testsize_days) + timedelta(seconds=1)
            
            testset = self.dataset[testset_starttime : testset_endtime]
            testset_starttime = testset.head(1).index   
            testset_endtime = testset.tail(1).index
            
            testset_startindex = self.dataset.index.get_indexer_for(testset_starttime)[0]
            testset_endindex = self.dataset.index.get_indexer_for(testset_endtime)[0]
            
            gapset = self.dataset[testset_starttime[0] - timedelta(days=self.gapsize_days) : testset_starttime[0]]
            gapset_starttime = gapset.head(1).index   
            gapset_endtime = gapset.tail(1).index   
            
            if (n==n_folds-1):
                # for last fold, take all remaining data points as gap to accomodate rounding errors
                gapset_startindex = 0    
            else:
                gapset_startindex = self.dataset.index.get_indexer_for(gapset_starttime)[0]
                
            gapset_endindex = self.dataset.index.get_indexer_for(gapset_endtime)[0]
            
            test_idx = range(testset_startindex, testset_endindex+1)
            gap_idx = range(gapset_startindex, gapset_endindex)
            
            # training set is everything apart from test and gap set
            train_idx = [ i for i in xrange(len(self.dataset)) if i not in test_idx and i not in gap_idx ]
            
            self.folds.append((train_idx, test_idx))
        

    def __iter__(self):
        for train_idx, test_idx in self.folds:
            yield train_idx, test_idx


    def __repr__(self):
        return '%s.%s(gapsize_days=%s, n_folds=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.gapsize_days,
            self.n_folds)
    
    
    def __len__(self):
        return self.n_folds


class RandomFoldWithDiscarding():
    """
    Randomly samples k data points as test points and discards all dependant
    data points.
    
    Parameters
    ----------
    dataset: the dataset for which the split is to be computed
    testsize: number of data points used for test set
    discard_days: list of day delays for which data points are to be discarded
    n_folds: number of splits to be created
    
    """    
    
    def __init__(self, dataset, testsize, discard_days, n_folds):
        self.dataset = dataset
        self.testsize = testsize
        self.discard_days = discard_days
        self.n_folds = n_folds

        self.folds = []
        datasize = len(dataset)
        
        for n in range(n_folds):
            test_idx = []
            gap_idx = []
            for i in range(testsize):
                
                testpoint_idx = random.randint(0, datasize-1)
                test_idx.append(testpoint_idx)
                testpoint_time = dataset.index[testpoint_idx]
                # add all data points corresponding to given day delays to gap set            
                for d in discard_days:
                    discard_time = testpoint_time - timedelta(days=d)
                    if discard_time in dataset.index:
                        discard_idx = dataset.index.get_loc(discard_time)
                        gap_idx.append(discard_idx)
            
            # training set is everything apart from test and gap set            
            train_idx = [ i for i in xrange(len(self.dataset)) if i not in test_idx and i not in gap_idx ]
            self.folds.append((train_idx, test_idx))


    def __iter__(self):
        for train_idx, test_idx in self.folds:
            yield train_idx, test_idx


    def __repr__(self):
        return '%s.%s(testsize=%s, discard_days=%s, n_folds=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.testsize,
            self.discard_days,
            self.n_folds)
    
    
    def __len__(self):
        return self.n_folds



import unittest
import datetime
import pandas as pd
from pandas.tslib import Timestamp

class TestFoldingStrategies(unittest.TestCase):

    def setUp(self):
        self.dataset = self.createTestDataset()


    def createTestDataset(self):
        start_date = datetime.datetime(2014, 1, 1, 0, 0, 0)
        end_date = datetime.datetime(2014, 12, 31, 23, 0, 0)
        index = pd.date_range(start_date, end_date, freq='H')
        columns = ['X', 'y']
        dataset = pd.DataFrame(index=index, columns=columns)
        return dataset
        
    
    def testFoldWithGap(self):
        folds = FoldWithGap(self.dataset, 10, 10)
        self.assertEqual(1, len(folds))
        
        for train_idx, test_idx in folds:
            # check test set
            self.assertEqual(10*24, len(test_idx))
            startTimeTestSet = self.dataset.ix[test_idx].head(1).index[0]
            self.assertEqual(Timestamp('2014-12-22 00:00:00'), startTimeTestSet)
            endTimeTestSet = self.dataset.ix[test_idx].tail(1).index[0]
            self.assertEqual(Timestamp('2014-12-31 23:00:00'), endTimeTestSet)
            # check training set
            self.assertEqual(345*24, len(train_idx))
            startTimeTrainSet = self.dataset.ix[train_idx].head(1).index[0]
            self.assertEqual(Timestamp('2014-01-01 00:00:00'), startTimeTrainSet)
            endTimeTrainSet = self.dataset.ix[train_idx].tail(1).index[0]
            self.assertEqual(Timestamp('2014-12-11 23:00:00'), endTimeTrainSet)
         

    def testKFoldWithGap(self):
        gapsize = 10
        n_folds = 10
        folds = KFoldWithGap(self.dataset, gapsize, n_folds)
        self.assertEqual(n_folds, len(folds))
        testsize = (365 - gapsize) / n_folds
        n=0
        for train_idx, test_idx in folds:
            n+=1
            
            # check test set
            self.assertEqual(testsize*24, len(test_idx))
            startTimeTestSet = self.dataset.ix[test_idx].head(1).index[0]
            expStartTimeTestSet = Timestamp('2014-12-31 23:00:00')-n*timedelta(days=testsize)+timedelta(seconds=3600)
            self.assertEqual(expStartTimeTestSet, startTimeTestSet)
            endTimeTestSet = self.dataset.ix[test_idx].tail(1).index[0]
            expEndTimeTestSet = Timestamp('2014-12-31 23:00:00')-(n-1)*timedelta(days=testsize)
            self.assertEqual(expEndTimeTestSet, endTimeTestSet)
            # check training set
            startTimeTrainSet = self.dataset.ix[train_idx].head(1).index[0]
            endTimeTrainSet = self.dataset.ix[train_idx].tail(1).index[0]
            
            if (n==n_folds):
                expStartTimeTrainSet1 = expEndTimeTestSet + timedelta(seconds=3600)
            else:
                expStartTimeTrainSet1 = Timestamp('2014-01-01 00:00:00')

            if (n==1):
                expEndTimeTrainSet1 = expStartTimeTestSet - timedelta(days=10) - timedelta(seconds=3600)
            else:
                expEndTimeTrainSet1 = Timestamp('2014-12-31 23:00:00')
            
            self.assertEqual(expStartTimeTrainSet1, startTimeTrainSet)
            self.assertEqual(expEndTimeTrainSet1, endTimeTrainSet)
            
            
    def testRandomFoldWithDiscarding(self):
        testsize = 30
        n_folds = 10
        discard_days = [1,7,14]
        folds = RandomFoldWithDiscarding(self.dataset, testsize, discard_days, n_folds)
        self.assertEqual(n_folds, len(folds))
        n=0
        for train_idx, test_idx in folds:
            n+=1
            # check test set
            self.assertEqual(testsize, len(test_idx))
            self.assertTrue(len(train_idx) >= len(self.dataset)-(len(discard_days)+1)*testsize)            
            
            test_set = self.dataset.iloc[test_idx]            
            train_set = self.dataset.iloc[train_idx]
            
            for test_point_index, _ in test_set.iterrows():            
                
                # check discarded elements not in training set
                for d in discard_days:
                    discarded = test_point_index - timedelta(days=d)
                    self.assertFalse(discarded in train_set.index.tolist())     


if __name__ == '__main__':
    unittest.main()
