
#This is used to calculate the DTW

# util.py
# Utils for pyCluster
# May 27, 2013
# daveti@cs.uoregon.edu
# http://daveti.blog.com

from math import sqrt
import numpy as np


def euclidean_distance(vector1, vector2):
	dist = 0
	for i in range(len(vector1)):
		dist += (vector1[i] - vector2[i])**2
	return(dist)

def manhattan_distance(vector1, vector2):
	dist = 0
	for i in range(len(vector1)):
		dist += abs(vector1[i] - vector2[i])
	return(dist)

def pearson_distance(vector1, vector2):
	"""
	Calculate distance between two vectors using pearson method
	See more : http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
	"""
	sum1 = sum(vector1)
	sum2 = sum(vector2)

	sum1Sq = sum([pow(v,2) for v in vector1])
	sum2Sq = sum([pow(v,2) for v in vector2])

	pSum = sum([vector1[i] * vector2[i] for i in range(len(vector1))])

	num = pSum - (sum1*sum2/len(vector1))
	den = sqrt((sum1Sq - pow(sum1,2)/len(vector1)) * (sum2Sq - pow(sum2,2)/len(vector1)))

	if den == 0 : return 0.0
	return(1.0 - num/den)

def DTWDistance(s1,s2,w=None):

		'''
		Calculates dynamic time warping Euclidean distance between two
		sequences. Option to enforce locality constraint for window w.
		'''
		DTW={}

		if w:
			w = max(w, abs(len(s1)-len(s2)))

			for i in range(-1,len(s1)):
				for j in range(-1,len(s2)):
					DTW[(i, j)] = float('inf')

		else:
		    for i in range(len(s1)):
		        DTW[(i, -1)] = float('inf')
		    for i in range(len(s2)):
		        DTW[(-1, i)] = float('inf')

		DTW[(-1, -1)] = 0

		for i in range(len(s1)):
			if w:
				for j in range(max(0, i-w), min(len(s2), i+w)):
					dist= (s1[i]-s2[j])**2
					DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
			else:
				for j in range(len(s2)):
					dist= (s1[i]-s2[j])**2
					DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

		return np.sqrt(DTW[len(s1)-1, len(s2)-1])
