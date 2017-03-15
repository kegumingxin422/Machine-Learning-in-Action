from numpy import *
import operator
import pdb

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.1],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX,dataSet,labels,k):
	diffMat = tile(inX,(dataSet.shape[0],1))-dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distance = sqDistances**0.5
	sortedDistIndicies = distance.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
		# pdb.set_trace()
	sortedClassCount = sorted(classCount,reverse = True)
	pdb.set_trace()

if __name__ == '__main__':
	# group,labels = createDataSet()
	# print (group,labels)
	# classify0([2,2],array([[0,1],[1,1],[1,0],[0,0]]),['A','B','B','C'],3)
	tmpA = [[1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
	retD = []
	for tmp in tmpA:
		if tmp[1] == 1:
			reduced = tmp[:1]
			print(reduced)
			reduced.extend(tmp[2:])
			print(reduced)
			retD.append(reduced)
	print(retD)
