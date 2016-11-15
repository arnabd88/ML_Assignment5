

import sys
import re
import copy
import math
import numpy

def parseInfo ( xvecs, labels ):
	dsize = len(xvecs)
	xdata = []
	dtemp = []
	ylabel = []
	if(dsize != len(labels)):
		print 'Error: mismatch in data size and label size...Exiting!'
		sys.exit()
	else:
		for i in range(0,dsize):
			line = xvecs[i].split()
			dtemp = [1.0]  ## Adding the bias term
			for j in range(0,len(line)):
				dtemp.append(float(line[j]))
			xdata.append(dtemp)
			ylabel.append(int(labels[i]))
		return [xdata, ylabel]


def CreateGlobalDict( exdata, exlabel, tdata, tlabel , cont):
	##--- Extract the features from the data file in order ---##
	sz = len(exdata[0]) ## number of features
	AttrDict = dict(dict([]))
	AttrDict['_AttrOrder_'] = []
	AttrFeature = dict(dict([]))
	AttrDict['_AttrOrder_'] = []
	for i in range(0,sz):
		AttrDict['_AttrOrder_'].append('F'+str(i))
	FeatureOrder = AttrDict['_AttrOrder_']
	ExampleDict = dict([])
	tempTranspose = [[row[i] for row in exdata] for i in range(0,len(exdata[0]))]
	for i in range(0,len(FeatureOrder)):
		ExampleDict[FeatureOrder[i]] = tempTranspose[i]
	ExampleDict['Result'] = exlabel

	TestDict = dict([])
	tempTranpose = [[row[i] for row in tdata] for i in range(0,len(tdata[0]))]
	for i in range(0,len(FeatureOrder)):
		TestDict[FeatureOrder[i]] = tempTranpose[i]
	TestDict['Result'] = tlabel
	##--- List the labelling ---##
	AttrDict['classes'] = [-1,1]
	if(cont == 1): ## Mark the labels
		for i in range(0,len(AttrDict['_AttrOrder_'])):
			tempTranspose = [[row[j] for row in exdata] for j in range(0,len(exdata[0]))]
			#AttrDict[AttrDict['_AttrOrder_'][i]] = list(set(tempTranspose[i]))
			AttrDict[AttrDict['_AttrOrder_'][i]] = ""
			
			
		
	return [AttrDict, ExampleDict, TestDict]


def sampleExampleList(ExampleDict, M):
	exDict = dict([]) 
	if (len(ExampleDict['Result']) <= len(M)):
		return ExampleDict
	else:
		for item in ExampleDict.keys():
			exDict[item] = []
			for m in M:
				exDict[item].append(ExampleDict[item][m])
	return exDict
			
		

	
def pruneExampleList(ExampleDict, Features):
    exDict = copy.deepcopy(ExampleDict)
    for k in exDict.keys():
		if k not in Features and k!='Result':
			del exDict[k]
    return exDict

def Entrophy(Example, AttrDict):
  	res = AttrDict['classes']
	LabelCount = 0.0
	LabelCount_0_Count = 0.0
	LabelCount_1_Count = 0.0
	for i in Example['Result']:
		LabelCount = LabelCount + 1.0
		if(i==res[0]):
			LabelCount_0_Count = LabelCount_0_Count + 1.0
		else:
			LabelCount_1_Count = LabelCount_1_Count + 1.0
	if(LabelCount_1_Count==0 or LabelCount_0_Count==0):
		Entr_S = 0
	else:
		Entr_S = -(LabelCount_0_Count/LabelCount)*numpy.log2(LabelCount_0_Count/LabelCount) - (LabelCount_1_Count/LabelCount)*numpy.log2(LabelCount_1_Count/LabelCount)
	
	return [LabelCount_0_Count, LabelCount_1_Count, LabelCount, Entr_S]
		
  

def getIG1(Example, Attr, feat, Entrophy_S):
  	AvgEnt = 0.0
	values_feat = Attr[feat]
	sampleExList = Example[feat]
	for v in values_feat:
		localExampleDict = dict([])
		localExampleDict['Result'] = []
		for i in range(0,len(sampleExList)):
			if(v == sampleExList[i]):
				localExampleDict['Result'].append(Example['Result'][i])
		tempEntrophyRet = Entrophy(localExampleDict, Attr);
		AvgEnt = AvgEnt + (tempEntrophyRet[2]/len(sampleExList)) * tempEntrophyRet[3]
	return (Entrophy_S - AvgEnt)


def getIG(Example, Attr, feat, Entrophy_S):
	AvgEnt = 0.0
	tupSet = [[Example[feat][j], Example['Result'][j]] for j in range(0,len(Example['Result']))]
	tupSet.sort(key=lambda tup:tup[0])
	splitList = []
	pLabel = tupSet[0][1]
	for k in range(1,len(tupSet)):
		cLabel = tupSet[k][1]
		if(pLabel != cLabel):
			splitList.append(float(tupSet[k][0] + tupSet[k-1][0])/2)
		pLabel = cLabel

	sampleExList = Example[feat]
	bestIg = 0
	bestSplit = 0
	for k in splitList:
		localExampleDict = dict([])
		localExampleDict['Result'] = []
		AvgEnt = 0.0
		for th in range(0,2):
			for i in range(0, len(sampleExList)):
				if(sampleExList[i] >= k):
					localExampleDict['Result'].append(Example['Result'][i])
			tempEntrophyRet = Entrophy(localExampleDict, Attr)
			AvgEnt + AvgEnt + (tempEntrophyRet[2]/len(sampleExList))*tempEntrophyRet[3]
		ig = (Entrophy_S - AvgEnt)
		if(ig >= bestIg):
			bestIg = ig
			bestSplit = k

	return [bestIg, bestSplit]
		





def decideRoot(ExampleDict, AttrDict, k):
	BestFeature = ''
	BestFeatureValue = 0
	E_S = Entrophy(ExampleDict, AttrDict)
	if(len(AttrDict['_AttrOrder_']) > k):
		indexList = numpy.random.permutation(len(AttrDict['_AttrOrder_']))[0:k]
	else:
		indexList = numpy.random.permutation(len(AttrDict['_AttrOrder_']))
	for f in indexList:
		feat = AttrDict['_AttrOrder_'][f]
		[IGv, csplit] = getIG(ExampleDict, AttrDict, feat, E_S[3])
		if(IGv >= BestFeatureValue):
			BestFeatureValue = IGv ;
			BestFeature = feat
	return [BestFeature, csplit]


def decideRoot1(ExampleDict, AttrDict, k):
	BestFeature = ''
	BestFeatureValue = 0
	E_S = Entrophy(ExampleDict, AttrDict)
	if(len(AttrDict['_AttrOrder_']) > k):
		indexList = numpy.random.permutation(len(AttrDict['_AttrOrder_']))[0:k]
	else:
		indexList = AttrDict['_AttrOrder_']
	for f in indexList:
		feat = AttrDict['_AttrOrder_'][f]
		IGv = getIG(ExampleDict, AttrDict, feat, E_S[3])
		if(IGv >= BestFeatureValue):
			BestFeatureValue = IGv ;
			BestFeature = feat;
	return BestFeature



def Validate(dt,  vectDict, Result):
	testV = dict([])
	CollectResult = []
	for x in range(0,len(vectDict[vectDict.keys()[0]])):
		for w in range(0,len(vectDict.keys())):
			if(vectDict.keys()[w] != 'Result'):
				testV[vectDict.keys()[w]] = vectDict[vectDict.keys()[w]][x]

		pres = dt.predictResult(testV)
		if(pres != Result[x]):
			CollectResult.append(testV)
#	if(len(CollectResult) == 0):
#		print "Validation Successful: Accurate Prediction"
#	else:
#		print "Inaccuracy in Prediction"
	return CollectResult

					
				

			
				

			

		
