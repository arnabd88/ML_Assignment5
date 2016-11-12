

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
			AttrDict[AttrDict['_AttrOrder_'][i]] = list(set(tempTranspose[i]))
			
			
		
	return [AttrDict, ExampleDict, TestDict]
	
	
def pruneExampleList(ExampleDict, Features):
    exDict = copy.deepcopy(ExampleDict)
    for k in exDict.keys():
		if k not in Features and k!='Result':
			del exDict[k]
    return exDict

def Entrophy(Example, Features, AttrDict):
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
		
  

def getIG(Example, kfeatures, Attr, feat, Entrophy_S):
  	AvgEnt = 0.0
	values_feat = Attr[feat]
	sampleExList = Example[feat]
	for v in values_feat:
		localExampleDict = dict([])
		localExampleDict['Result'] = []
		for i in range(0,len(sampleExList)):
			if(v == sampleExList[i]):
				localExampleDict['Result'].append(Example['Result'][i])
		tempEntrophyRet = Entrophy(localExampleDict, kfeatures, Attr);
		AvgEnt = AvgEnt + (tempEntrophyRet[2]/len(sampleExList)) * tempEntrophyRet[3]
	return (Entrophy_S - AvgEnt)


def decideRoot(ExampleDict, kfeatures, AttrDict):
  	Features = copy.deepcopy(kfeatures)
	print "Features: ", kfeatures
	BestFeature = ''
	BestFeatureValue = 0
	E_S = Entrophy(ExampleDict, Features, AttrDict)
	for feat in Features:
		IGv = getIG(ExampleDict, kfeatures, AttrDict, feat, E_S[3])
		if(IGv >= BestFeatureValue):
			BestFeatureValue = IGv ;
			BestFeature = feat;
	return BestFeature




	#if(cont==1) : ## if data is continuous , performs the splits
	#	#for i in range(0,len(AttrDict['_AttrOrder_'])):
	#	for i in range(0,1):
	#		tup2 = []
	#		for j in range(0,len(exdata)):
	#			tup2.append([exdata[j][i], exlabel[j]])
	#		tup2.sort(key=lambda tup: tup[0]) ## generates sorted data for a feature. Now mark the partitions
	#		print tup2
	#		splitList = [] ## tracks the split points
	#		prevLabel = tup2[0][1] ## take this as the default label
	#		for k in range(0,len(tup2)):
	#			currLabel = tup2[k][1]
	#			if(currLabel != prevLabel):
	#				splitList.append(k)
	#			prevLabel = currLabel
	#		print splitList

	#		ldict = dict([])
	#		for k in range(0,len(splitList)):
	#			boundDict = dict([])
	#			if(k==0):
	#				boundDict['L'] = '-'
	#				boundDict['U'] = tup2[splitList[k]][0]
	#			elif(k==len(splitList)-1):
	#				boundDict['L'] = tup2[splitList[k]][0]
	#				boundDict['U'] = '-'
	#			else:
	#				boundDict['L'] = tup2[splitList[k-1]][0]
	#				boundDict['U'] = tup2[splitList[k]][0]
	#			ldict['L'+str(k)] = boundDict
	#		AttrFeature[AttrDict['_AttrOrder_'][i]] = ldict
	#	print AttrFeature

					
				

			
				

			

		
