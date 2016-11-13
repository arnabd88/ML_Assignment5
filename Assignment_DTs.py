
import sys
import re
import copy
import dt_func
import svm_func
import graph
import math
import numpy

fold_index = -1
test_index = -1

if('-fold' in sys.argv):
	fold_index = sys.argv.index('-fold')
if ('-test' in sys.argv):
	test_index = sys.argv.index('-test')

if(test_index != -1):
	testFileData = open(sys.argv[test_index+  1], 'r+').read().splitlines()
	testFileLabel = open(sys.argv[test_index+  2], 'r+').read().splitlines()



if(fold_index != -1):
	foldValue = int(sys.argv[fold_index + 1])
	trainFileData = open(sys.argv[fold_index+ 1 + 1], 'r+').read().splitlines()
	trainFileLabel = open(sys.argv[fold_index+ 1 + 2], 'r+').read().splitlines()
else:
	print 'Training Files not provided ........... Exiting!!'
	sys.exit()

if(test_index != -1):
	testFileData = open(sys.argv[test_index+  1], 'r+').read().splitlines()
	testFileLabel = open(sys.argv[test_index+  2], 'r+').read().splitlines()


[XData, YData] = svm_func.parseInfo(trainFileData, trainFileLabel)
if(test_index != -1):
	[XTest, YTest] = svm_func.parseInfo(testFileData, testFileLabel)

#[GlobalAttrDict, ExampleStruct, TestStruct] = dt_func.CreateGlobalDict(XData, YData, XTest, YTest)
XT =[ XData[i][1:] for i in range(0,len(XData))]
[ GlobalAttrDict, ExampleDict, TestDict] = dt_func.CreateGlobalDict(XT, YData, XTest, YTest, 1)

sz = len(XT[0])
##BreakDown to N decision trees of k features
#NList = [5,10,30,100]
NList=[5]

TreeForest = []
for n in NList:
	k = math.ceil(math.log(sz,2))
	M = math.ceil(float(len(XT)/n))
	for ntree in range(0,n):
		selMExamples = numpy.random.permutation(len(XT))[0:M]
		ExampleList = dt_func.sampleExampleList(ExampleDict, selMExamples)
		Root = dt_func.decideRoot(ExampleList, GlobalAttrDict,k)
		gRoot = graph.graph(Root, 'ROOT', ExampleList, GlobalAttrDict, 0, -1, k)
		sflag = gRoot.ID3()
		TreeForest.append(gRoot)
		y = dt_func.Validate( TreeForest[ntree], TestDict, TestDict['Result'])
		print len(y)

##--- Create the input dataSet for SVM ---


## for n in NList:
## 	k = math.ceil(math.log(sz,2))
## 	TreeList = dict(dict([]))
## 	for ntree in range(0,1): ## working on each tree
## 		localTree = dict([])
## 		## prune the feature list and the example set
## 		permIndex = numpy.random.permutation(len(GlobalAttrDict['_AttrOrder_']))
## 		permIndex = permIndex[0:k]
## 		localTree['Feature'] = []
## 		for k in permIndex:
## 			f = GlobalAttrDict['_AttrOrder_'][k]
## 			localTree['Feature'].append(f)
## 			localTree[f] = GlobalAttrDict[f]
## 		localTree['classes'] = GlobalAttrDict['classes']
## 		ExampleList = dt_func.pruneExampleList(ExampleDict, localTree['Feature'])
## 		TestList = dt_func.pruneExampleList(TestDict, localTree['Feature'])
## 		Root = dt_func.decideRoot(ExampleList, localTree)
## 		print "Root: ",Root
## 		gRoot = graph.graph(Root, 'ROOT', ExampleList, localTree, 0, -1)
## 		sFlag = gRoot.ID3()
## 		localTree['TreePointer']=gRoot
		


 ##--- Just work on the decision tree ----
 #Root = func.decideRoot(ExampleDict, GlobalAttrDict)


