
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
NList=[5,10,100]

TreeForest = []
for n in NList: ## Trying different tree sizes
	k = math.ceil(math.log(sz,2))
	#M = math.ceil(float(len(XT)/n))
	M = math.ceil(70*(float(len(XT))/100))
	for ntree in range(0,n):
		selMExamples = numpy.random.permutation(len(XT))[0:M]
		ExampleList = dt_func.sampleExampleList(ExampleDict, selMExamples)
		Root = dt_func.decideRoot(ExampleList, GlobalAttrDict,k)
		gRoot = graph.graph(Root, 'ROOT', ExampleList, GlobalAttrDict, 0, -1, k)
		sflag = gRoot.ID3()
		TreeForest.append(gRoot)
		y = dt_func.Validate( TreeForest[ntree], TestDict, TestDict['Result'])
		#print len(y)

	##--- Create the input dataSet for SVM ---
	SVM_XDATA = []
	TEST_XDATA = []
	for i in range(0,len(XData)):
		##--create a test-vector as below --
		testV = dict([])
		for f in range(0,len(GlobalAttrDict['_AttrOrder_'])):
			testV[GlobalAttrDict['_AttrOrder_'][f]] = ExampleDict[GlobalAttrDict['_AttrOrder_'][f]][i]
		SVM_XDATA.append([1] + [TreeForest[ni].predictResult(testV) for ni in range(0,len(TreeForest))])
	print len(SVM_XDATA), len(YData)
	for i in range(0,len(XTest)):
		##--create a test-vector as below --
		testV = dict([])
		for f in range(0,len(GlobalAttrDict['_AttrOrder_'])):
			testV[GlobalAttrDict['_AttrOrder_'][f]] = TestDict[GlobalAttrDict['_AttrOrder_'][f]][i]
		TEST_XDATA.append([1] + [TreeForest[ni].predictResult(testV) for ni in range(0,len(TreeForest))])
	
	wvecLearn = svm_func.SVM(SVM_XDATA, YData, 0.1, 0.01, 60, len(SVM_XDATA[0]))
	#print wvecLearn
	TestStruct = svm_func.SVM_TEST(TEST_XDATA, YTest, wvecLearn)
	TrainingStruct = svm_func.SVM_TEST(SVM_XDATA, YData, wvecLearn)
	print "############  Report for N = ",n," ###################"
	PrecisionScoreTrain = TrainingStruct[0]/float(TrainingStruct[0] + TrainingStruct[1])
	RecallTrain = TrainingStruct[0]/float(TrainingStruct[0] + TrainingStruct[2])
	F1Train = (2*PrecisionScoreTrain*RecallTrain)/float(PrecisionScoreTrain + RecallTrain)
	print "Training-Mistake Count: ", TrainingStruct[3]
	print "Training-Accuracy = ",  ((len(XData) - TrainingStruct[3])/float((len(XData))))*100 , "%"
	print "Training - Precision Score = ", PrecisionScoreTrain
	print "Training - Recall Score = ", RecallTrain
	print "Training - F1-Score = ", F1Train
	PrecisionScoreTest = TestStruct[0]/float(TestStruct[0] + TestStruct[1])
	Recalltest = TestStruct[0]/float(TestStruct[0] + TestStruct[2])
	F1test = (2*PrecisionScoreTest*Recalltest)/float(PrecisionScoreTest + Recalltest)
	print "Test-Mistake Count: ", TestStruct[3]
	print "Test-Accuracy= ",  ((len(XTest) - TestStruct[3])/(float((len(XTest)))))*100, "%"
	print "Test - Precision Score = ", PrecisionScoreTest 
	print "Test - Recall Score = ", Recalltest
	print "Test - F1-Score = ", F1test
	print "========= End Report ==============================\n\n"

		


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


