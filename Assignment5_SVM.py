##------ Assignment5 : SVM ------------------
import sys
import re
import copy
import svm_func
import math
import numpy

## -fole -test -margi9n -sanity

fold_index = -1 ## default required for single training file
test_index = -1 ## when testing required pass this switch
foldValue = 0 ## required with fold index
initRandom = 0 ## enable random bit. Default zero

CList = [1,  pow(2,1), pow(2,-1), pow(2,-2), pow(2,-4), pow(2,2), 0.01, 0.1]
GList = [1, 0.1, 0.01, 0.001, 0.0001, 10, 100]

trainFileData=[]
trainFileLabel=[]

if('-fold' in sys.argv):
	fold_index = sys.argv.index('-fold')
if('-test' in sys.argv):
	test_index = sys.argv.index('-test')

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
[XTest, YTest] = svm_func.parseInfo(testFileData, testFileLabel)
dsize = len(XTest[0])
# print dsize
# print numpy.dot(XTest[0],XTest[3])


def Run_Q3_1():
	wvecLearn = svm_func.SVM(XData, YData, 1, 0.001, 50, len(XData[0]))
	#print wvecLearn
	TestStruct = svm_func.SVM_TEST(XTest, YTest, wvecLearn)
	TrainingStruct = svm_func.SVM_TEST(XData, YData, wvecLearn)
	## print "Test-Set size = ", len(XTest)
	## print "Data-Set Size = ", len(XData)
	## print "Test-MistakeCount = ", TestStruct[3]
	## print "Test-Accuracy = ", 100*float(len(XTest) - TestStruct[3])/len(XTest), "%"
	## print "Training-MistakeCount = ", TrainingStruct[3]
	## print "Training-Accuracy = ", 100*float(len(XData) - TrainingStruct[3])/len(XData), "%\n\n"
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


def Run_Q3_2():
	CGTrainAcc = []
	CGTestAcc = []
	bestC = 0
	bestG = 0
	MaxAccuracy = 0
	for c in CList:
		GammaTestAcc = []
		GammaTrainAcc = []
		for g in GList:
			blockSize = len(XData)/foldValue
			testAcc = 0
			trainAcc = 0
			for k in range(0,foldValue):
				print "C = G = ", c, g
				NewXTest = XData[k*blockSize: (k+1)*blockSize]
				NewYTestLabel = YData[k*blockSize: (k+1)*blockSize]
				NewXData = [XData[i] for i in range(0,len(XData)) if(i<k*blockSize or i>=(k+1)*blockSize)]
				NewYLabel = [YData[i] for i in range(0,len(XData)) if(i<k*blockSize or i>=(k+1)*blockSize)]
				wvecLearn = svm_func.SVM(NewXData, NewYLabel, c, g, 50, len(NewXData[0]))
				##TestStruct = [TruePos, FalsePos, FalseNeg, mistakeCounter]
				TrainingStruct = svm_func.SVM_TEST(NewXData,NewYLabel, wvecLearn)
				TestStruct = svm_func.SVM_TEST(NewXTest, NewYTestLabel, wvecLearn)
				testAcc = testAcc + (float(len(NewXTest) - TestStruct[3])/len(NewXTest))*100
				trainAcc = trainAcc + (float(len(NewXData) - TrainingStruct[3])/len(NewXData))*100
			avgtrainacc = float(trainAcc)/foldValue
			#print "avgtrain = ", avgtrainacc
			avgtestacc = float(testAcc)/foldValue
			#print "avgtest = ", avgtestacc
			GammaTrainAcc.append(avgtrainacc)
			GammaTestAcc.append(avgtestacc)
			if(MaxAccuracy < avgtestacc):
				MaxAccuracy = avgtestacc
				bestC = c
				bestG = g
			#print "Current-C =", c, "Current-G =", g
			#print "Best-C = ", bestC, " Best-G = ", bestG, "\n\n"
		CGTrainAcc.append(GammaTrainAcc)
		CGTestAcc.append(GammaTestAcc)

	print bestC, bestG
	print CGTrainAcc
	
	##------- Learn with the best fit parameters ---------####
	wvecLearn = svm_func.SVM(XData, YData, bestC, bestG, 50, len(XData[0]))

	##------- Initiate Testing -------------##
	TrainingStruct = svm_func.SVM_TEST(XData, YData, wvecLearn)
	TestStruct = svm_func.SVM_TEST(XTest, YTest, wvecLearn)
	print TrainingStruct[3], len(XTest)
	print "========== Report for Q3_2 (k=5) fold validation ================="
	for c in range(0,len(CList)):
		for g in range(0,len(GList)):
			print "======== C = ",CList[c],", Gamma = ",GList[g], ",=================="
			print "Average Training Accuracy for pair(c=",CList[c],",g=",GList[g],") Accuracy = ", CGTrainAcc[c][g],"%"
			print "Average Test Accuracy for pair(c=",CList[c] ,",g=", GList[g], ") Accuracy = ", CGTestAcc[c][g],"%\n\n"
##	print "Average Test Accuracy = ", (len(XTest) - mainTestMistake)/(len(XTest)))
	print "Best-C = ", bestC
	print "Best-Gamma = ", bestG
	print "len-XTest", len(XTest)

	##----- Report for the Test on the training data -----##
	PrecisionScoreTrain = TrainingStruct[0]/float(TrainingStruct[0] + TrainingStruct[1])
	RecallTrain = TrainingStruct[0]/float(TrainingStruct[0] + TrainingStruct[2])
	F1Train = (2*PrecisionScoreTrain*RecallTrain)/float(PrecisionScoreTrain + RecallTrain)
	print "Training-Mistake Count: ", TrainingStruct[3]
	print "Training-Accuracy = ",  ((len(XData) - TrainingStruct[3])/float((len(XData))))*100 , "%"
	print "Training - Precision Score = ", PrecisionScoreTrain
	print "Training - Recall Score = ", RecallTrain
	print "Training - F1-Score = ", F1Train

	##----- Report For the Test Data -----##
	PrecisionScoreTest = TestStruct[0]/float(TestStruct[0] + TestStruct[1])
	Recalltest = TestStruct[0]/float(TestStruct[0] + TestStruct[2])
	F1test = (2*PrecisionScoreTest*Recalltest)/float(PrecisionScoreTest + Recalltest)
	print "Test-Mistake Count: ", TestStruct[3]
	print "Test-Accuracy= ",  ((len(XTest) - TestStruct[3])/(float((len(XTest)))))*100, "%"
	print "Test - Precision Score = ", PrecisionScoreTest 
	print "Test - Recall Score = ", Recalltest
	print "Test - F1-Score = ", F1test
	print "========= End Report =============================="

	
				
	test_index = sys.argv.index('-test')
if('-q31' in sys.argv):
	Run_Q3_1()
elif('-q32' in sys.argv):
	Run_Q3_2()
else:
	print "Incorrect Parameter combinations!!... Exiting!\n"
	sys.exit()
