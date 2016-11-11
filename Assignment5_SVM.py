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

CList = [0.01, pow(2,1), pow(2,-1), pow(2,-2), pow(2,-4), pow(2,2)]
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
print dsize
print numpy.dot(XTest[0],XTest[3])


def Run_Q3_1():
	wvecLearn = svm_func.SVM(XData, YData, 1, 0.001, 50, len(XData[0]))
	#print wvecLearn
	mistakeCount = svm_func.SVM_TEST(XTest, YTest, wvecLearn)
	mistakeCountTrain = svm_func.SVM_TEST(XData, YData, wvecLearn)
	print "MistakeCount = ", mistakeCount
	print "MistakeCountTrain = ", mistakeCountTrain
	print "TestSize = ", len(XTest)
	print "DataSize = ", len(XData)


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
				trainMistake = svm_func.SVM_TEST(NewXData,NewYLabel, wvecLearn)
				testMistake = svm_func.SVM_TEST(NewXTest, NewYTestLabel, wvecLearn)
				print 'TrainMistake = ', trainMistake
				testAcc = testAcc + (float(len(NewXTest) - testMistake)/len(NewXTest))*100
				trainAcc = trainAcc + (float(len(NewXData) - trainMistake)/len(NewXData))*100
			avgtrainacc = float(trainAcc)/foldValue
			print "avgtrain = ", avgtrainacc
			avgtestacc = float(testAcc)/foldValue
			print "avgtest = ", avgtestacc
			GammaTrainAcc.append(avgtrainacc)
			GammaTestAcc.append(avgtestacc)
			if(MaxAccuracy < avgtestacc):
				MaxAccuracy = avgtestacc
				bestC = c
				bestG = g
			print "Current-C =", c, "Current-G =", g
			print "Best-C = ", bestC, " Best-G = ", bestG, "\n\n"
		CGTrainAcc.append(GammaTrainAcc)
		CGTestAcc.append(GammaTestAcc)

	print bestC, bestG
	print CGTrainAcc
	
	##------- Learn with the best fit parameters ---------####
	wvecLearn = svm_func.SVM(XData, YData, bestC, bestG, 50, len(XData[0]))

	##------- Initiate Testing -------------##
	mainTrainMistake = svm_func.SVM_TEST(XData, YData, wvecLearn)
	mainTestMistake = svm_func.SVM_TEST(XTest, YTest, wvecLearn)
	print mainTrainMistake, len(XTest)
	print "========== Report for Q3_2 (k=5) fold validation ================="
	for c in range(0,len(CList)):
		for g in range(0,len(GList)):
			print "======== C = ",CList[c],", Gamma = ",GList[g], ",=================="
			print "Average Training for pair(c,g) Accuracy = ", CGTrainAcc[c][g],"%"
			print "Average Test for pair(c,g) Accuracy = ", CGTestAcc[c][g],"%\n\n"
##	print "Average Test Accuracy = ", (len(XTest) - mainTestMistake)/(len(XTest)))
	print "Best-C = ", bestC
	print "Best-Gamma = ", bestG
	print "len-XTest", len(XTest)
	print "Training-Accuracy = ",  ((len(XData) - mainTrainMistake)/float((len(XData))))*100 , "%"
	print "mainTrainMistake: ", mainTrainMistake
	print "mainTestMistake: ", mainTestMistake
	print "Test-Accuracy= ",  ((len(XTest) - mainTestMistake)/(float((len(XTest)))))*100, "%"
	print "========= End Report =============================="

	
				

#Run_Q3_1()
Run_Q3_2()
