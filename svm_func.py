import sys
import re
import copy
import math
import numpy

##---- Trims a string towards the left
def TrimLeft( trimString ):
	while( re.match(' ', trimString)):
		trimString = trimString[1:]
	return trimString


##--- concats elements of list and returns a string ---##
def concatList(l1):
	l2 = ''
	if(len(l1)!=0):
		for i in l1:
			l2 = l2+i
	return l2


##--- trims all spaces from a string
def trimStr( str1 ):
	l2 = ''
	for i in str1:
		if(i!=' '):
			l2 = l2+i
	return l2

def trimList( list1 ):
	l2 =[]
	for i in list1:
		if(i != '' and i!='\n'):
			l2.append(i)
	return l2

def sgn(value, margin):
	if( value >= margin):
		return 1
	else:
		return -1

def permuteDataLabel( xdata, ylabel):
	newdata = []
	newlabel = []
	shufIdx = numpy.random.permutation(len(xdata))
	if (len(xdata) != len(ylabel)):
		print 'Error...........Mismatch in data and label size.... Exiting!'
		sys.exit()
	
	for i in shufIdx:
		newdata.append(xdata[i])
		newlabel.append(ylabel[i])
	
	return [newdata, newlabel]

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


def SVM (xdata, ydata, c, gamma, epochs, vecSize):
	##--- initialise the required data ---
	wvec = [0.0]*vecSize
	#wvec = numpy.random.normal(0,0.01,vecSize)	
	count = 0
	for t in range(0,epochs):
		[xdata, ydata] = permuteDataLabel(xdata, ydata)
		for i in range(0,len(xdata)): ## looping over each example
			xvec = xdata[i]
			wtxSum = numpy.dot(wvec, xvec)
			count = count+1
			gammaT = gamma/(1 + (float(gamma)/c)*count) 
			if ( float(ydata[i]*wtxSum) <= 1 ):
				wvec = [(wvec[j]*(1-gammaT) + gammaT*c*ydata[i]*xvec[j]) for j in range(0,len(wvec))]
			else:
				wvec = [wvec[j]*(1-gammaT) for j in range(0,len(wvec))]
	return wvec


def SVM_TEST( xdata, ydata, wvec ):
	mistakeCounter = 0
	wtxSum = 0
	TruePos = 0.0
	FalsePos = 0.0
	FalseNeg = 0.0
	for i in range(0,len(xdata)):
		xvec = xdata[i]
		ylabel = ydata[i]
		#print ylabel*numpy.dot(wvec,xvec)
		if( numpy.dot(wvec, xvec)*ylabel < 0 ):
			mistakeCounter = mistakeCounter+1
			if ( numpy.dot(wvec, xvec) < 0 and ylabel >= 0): ## False negative
				FalseNeg = FalseNeg + 1.0
			elif ( numpy.dot(wvec, xvec) >= 0 and ylabel < 0): ## False positive
				FalsePos = FalsePos + 1
		else:
			if(numpy.dot(wvec,xvec) >=0 and ylabel >=0): #true Positive
				TruePos = TruePos + 1.0 
	return [TruePos, FalsePos, FalseNeg, mistakeCounter]
		
