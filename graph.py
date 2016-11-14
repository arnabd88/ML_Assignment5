
import sys
import re
import copy
import dt_func
import svm_func
import math
import numpy

class graph:
	def __init__ (self, Name, Type, Examples, Attr, CurrentDepth, limitDepth, kfeatures):
		self.name = Name
		self.Type = copy.deepcopy(Type)
		self.examples = copy.deepcopy(Examples)
		self.attr = Attr
		self.depth = copy.deepcopy(CurrentDepth)
		self.decision = dict([])
		self.limitDepth = limitDepth
		self.kfeatures = kfeatures
		if(Type == 'leaf'):
			self.decision[self.name] = []

	def ExtractAttrSubset(self, attr):
		subAttr = copy.deepcopy(attr)
		if(self.name in attr):
			del subAttr[self.name]
			subAttr['_AttrOrder_'].remove(self.name)
		return subAttr


	def ExtractExSubset(self, examples, value):
		subExample = dict([])
		for j in examples.keys():
			if(j!=self.name):
				subExample[j] = []
				for i in range(0,len(examples[self.name])):
					if( examples[self.name][i]==value):
						subExample[j].append(examples[j][i])
		return subExample

	def getCommonLabel(self, examples):
		length = len(examples['Result'])
		res = self.attr['classes']
		lc0 = 0
		lc1 = 1
		for i in examples['Result']:
			if(i==res[0]):
				lc0 = lc0+1.0
			else:
				lc1 = lc1+1.0
		#print "Reached here atleast"
		if(lc0 >= lc1):
			return res[0]
		else:
			return res[1]


	def ID3(self):
		res = self.attr['classes']
		if (not( res[0] in self.examples['Result'] and res[1] in self.examples['Result'])):
			if(res[0] in self.examples['Result']):
				self.decision[res[0]] = []
			else:
				self.decision[res[1]] = []
			self.Type = 'leaf'
			return 1
		else:
			#print "Name: ", self.name
			self.Values = self.attr[self.name]
			subAttr = self.ExtractAttrSubset(self.attr)
			#print "Attr:", self.attr
			#print "SubAttr: ", subAttr
			for v in self.Values:
				#print "Examples: " , self.examples
				subExample = self.ExtractExSubset(self.examples, v)
				#print "subExample: ", subExample
				if(len(subExample['Result'])==0 ): #no input available
					MaxLabel = self.getCommonLabel(self.examples)
					#print "MaxLabel: ", MaxLabel
					gnext = graph(MaxLabel, 'leaf', subExample, subAttr, self.depth+1, self.limitDepth, self.kfeatures)
					self.decision[v] = [MaxLabel, gnext]
				elif(len(subAttr['_AttrOrder_'])==0):
					MaxLabel = self.getCommonLabel(subExample)
					#print "MaxLabel: ", MaxLabel
					gnext = graph(MaxLabel, 'leaf', subExample, subAttr, self.depth+1, self.limitDepth, self.kfeatures)
					self.decision[v] = [MaxLabel, gnext]

				else:
					nextNode = dt_func.decideRoot(subExample, subAttr, self.kfeatures)
					if(self.depth+1 == self.limitDepth):
						MaxLabel = self.getCommonLabel(subExample)
						#print "MaxLabel = ", maxLabel
						gnext = graph(maxLabel, 'leaf', subExample, subAttr, self.depth+1, self.limitDepth, self.kfeatures)
						self.decision[v] = [MaxLabel, gnext]
					else:
						#print "nextNode = ", nextNode
						gnext = graph(nextNode, 'internal' , subExample, subAttr, self.depth+1, self.limitDepth, self.kfeatures)
						self.decision[v] = [nextNode, gnext]
						succ = gnext.ID3()
		return 1


	def predictResult(self, testV):
		if( self.Type=='leaf' and len(self.decision.keys())==1 ):
			return self.decision.keys()[0]
		else :		
			return self.decision[testV[self.name]][1].predictResult(testV)
		
