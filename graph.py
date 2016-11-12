
import sys
import re
import copy
import dt_func
import svm_func
import math
import numpy

class graph:
	def __init__ (self, Name, Type, Examples, Attr, CurrentDepth, limitDepth):
		self.name = Name
		self.Type = copy.deepcopy(Type)
		self.examples = copy.deepcopy(Examples)
		self.attr = Attr
		self.depth = copy.deepcopy(CurrentDepth)
		self.decision = dicy([])
		if(Type == 'leaf'):
			self.decision[self.name] = []
