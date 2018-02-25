import numpy as np
import random
import keras 
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers import Dense,Dropout,Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
from keras import initializers


class ConvNet:
	def __init__(self,inputs,outputs,gamma )
		self.input_size = inputs
		self.output_size = outputs
		self.gamma = gamma

	def initNetworks(self, hiddenLayers):
		model = self.createModel(self.input_size, self.output_size, hiddenLayers, "sigmoid")
		self.model = model