import numpy as np
import random
from collections import deque
import keras 
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers import Dense,Dropout,Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
from keras import initializers


class DQNAgent:
	def __init__(self, state_size, action_size,hiddenLayers,activation_function):
	 

		# get size of state and action
		self.state_size = state_size
		self.action_size = action_size

		# These are hyper parameters for the DQN
		self.hiddenLayers = hiddenLayers
		self.activationType = activation_function
		self.discount_factor = 0.99
		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = 0.9992
		self.epsilon_min = 0.01
		self.batch_size = 32
		self.train_start = 1000
		# create replay memory using deque
		self.memory = deque(maxlen=2000)

		# create main model and target model
		self.model = self.build_model(self.hiddenLayers,self.activationType)
		self.target_model = self.build_model(self.hiddenLayers,self.activationType)
		print(self.model,"5555555555555555")
		# initialize target model
		# self.update_target_model()




	# approximate Q function using Neural Network
	# state is input and Q Value of each action is output of network


	def build_model(self, hiddenLayers, activationType):
		model = Sequential()
		if len(hiddenLayers) == 0: 
			model.add(Dense(self.action_size, input_dim=self.state_size)  ) # model.add(Dense(self.output_size, input_shape=(self.state_size,))  ) #
			model.add(Activation("linear"))
		else :
			model.add(Dense(hiddenLayers[0], input_dim = self.state_size) )
				
			for index in range(1, len(hiddenLayers)):
				
				layerSize = hiddenLayers[index]
				model.add(Dense(layerSize))
				model.add(Activation(self.activationType))

			model.add(Dense(self.action_size))
			model.add(Activation("linear"))
		
		# optimizer = optimizers.RMSprop(lr=self.learningRate, rho=0.9, epsilon=1e-06)
		optimizer = optimizers.SGD(lr=self.learning_rate, clipnorm=1.)
		# optimizer = optimizers.Adam(lr=self.learning_rate)
		
		model.summary()

		model.compile(loss="mse", optimizer=optimizer)


	# after some time interval update the target model to be same with model
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	# get action from model using epsilon-greedy policy
	def get_action(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			q_value = self.model.predict(state)
			return np.argmax(q_value[0])

	# save sample <s,a,r,s'> to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	# pick samples randomly from replay memory (with batch_size)
	def train_model(self):
		if len(self.memory) < self.train_start:
			return
		batch_size = min(self.batch_size, len(self.memory))
		mini_batch = random.sample(self.memory, batch_size)

		update_input = np.zeros((batch_size, self.state_size))
		update_target = np.zeros((batch_size, self.state_size))
		action, reward, done = [], [], []

		for i in range(self.batch_size):
			update_input[i] = mini_batch[i][0]
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			update_target[i] = mini_batch[i][3]
			done.append(mini_batch[i][4])

		target = self.model.predict(update_input)
		target_val = self.target_model.predict(update_target)

		for i in range(self.batch_size):
			# Q Learning: get maximum Q value at s' from target model
			if done[i]:
				target[i][action[i]] = reward[i]
			else:
				target[i][action[i]] = reward[i] + self.discount_factor * (
					np.amax(target_val[i]))

		# and do the model fit!
		self.model.fit(update_input, target, batch_size=self.batch_size,
					   epochs=1, verbose=1)


	def saveQValues(self,episode_num,file_path):
		f = open(file_path,'a')
		print( '\nQ VALUES for ',episode_num,file = f)
		state = np.array([0,0,0])
		for i in range (13,100,30):
			for j in range(13,100,30):
				for k in range(13,100,30):
					state = np.array([i/100,j/100,k/100])
		
					predicted = self.model.predict(state.reshape(1,len(state)))
					print(state,' -> ', predicted[0],file = f)

		f.close()

	def saveQActions(self,episode_num,file_path):
		f = open(file_path,'a')
		print( '\nQ VALUES for ',episode_num,file = f)
		state = np.array([0,0,0])
		for i in range (13,100,30):
			for j in range(13,100,30):
				for k in range(13,100,30):
					state = np.array([i/100,j/100,k/100])
					q_value = self.model.predict(state)
					print(state,' -> ', np.argmax(q_value[0]),file = f)

		f.close()