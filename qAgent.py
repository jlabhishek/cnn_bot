import qEnvironment
import deepQ
import time
import h5py
import os, os.path
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random

'''
	Agents : Takes Action in the environment,sets up simulation(Episodes, step size)
	Environment : sets up the environment(v-rep simulation), returns rewards and next states to agent based on the actions it took
	Algorithm :  The Learning Algorithm
'''
class Agent:
	def __init__(self):
		self.epsilon = 0.25
		self.inputs = 3 #state
		self.outputs = 3 #action
		self.dicountFactor = 0.8
		self.learningRate = 0.25
		self.savePath='pioneer_qlearn_deep/ep'
		self.network_layers = [4]
		self.graphPath = 'Images/'
		self.rewardFile = 'Files/rewards.pickle'
		self.stepFile = 'Files/steps.pickle'

		self.dict = {}

	def start(self):
		

		file_count= len([name for name in os.listdir('pioneer_qlearn_deep') ])
		# print("FILE",file_count)
		weights_path = 'pioneer_qlearn_deep/ep'+str(file_count)+'.h5'
		deepQLearn = deepQ.NNQ(self.inputs,self.outputs,self.dicountFactor,self.learningRate)
		deepQLearn.initNetworks(self.network_layers)
		
		#deepQLearn.plotModel('/home/kaizen/BTP/Python/NeuralNet/Images/')		

		if file_count != 0:
			deepQLearn.loadWeights(weights_path)
			print("Weights Loaded from path ",weights_path,"\n")

		env = qEnvironment.Environment()


		num_episodes = 10000
		steps = 200
		start_time = time.time()

		#  for plotting,data per episode
		stepList = []
		rewardList = []
		index = 1
		
		replay = []  # stores tuples of (S, A, R, S').
		total_steps_in_simulation = 0
		observe = 500
		max_buffer_size = 10000 # average 30 steps, 10000/30 = 333 episodes 
		batch_size = 100

		for episode in range(num_episodes):
			
			state = env.reset()
			while type(state) ==int:
				state = env.reset()
			cumulated_reward = 0
			
			for step in range(steps):

				print("state",state,end="" )

				qValues = deepQLearn.getQValues(state)
				action = deepQLearn.selectAction(qValues, self.epsilon )
				self.dict[''.join(str(e) for e in state)] = action

				nextState,reward,done,info = env.step(action)
				cumulated_reward += reward

				#  LEARNING PART
				replay.append((state,action,reward,nextState,done))
				if total_steps_in_simulation > observe:

					if len(replay) > max_buffer_size:
						replay.pop(0)

					training_set = random.sample(replay,observe)

					
					deepQLearn.learn_on_minbatch(training_set,batch_size)
					# deepQLearn.learn_on_one_example(state,action,reward,nextState,done,batch_size = batch_size)

				if not(done):
					state = nextState
				else:
					print('done')
					break
				print("Step = ",step)

				#  Average time per step = 0.004s

				total_steps_in_simulation += 1

				time.sleep(0.8)
				
				
			stepList.append(step)
			rewardList.append(cumulated_reward)


			m, s = divmod(int(time.time() - start_time), 60)
			h, m = divmod(m, 60)
			print ("\n\n\EP "+str(episode+1)+" Reward: "+ str(cumulated_reward)  +" Time: %d:%02d:%02d" % (h, m, s))                   
			# time.sleep(0.5)

			if (episode +1)%50 == 0:
				print(replay)

				rewardList = pickle.load(open(self.rewardFile, 'rb'))  + rewardList
				stepList = pickle.load(open(self.stepFile, 'rb'))  + stepList 
				print(len(rewardList))
				#print(stepList)

				pickle.dump(rewardList, open(self.rewardFile, 'wb'))
				pickle.dump(stepList, open(self.stepFile, 'wb'))

				''' COMMMENT THESE IF TESTING ANYTHING TO PREVENT DATA DAMAGE'''

				deepQLearn.saveModel(self.savePath+str(file_count + index)+'.h5')
				index = index + 1
				
				deepQLearn.saveQValues(episode)
				deepQLearn.saveWeights(episode)

				stepList = []
				rewardList = []
				print(self.dict)
				
				# print values
				f = open('Files/deep_q_table.txt','a')
				print(episode + 1,file = f)


				print_state = np.empty((0,3), int)
				print_state = np.append(print_state,[0,0,0])

				for i in range(4):
					print_state[0] = i
					for j in range(4):
						print_state[1] = j
						for k in range(4):
							print_state[2] = k
							qValues = deepQLearn.getQValues(print_state)
							action = deepQLearn.selectAction(qValues, 0 )# no randomness 
							print(print_state , " ", action, file = f)
		
				f.close()

if __name__ == '__main__':
	agent = Agent()
	agent.start()
