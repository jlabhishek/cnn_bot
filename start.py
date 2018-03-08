import environment
import vrep_dqn
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
EPISODES = 4000
STEPS = 100

# PATHS
save_path = 'save_model/'
file_count= len([name for name in os.listdir(save_path) ])


q_val = 'Files/q_val_table.txt'
action_val = 'Files/q_action_table.txt'
# rewardFile = 'Files/rewards.pickle'
stepFile = 'Files/steps.pickle'

def start_simulation():
	state_size = 3
	action_size = 3
	hiddenLayers = [6,5]
	activation_function = "sigmoid"

	agent = vrep_dqn.DQNAgent(state_size,action_size,hiddenLayers,activation_function)
	env = environment.Env()

	scores, episodes, stepList = [], [], []

	if file_count != 0:
		agent.model.load_weights(save_path+str(file_count)+".h5")

	for e in range(EPISODES):
		done = False
		score = 0
		state,_ = env.reset()
		state = np.reshape(state, [1, state_size])
		# time.sleep(10)
		for step in range(STEPS):
			# if agent.render:
			#     env.render()

			# get action for the current state and go one step in environment
			action = agent.get_action(state)
			
			next_state, reward, done, info = env.step(action)
			next_state = np.reshape(next_state, [1, state_size])

			# if an action make the episode end, then gives penalty of -1
			reward = reward if not done else -1
			# reward = reward if not done or steps == 499 else -1

			# save the sample <s, a, r, s'> to the replay memory
			agent.append_sample(state, action, reward, next_state, done)
			# every time step do the training
			agent.train_model()
			score += reward
			state = next_state

			if done:
				break

			time.sleep(0.4)

		# every episode update the target model to be same with model
		agent.update_target_model()

		# every episode, plot the play time
		print(step,"step")
		stepList.append(step)

		scores.append(score)
		episodes.append(e)
		# pylab.plot(episodes, scores, 'b')
		# pylab.savefig("./save_graph/cartpole_dqn.png")
		print("episode:", e, "  score:", score, "  memory length:",
			  len(agent.memory), "  epsilon:", agent.epsilon)

		# if the mean of scores of last 10 episode is bigger than 490
		# stop training
		# if np.mean(scores[-min(10, len(scores)):]) > 490:
		# 	sys.exit()

	# save the model
		if (e+1) % 50 == 0:
			agent.model.save_weights(save_path+str(file_count + 1 + (e+1)//50 )+'.h5')
			agent.saveQValues((e+1),q_val,state_size)
			agent.saveQActions((e+1),action_val,state_size)
			agent.saveWeights((e+1))

			stepList = pickle.load(open(stepFile, 'rb'))  + stepList 
			pickle.dump(stepList, open(stepFile, 'wb'))

			stepList = []






if __name__ == '__main__':
	start_simulation()




