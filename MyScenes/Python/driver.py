import vrep                  #V-rep library
import sys
import time                #used to keep track of time
import numpy as np   
import matplotlib.pyplot as plt
import math 
import environment
import qlearn
import pickle

PI = math.pi

if __name__ == '__main__':
	# Establish Communication
	

	last_time_steps = np.ndarray(0)
	environment = environment.Environment()
	qlearn = qlearn.QLearn(actions=range(len(environment.action_space)),
					alpha=0.2, gamma=0.8, epsilon=0.9)

	initial_epsilon = qlearn.epsilon

	epsilon_discount = 0.9986

	start_time = time.time()
	total_episodes = 10000
	highest_reward = 0
	

	f = open('q_table.txt','a')
	f2 = open('q_table_list.pickle','wb')
	for x in range(total_episodes):
		done = False

		cumulated_reward = 0 #Should going forward give more reward then L/R ?

		observation =environment.reset()

		if qlearn.epsilon > 0.05:
			qlearn.epsilon *= epsilon_discount

		state = ''.join(map(str, observation))
		# print("State = ",state," observation = ",observation)
		for i in range(1500):

			# Pick an action based on the current state
			action = qlearn.chooseAction(state)

			# Execute the action and get feedback
			observation, reward, done, info = environment.step(action)
			cumulated_reward += reward

			if highest_reward < cumulated_reward:
				highest_reward = cumulated_reward

			nextState = ''.join(map(str, observation))

			qlearn.learn(state, action, reward, nextState)

			# environment.monitor.flush(force=True)
			print(i," S= ", state, " A = ",action, 'observation = ',observation)
			if not(done):
				state = nextState
			else:
				last_time_steps = np.append(last_time_steps, [int(i + 1)])
				print('done')
				break

			time.sleep(0.5)

		print("EP: "+str(x+1)+" Done: ",done,"\nQ Table\n",sorted(qlearn.q.items()),file = f)
		pickle.dump(qlearn.q, f2)
		# print(qlearn.q)
		


		m, s = divmod(int(time.time() - start_time), 60)
		h, m = divmod(m, 60)
		print ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

	#Github table content
	print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |")

	l = last_time_steps.tolist()
	l.sort()

	#print("Parameters: a="+str)
	print("Overall score: {:0.2f}".format(last_time_steps.mean()))
	print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))


