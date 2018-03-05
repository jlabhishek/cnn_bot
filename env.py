import vrep_env

class Env:

	def __init__(self):
		env = vrep_env()

	def step(self, action):
		

		#  set velocity of left and right motor
		v=0.5   #forward velocity
		# can make this part more accurate
		kp=0.5  #steering gain
		steer = 0.5
		if action == 0: #FORWARD
			vl = v
			vr = v
		elif action == 1: #LEFT
			vl=v-kp*steer
			vr=v+kp*steer
		elif action == 2: #RIGHT
			vl=v+kp*steer
			vr=v-kp*steer

		#Set velocity of robot
		vrep_env.set_vel(cl,vr)

		#Get observation from sensors/images
		ob,done = vrep_env.get_observation()

		reward = 0.1 if action == 0 else 0.02   # action 0 is forward, other actions are left and right


		return ob, reward, done, {}

	def reset(self):
		env.stop()
		env.load_scene()
		env.init_sensors()
		env.start()
		# env.pause()
		#env.remove_bot()
		return get_observation()

	def render(self, mode='human', close=False):
		# write code to turf off display from code, google , there is a paramter for this
		pass

	def take_action(self, action):
		pass

	def get_reward(self):
		""" Reward is given for XY. """
		if self.status == FOOBAR:
			return 1
		elif self.status == ABC:
			return self.somestate ** 2
		else:
			return 0