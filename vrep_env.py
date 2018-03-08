import vrep
import numpy as np
import random 
import time

class Environment:

	def __init__(self):
		# Launch the simulation with the given launchfile name
		self.action_space = [i for i in range(3)] #F,L,R
		self.reward_range = (-np.inf, np.inf)
		
		self.sensors = 3
		self.sensor_h=[0,0,0]
		
		vrep.simxFinish(-1) # just in case, close all opened connections
		self.clientID=vrep.simxStart('127.0.0.1',20000,True,True,5000,5)


		if self.clientID!=-1:  #check if client connection successful
			print ('Connected to remote API server, clientID',self.clientID)

		else:
			print( 'Connection not successful')
			sys.exit('Could not connect')

	
	def set_vel(self,vl,vr):
		errorCode=vrep.simxSetJointTargetVelocity(self.clientID,self.left_motor_handle,vl, vrep.simx_opmode_streaming)
		errorCode=vrep.simxSetJointTargetVelocity(self.clientID,self.right_motor_handle,vr, vrep.simx_opmode_streaming)


	def get_sensor_data(self):

		sensor_val=np.array([])  
		for x in range(1,self.sensors + 1):
			errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(self.clientID,self.sensor_h[x-1],vrep.simx_opmode_buffer)                
			if detectionState == True :
				sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
			else:
				sensor_val=np.append(sensor_val,np.inf)
			# print("SENSOR data " ,errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector)
				
		return sensor_val

	def get_observation(self):

		sensorDistance = self.get_sensor_data()

		discretized_ranges = np.empty((0,3))
		min_range = 0.11
		done = False
	   	
		#  convert sensor values to descrete values for state encoding
		for i, item in enumerate(sensorDistance):
			# if item < 0.25:
			# 	discretized_ranges= np.append(discretized_ranges, 0)
			# elif(item < 0.5):
			# 	discretized_ranges= np.append(discretized_ranges, 1)
			# elif(item < 0.75):
			# 	discretized_ranges= np.append(discretized_ranges, 2)
			if item < 0.35:
				discretized_ranges= np.append(discretized_ranges, round(item,2))
			else:
				discretized_ranges= np.append(discretized_ranges, 0.35)

			if (min_range > item > 0):
				# print('Done = True , Sensor = ',i, 'val = ',item)
				done = True
		print("state = ",discretized_ranges)
		return discretized_ranges,done

	def init_sensors(self):

		# _,robot = vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',vrep.simx_opmode_oneshot_wait)
		# ret_val = vrep.simxSetObjectOrientation(self.clientID,robot ,robot, [0,0,random.choice([-0.05,0,0.05])],vrep.simx_opmode_oneshot)
		
		errorCode,self.left_motor_handle = vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
		errorCode,self.right_motor_handle = vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)
		# print(errorCode, " ++++++++++++++++++++++", self.left_motor_handle)

		for x in range(1,self.sensors + 1):
			errorCode,sensor_handle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
			self.sensor_h[x-1] = sensor_handle  
			

		sensor_val=np.array([])  
		for x in range(1,self.sensors + 1):
			errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(self.clientID,self.sensor_h[x-1],vrep.simx_opmode_streaming)                
			if detectionState == True :
				sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
			else:
				sensor_val=np.append(sensor_val,np.inf)


	def reset(self):
		
		sensorDistance = self.init_sensors()
		# vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)
		# ret_val = vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_oneshot_wait)
		message = 0
		count = 0
		while ((message &1) == 0):
			count+=1
			print("SIMULATION STOPPED")
			result,message = vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state )
			ret_val = vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_oneshot_wait)

			print("message = ",message)

		# if count != 3:
		# 	print("reset")
		# 	return count
			
		print("SIMULATION STARTED ",count)
		
		
		start_state,done = self.get_observation(sensorDistance)

		return start_state

	def stop(self):
		error_code = vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_blocking)
	
	def load_scene(self):
		# error_code = vrep.simxLoadScene(self.clientID,'MyScenes/square_demo'+str(random.choice([1,2]))+'.ttt',0xFF,vrep.simx_opmode_blocking)
		error_code = vrep.simxLoadScene(self.clientID,'MyScenes/primitive_test.ttt',0xFF,vrep.simx_opmode_blocking)

	def start(self):
		error_code = vrep.simxStartSimulation(self.clientID,vrep.simx_opmode_oneshot_wait)



