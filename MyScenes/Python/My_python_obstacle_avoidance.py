import vrep                  #V-rep library
import sys
import time                #used to keep track of time
import numpy as np   
import matplotlib.pyplot as plt
import math 

PI = math.pi

# Establish Communication
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19998,True,True,5000,5)
_,robot = vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx',vrep.simx_opmode_oneshot_wait)
# vrep.simxResetDynamicObject(self.clientID,robot)
vrep.simxRemoveModel(clientID,robot,vrep.simx_opmode_oneshot_wait)


a,b=vrep.simxLoadModel(clientID, '/home/rip/BTP/V-REP_PRO_EDU_V3_4_0_Linux/models/Custom_model/Pioneer_p3dx.ttm',0, vrep.simx_opmode_blocking)


if clientID!=-1:  #check if client connection successful
	print ('Connected to remote API server')

else:
	print( 'Connection not successful')
	sys.exit('Could not connect')


errorCode,left_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
errorCode,right_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)
errorCode,robot=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx',vrep.simx_opmode_oneshot_wait)

code,collision = vrep.simxReadCollision(clientID,robot ,vrep.simx_opmode_streaming )

# Preallocaation
sensor_h=[] #empty list for handles
sensor_val=np.array([]) #empty array for sensor measurements

#orientation of all the sensors: 
sensor_loc=np.array([-PI/2, -50/180.0*PI,-30/180.0*PI,-10/180.0*PI,10/180.0*PI,30/180.0*PI,50/180.0*PI,PI/2,PI/2,130/180.0*PI,150/180.0*PI,170/180.0*PI,-170/180.0*PI,-150/180.0*PI,-130/180.0*PI,-PI/2]) 


errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0.2, vrep.simx_opmode_streaming)
errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0.2, vrep.simx_opmode_streaming)

#for loop to retrieve sensor arrays and initiate sensors
for x in range(1,16+1):
	errorCode,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
	sensor_h.append(sensor_handle) #keep list of handles        
	errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handle,vrep.simx_opmode_streaming)                
	sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values

# t = time.time()    

#Post ALlocation
# errorCode,ultrasonic_sensor_1=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor1',vrep.simx_opmode_oneshot_wait)


# returnCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector=vrep.simxReadProximitySensor( clientID, ultrasonic_sensor_1, vrep.simx_opmode_streaming)

# errorCode,cam_handle=vrep.simxGetObjectHandle(clientID,'cam1',vrep.simx_opmode_oneshot_wait)
# # print(cam_handle	)
# returnCode,resolution, image=vrep.simxGetVisionSensorImage(clientID, cam_handle, 0,vrep.simx_opmode_streaming)
# time.sleep(1)   # sleep is necessary otherwise there is some difficulty in communication and the robot does not reveice value for 'image' just after this command
# returnCode,resolution, image=vrep.simxGetVisionSensorImage(clientID, cam_handle, 0,vrep.simx_opmode_streaming)

# while True:

# 	# returnCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector=vrep.simxReadProximitySensor( clientID, ultrasonic_sensor_1, vrep.simx_opmode_buffer)
# 	returnCode,resolution, image=vrep.simxGetVisionSensorImage(clientID, cam_handle, 0,vrep.simx_opmode_buffer)

# 	im = np.array(image,dtype =np.uint8)
# 	im.resize([resolution[0],resolution[1 ],3])
# 	plt.imshow(im,origin='lower')
# 	if np.random.randint(1000) > 600:
# 		plt.show()

t = time.time()
while (time.time()-t)<1000:
	#Loop Execution
	#  detected point wull have coordinates nan if there is no obstacle at a detectable distance
	sensor_val=np.array([])    
	for x in range(1,16+1):
		errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_h[x-1],vrep.simx_opmode_buffer)                
		sensor_val=np.append(sensor_val,np.linalg.norm(detectedPoint)) #get list of values
		print(detectedPoint)
	
	#controller specific
	sensor_sq=sensor_val[0:8]*sensor_val[0:8] #square the values of front-facing sensors 1-8
	sensor_sq[np.isnan(sensor_sq)] = np.inf
	print('1 = ',sensor_val[0],' \n8 = ',sensor_val[7],'\n4=',sensor_val[3],'\n5 = ',sensor_val[4],'\n9 = ',sensor_val[8])

	min_ind=np.where(sensor_sq==np.min(sensor_sq))
	
	# time.sleep(0.1) use this only if you are getting list is empty error, or out of bounds 	 
	# print(sensor_sq)
	# print(np.argmin(sensor_sq))
	min_ind=min_ind[0][0]
	
	if sensor_sq[min_ind]<0.4:
		steer=-1/sensor_loc[min_ind]
	else:
		steer=0

	steer = 0

	v=1	#forward velocity
	kp=0.5	#steering gain
	vl=v+kp*steer
	vr=v-kp*steer
	# print ("V_l =",vl)
	# print ("V_r =",vr)

	errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
	errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)

	code,collision = vrep.simxReadCollision(clientID,robot, vrep.simx_opmode_buffer )
	if collision:
		break
		print('Coll',code, collision)

	time.sleep(0.2) #loop executes once every 0.2 seconds (= 5 Hz)

#Post ALlocation
errorCode=vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
errorCode=vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)


