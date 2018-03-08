import matplotlib.pyplot as plt
import pickle 
import numpy
stepList = pickle.load(open('Files/steps.pickle', 'rb')) 


def movingaverage(interval, window_size):
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')



sum  = 0
print("\nStep Average over 50 = ")
fail = False
for i in range(len(stepList)):
	if stepList[i] > 195:
		fail = True
	sum += stepList[i]
	if (i+1) % 50 == 0: 
		print(sum/50.0, end=" ")
		sum = 0
print("####",len(stepList))
step_mav = movingaverage(stepList,window_size=10)
print(step_mav)
print("\nFail = ",fail)
range = 3
offset = 1
fig = plt.figure(1)
plt.subplot(211) # 211 = numrows,numcolumns, curront plot, 2 rows , 1 columns - first row first plot ; 212 = second row first plot
# https://matplotlib.org/users/pyplot_tutorial.html
plt.axis([1, len(stepList)+10, 0, 500])
plt.plot(stepList)
plt.title('Steps VS Episode')
plt.ylabel('steps per episode')
plt.xlabel('episode')

plt.subplot(212)
plt.axis([1, len(stepList)+10, -1*range, 500])
plt.plot(step_mav)
plt.title('Mav_Steps VS Episode')
plt.ylabel('Moving Average')
plt.xlabel('episode')

plt.show()
#fig.savefig(self.graphPath+str(episode + 1)+'.png')
