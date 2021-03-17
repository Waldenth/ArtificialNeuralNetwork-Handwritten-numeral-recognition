import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

with open('epochs.txt','r')as file_object:
	lines=file_object.readlines()
	
x_values=[]
y_values=[]
for line in lines:
	line=line.rstrip()
	line=line.split()
	#print(line)
	x_values.append(int(line[0]))
	y_values.append(float(line[1]))
'''	
#fig = plt.figure(figsize=(8, 20), dpi=80)
ax=plt.gca()	
y_major_locator=MultipleLocator(0.1)
ax.yaxis.set_major_locator(y_major_locator)
plt.ylim(0.1,0.5)
'''
#fig = plt.figure(figsize=(20,2), dpi=80)

x=range(1,51,1)
y=[0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,
0.44,0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,
0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.70]
plt.xticks(x)
plt.yticks(y)

plt.title("Epochs test",fontsize=20)
plt.xlabel("Epochs",fontsize=12)
plt.ylabel("Accuracy",fontsize=12)
print(x_values)
print(y_values)
plt.plot(x_values,y_values,linewidth=3)


#plt.scatter(x_values, y_values)
x1 = np.arange(0, 51, 1)
parameter = np.polyfit(x_values, y_values, 4)
y2 = parameter[0] * x1 ** 4 + parameter[1] * x1 ** 3 + parameter[2] * x1 ** 2 + parameter[3]* x1 +parameter[4]
plt.plot(x1, y2, color='r')

plt.show()
