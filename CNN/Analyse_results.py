import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

with open('train_data_test.txt','r')as file_object:
	lines=file_object.readlines()
	file_object.close()
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

x=range(2,20,1)
y=[0.6,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98,1.00]
plt.xticks(x)
plt.yticks(y)

plt.title("CNN Test Result",fontsize=20)
plt.xlabel("Epochs",fontsize=12)
plt.ylabel("Accuracy",fontsize=12)
print(x_values)
print(y_values)
plt.plot(x_values,y_values,linewidth=2,color='red')


with open('test_data_test.txt','r')as file_object:
	lines=file_object.readlines()
	file_object.close()
x_values=[]
y_values=[]
for line in lines:
	line=line.rstrip()
	line=line.split()
	#print(line)
	x_values.append(int(line[0]))
	y_values.append(float(line[1]))
print(x_values)
print(y_values)
plt.plot(x_values,y_values,linewidth=2)

plt.show()
