import numpy
import scipy.special
import scipy.ndimage
import scipy
import matplotlib.pyplot
import glob
import imageio
import pylab
import csv 
from tqdm import trange

training_data_file=open("mnist_train.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()


#训练集csv是一个文本表格文件,每一行为一张手写的24x24的PNG格式的数字，第一列是这个数字的实际值,
#后面784列是各个像素点的值
#以下是数据转图像代码
for i in trange(70):
    all_values=training_data_list[i].split(',')
    image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
    pre=all_values[0]
    a=i+1
    matplotlib.pyplot.title(str(pre), fontsize=24)
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.savefig('./MNIST_train_data_image_display/image'+str(a)+'.png')
    #pylab.show()
