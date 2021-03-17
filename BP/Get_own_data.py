import csv 
import numpy
import scipy.special
import scipy
import matplotlib.pyplot
import glob
import imageio
import pylab

our_own_dataset=[]

for image_file_name in glob.glob(r'./test_data/????.png'):
    #得到图片的序号 取到1前一个字符
    label=int(image_file_name[12])

    print("loading data {} now......please wait a moment".format(image_file_name))

    image_array=imageio.imread(image_file_name,as_gray=True)

    image_data =255-image_array.reshape(784)
    #image_data =(image_data/255.0 * 0.99)+0.01

    record = numpy.append(label,image_data)
    our_own_dataset.append(record)

for item in range(0,len(our_own_dataset)):
	with open("my_own_test_data.csv",'a',newline='') as t_file:
		data=our_own_dataset[item][0:]
		csv_writer=csv.writer(t_file)
		csv_writer.writerow(data)
            

