import csv 
import numpy
import scipy.special
import scipy
import matplotlib.pyplot
import glob
import imageio
import pylab
from PIL import Image
from tqdm import trange
our_own_dataset=[]

'''
img = Image.open('./Error_image/identified_1.png')
out=img.resize((28,28))
out.save('./Error_image/identified_1.png')
'''


for image_file_name in glob.glob(r'./Error_image/identified_?.png'):
    #图片表示的数字是错的，因此要人工识别，这里定为-1
	label=-1
	print("loading data {} now......please wait a moment".format(image_file_name))
	img=Image.open(image_file_name)
	out=img.resize((28,28))
	out.save(image_file_name)
	image_array=imageio.imread(image_file_name,as_gray=True)

	image_data =255-image_array.reshape(784)
	#image_data =(image_data/255.0 * 0.99)+0.01

	record = numpy.append(label,image_data)
	our_own_dataset.append(record)

for item in trange(0,len(our_own_dataset)):
	with open("my_own_test_data.csv",'a',newline='') as t_file:
		data=our_own_dataset[item][0:]
		csv_writer=csv.writer(t_file)
		csv_writer.writerow(data)
         


