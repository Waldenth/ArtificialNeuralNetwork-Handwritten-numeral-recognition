import numpy as np
from tqdm import tqdm
from tqdm import trange
import pickle
import numpy
import glob
from CNN_train import Convolutional_Neural_Network
import tensorflow as tf
import matplotlib.pyplot
import pylab
import csv 
from PIL import Image
import imageio
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    #忽略tensorflow的稀奇古怪的警告2333

file=open('New_CNN_data.txt','rb')
Neural_Network=pickle.load(file)

def Identify_Draw_Board(paper):
    '''
    our_own_dataset=[]
    for image_file_name in glob.glob(r'temp.png'):
        
        image_array=imageio.imread(image_file_name,as_gray=True)

        image_data =255.0-image_array.reshape(784)

        image_data =(image_data/255.0 * 0.99)+0.01
        
        record = numpy.append(-1,image_data)

        our_own_dataset.append(record)
    '''
    our_own_dataset=[]
    string_data=[]
    for i in range(0,paper):
        image_file_name='./image/'+str(i)+'.png'
        
        image_array=imageio.imread(image_file_name,as_gray=True)

        image_data =255.0-image_array.reshape(784)

        image_data =(image_data/255.0 * 0.99)+0.01
        
        record = numpy.append(-1,image_data)

        our_own_dataset.append(record)

    for item in range(0,len(our_own_dataset)):
        inputs=our_own_dataset[item][1:]
        pred=Neural_Network.conv_net(inputs)
        outputs=tf.argmax(pred, 1)
        #print(outputs)  #only be used for debug
        label=outputs
        label=int(label)
        img = Image.open('temp2.png')
        string='./classification/'+str(label)+'/identified_'+str(paper)+'.png'
        img.save(string)
        string_data.append(label)
    return string_data

def main():
    pass

if __name__ == '__main__':
    main()
