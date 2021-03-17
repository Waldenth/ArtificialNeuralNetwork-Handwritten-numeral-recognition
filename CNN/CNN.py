import numpy as np
from tqdm import tqdm
from tqdm import trange
import pickle
from CNN_train import Convolutional_Neural_Network
import tensorflow as tf
import matplotlib.pyplot
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    #忽略tensorflow的稀奇古怪的警告2333
#读取测试集数据
test_data=open("test.csv",'r')
test_data_pri_list=test_data.readlines()
test_data.close()

test_data_list=[]
test_data_label_list=[]

for record in test_data_pri_list:
    all_values=record.split(',')
    inputs=(np.asfarray(all_values[1:])/255.0*0.99)+0.01#将各个像素点的数据映射到0-1范围内
    inputs=np.asfarray(inputs).reshape((28,28))#将数据重构为28x28矩阵数组形式
    test_data_list.append(inputs)
    label=int(float(all_values[0]))
    test_data_label_list.append(label)

print('--------------测试集数据加载完毕!--------------')
    


file=open('CNN_data.txt','rb')
Neural_Network=pickle.load(file)

print('\n--------------卷积神经网络导入完毕!--------------')

def display_test():
    pred=Neural_Network.conv_net(test_data_list)
    print("\n在测试集上准确率:{:.2f}%".format(Neural_Network.accuracy(pred,test_data_label_list)*100))

def display_image():
    for item in trange(50):
        all_values=test_data_list[item]
        label=test_data_label_list[item]
        pred=Neural_Network.conv_net(all_values)
        outputs=tf.argmax(pred, 1)
        image_array=np.asfarray(all_values[:]).reshape((28,28))
        matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
        pre=outputs
        pre=int(pre)
        a=item+1
        matplotlib.pyplot.title(str(pre), fontsize=24)
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.savefig('./MNIST_test_image_classification/image'+str(a)+'.png')
    
    print("\n验证集样本导出完毕!\n")

def main():
    display_test()
    display_image()

if __name__ == '__main__':
    main()
