import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tqdm import trange
import pickle
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    #忽略tensorflow的稀奇古怪的警告2333
#NOTICE 本文件约定,一层卷积和后续的一层池化 合并称为 一个卷积层

class Convolutional_Neural_Network:
    def __init__(self,conv1_filters,conv2_filters,fc1_units,learning_rate,training_steps,num_classes):
        self.conv1_filters=conv1_filters
        self.conv2_filters=conv2_filters
        self.fc1_units=fc1_units
        self.learning_rate=learning_rate
        self.training_steps=training_steps
        self.num_classes=num_classes
        random_normal=tf.initializers.RandomNormal()
        #卷积网络各层的权重连接
        self.weights={
            # 第一卷积层: 5x5卷积,1个输入,32个卷积核(MNIST只有一个颜色通道)
            'wc1':tf.Variable(random_normal([5,5,1,self.conv1_filters])),
            
            # 第二卷积层: 5x5卷积,32个输入,64个卷积核
            'wc2':tf.Variable(random_normal([5,5,self.conv1_filters,self.conv2_filters])),

            #全连接层:    7x7x64个输入,1024个神经元
            'wd1': tf.Variable(random_normal([7*7*64, self.fc1_units])),

            # 全连接层输出层: 1024个输入， 10个神经元（所有类别数目)
            'out': tf.Variable(random_normal([self.fc1_units, self.num_classes]))

        }

        #各层偏置量
        self.biases = {
            'bc1': tf.Variable(tf.zeros([self.conv1_filters])),
            'bc2': tf.Variable(tf.zeros([self.conv2_filters])),
            'bd1': tf.Variable(tf.zeros([self.fc1_units])),
            'out': tf.Variable(tf.zeros([self.num_classes]))
        }
    
    def conv2d(self,x,W,b,strides=1):   #conv2d包装器,带有偏置量和relu激活函数
        x=tf.cast(x,dtype=tf.float32)   #数据转化为tf库支持的格式
        x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')    #标准卷积会使图片变小,改变padding可以改变输出图片大小,这里让图片大小不变
        x=tf.nn.bias_add(x,b)   #添加偏置量
        return tf.nn.relu(x)

    def maxpool2d(self,x,k=2):#(最大)池化层包装器
        return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

    def conv_net(self,x):
        #输入数据形状:[-1,28,28,1]:一批28x28x1  (灰度)图像数据
        x=tf.reshape(x,[-1,28,28,1])

        #第一卷积层
        #卷积层,输出数据形状:[-1,28,28,32]
        conv1=self.conv2d(x,self.weights['wc1'],self.biases['bc1'])

        #最大池化层,输出数据形状:[-1,14,14,32]  (两两采样)
        conv1=self.maxpool2d(conv1,k=2)

        #第二卷积层
        #卷积层,输出数据形状:[-1,14,14,64]
        conv2=self.conv2d(conv1,self.weights['wc2'],self.biases['bc2'])

        #最大池化层,输出数据形状:[-1,7,7,64]
        conv2=self.maxpool2d(conv2,k=2)

        #对第二最大池化层输出数据进行调整(设置为标准向量),从而适应全连接层输入
        #调整形状:[-1,7*7*64]
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])

        # 全连接层， 输出形状： [-1, 1024]
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])

        #将ReLU应用于fc1输出以获得非线性
        fc1 = tf.nn.relu(fc1)

        #全连接层,输出数据形状[-1,10]
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

        #分类问题,卷积网络使用softmax函数激活
        return tf.nn.softmax(out)

    # 交叉熵损失函数
    def cross_entropy(self,y_pred, y_true):
        # 将标签编码为独热向量
        y_true = tf.one_hot(y_true, depth=self.num_classes)
        # 将预测值限制在一个范围之内以避免log（0）错误
        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
        # 计算交叉熵
        return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

    #  准确率评估
    def accuracy(self,y_pred, y_true):
        # 预测类是预测向量中最高分的索引（即argmax）
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

    def run_optimization(self,x, y):
        # ADAM 优化器
        optimizer = tf.optimizers.Adam(self.learning_rate)
        # 将计算封装在GradientTape中以实现自动微分
        with tf.GradientTape() as g:
            pred = self.conv_net(x)
            loss = self.cross_entropy(pred, y)
            
        # 要更新的变量，即可训练的变量
        p=list(self.weights.values())
        q=list(self.biases.values())
        trainable_variables = p+q

        # 计算梯度
        gradients = g.gradient(loss, trainable_variables)
        
        # 按gradients更新 W 和 b
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        

 
def main():
    
    #读取训练集数据
    training_data=open("mnist_train_15000.csv",'r')
    training_data_pri_list=training_data.readlines()
    training_data.close()

    #读取测试集数据
    test_data=open("test.csv",'r')
    test_data_pri_list=test_data.readlines()
    test_data.close()

    #数据矩阵化
    training_data_list=[]
    training_data_label_list=[]

    test_data_list=[]
    test_data_label_list=[]


    for record in training_data_pri_list:
        all_values=record.split(',')
        inputs=(np.asfarray(all_values[1:])/255.0*0.99)+0.01#将各个像素点的数据映射到0-1范围内
        inputs=np.asfarray(inputs).reshape((28,28))#将数据重构为28x28矩阵数组形式
        training_data_list.append(inputs)
        label=int(all_values[0])
        training_data_label_list.append(label)

    print('--------------训练集数据加载完毕!--------------')

    for record in test_data_pri_list:
        all_values=record.split(',')
        inputs=(np.asfarray(all_values[1:])/255.0*0.99)+0.01#将各个像素点的数据映射到0-1范围内
        inputs=np.asfarray(inputs).reshape((28,28))#将数据重构为28x28矩阵数组形式
        test_data_list.append(inputs)
        label=int(all_values[0])
        test_data_label_list.append(label)

    print('\n--------------测试集数据加载完毕!--------------')



    #卷积神经网络的相关参数
    conv1_filters=32    #第一卷积层 卷积核的数目
    conv2_filters=64    #第二卷积层 卷积核的数目
    fc1_units=1024      #第一全连接层 神经元的数目

    #MNIST数据集的相关参数
    num_classes=10      #所有的类别 (数字0-9)

    #卷积神经网络的训练参数
    learning_rate=0.001  #学习率
    training_steps=200  #总训练次数

    Network=Convolutional_Neural_Network(conv1_filters,conv2_filters,fc1_units,learning_rate,training_steps,num_classes)
    epochs=10
    for i in range(10):
        for j in trange(epochs):
            Network.run_optimization(training_data_list,training_data_label_list)
        pred=Network.conv_net(training_data_list)
        acc_train=Network.accuracy(pred,training_data_label_list)
        print("对训练集拟合准确率:{:.2f}%".format(acc_train*100))
        pred=Network.conv_net(test_data_list)
        acc_test=Network.accuracy(pred,test_data_label_list)
        print("在测试集上准确率:{:.2f}%".format(acc_test*100))
        acc_train=float(acc_train)
        acc_test=float(acc_test)
        with open("train_data_test.txt",'a')as file_object:
            data=str(i+1)+' '+str(acc_train)+'\n'
            file_object.write(data)
            file_object.close()
        with open("test_data_test.txt",'a')as file_object:
            data=str(i+1)+' '+str(acc_test)+'\n'
            file_object.write(data)
            file_object.close()
        '''
        pred=Network.conv_net(training_data_list)
        acc=Network.accuracy(pred,training_data_label_list)
        print("对训练集拟合准确率:{:.2f}%".format(acc*100))
        '''
    file=open("CNN_data.txt",'wb')
    pickle.dump(Network,file)

if __name__ == '__main__':
    main()