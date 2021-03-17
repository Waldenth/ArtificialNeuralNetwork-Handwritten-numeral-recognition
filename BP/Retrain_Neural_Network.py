import numpy
import scipy.special
import scipy
import matplotlib.pyplot
import glob
import imageio
import pylab
import csv 
from PIL import Image
from tqdm import trange
class neuralNetwork:
    #神经网络的初始化
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #输入层的结点数，隐藏层结点数，输出层结点数，学习率
        #本神经网络为三层，输入层，隐藏层，输出层，对每层结点数进行初始化
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        #初始状态时先对输入&隐藏层每个结点对其下一层的每个结点的传递权重进行随机初始化
        self.wih=numpy.random.normal(0.0,pow(self.inodes,-0.5),(self.hnodes,self.inodes))
        #输入层结点到隐藏层的结点的每一个传递权重初始化，使用标准正态分布u=0.0，w_ij采用矩阵表示,hnodes行，inodes列

        self.who=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))
        #与上述同理

        #学习率
        self.lr=learningrate

        #激活函数 高斯S函数
        self.activation_function = lambda x : scipy.special.expit(x)

        pass
    
    #训练神经网络
    def train(self,inputs_list, targets_list):
        #将输入的各个量转换为二维矩阵并转置为n行1列型
        inputs=numpy.array(inputs_list,ndmin=2).T
        #结果同样处理
        targets=numpy.array(targets_list,ndmin=2).T

        #计算传入到隐藏层各个结点的hidden_inputs
        hidden_inputs=numpy.dot(self.wih,inputs)
        #激活隐藏层各个结点,得到隐藏层的输出,也就是输出层的输入
        hidden_outputs=self.activation_function(hidden_inputs)
        
        #计算输出层的输入矩阵
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #激活输出层,得到最终输出
        final_outputs=self.activation_function(final_inputs)

        #开始反馈更新权重
        #计算结果的误差  output_errors也是一个n行1列矩阵
        output_errors=targets-final_outputs
        #计算反馈给隐藏层的误差
        hidden_errors=numpy.dot(self.who.T,output_errors)
        #更新隐藏层到输出层的链接权重 *是正常乘法,不是点乘
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        #更新输入层到隐藏层的链接权重
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))

        pass

    #查询神经网络 接受神经网络的输入，返回网络的输出
    def query(self, inputs_list):
        #将输入各个值的inputs_list数组转换成一个2维数组(矩阵)，转置(n行1列)
        inputs = numpy.array(inputs_list, ndmin=2).T

        #构建隐藏层的输入hidden_inputs,采用输入层到隐藏层的权重链接矩阵 点乘 上述inputs矩阵,得到隐藏层每个输入结点的值构成的单列矩阵
        hidden_inputs = numpy.dot(self.wih, inputs)

        #使用激活函数，对隐藏层的每个结点输入得到相应输出
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #构建输出层的输入final_inputs,采用隐藏层到输出层的权重连接矩阵 点乘 隐藏层的输出矩阵,得到输出层每个输入结点的值构成的单列矩阵
        final_inputs = numpy.dot(self.who, hidden_outputs)
        
        #对输出层每个结点进行激活，得到最后的结果
        final_outputs = self.activation_function(final_inputs)
        
        #返回经过神经网络得到的结果
        return final_outputs
def main():
#训练集的数据为28x28的PNG格式图片,因此输入层结点784个
	input_nodes=784
	#隐藏层,设为22个
	hidden_nodes=90
	#输出层,由于只是识别0-9即可m设置为10个
	output_nodes=10

	#学习率,设置为0.1
	leraning_rate=0.1

	Network_data_file=open("Network_data_save.csv",'r')
	Network_data=Network_data_file.readlines()
	Network_data_file.close()

	n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,leraning_rate)

	for i in range(0,90):
		record=Network_data[i]
		all_values=record.split(',')
		for j in range(0,784):
			all_values[j]=float(all_values[j])
		n.wih[i]=all_values

	for j in range(90,100):
		record=Network_data[j]
		all_values=record.split(',')
		for i in range(0,90):
			all_values[i]=float(all_values[i])
		n.who[j-90]=all_values

	training_data_file=open("my_own_test_data.csv",'r')
	training_data_list=training_data_file.readlines()
	training_data_file.close()
		
	epochs=7
	for e in trange (epochs):
		for record in training_data_list:
			all_values=record.split(',')

			#颜色值RGB值的范围0-255,为了简化,将每个像素点映射到0-1的范围内
			inputs=(numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01

			#创建一个output_nodes个元素大小的数组,用零填充,随后+0.01,克服高斯S函数 1/(1+e^-x)不能到1和0的事实
			targets=numpy.zeros(output_nodes)+0.01
			#标记这个"图像"表示的实际数字的对应位置的数组元素
			targets[int(float(all_values[0]))]=0.99
			#用该项训练
			n.train(inputs,targets)
			'''
			inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
			n.train(inputs_plusx_img.reshape(784), targets)
			inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
			n.train(inputs_minusx_img.reshape(784), targets)
			'''
		

	with open("New_Network_data_save.csv",'w',newline='') as s_file:
		for item in range(0,n.hnodes):
			data=n.wih[item][0:]
			csv_writer=csv.writer(s_file)
			csv_writer.writerow(data)
		for item in range(0,n.onodes):
			data=n.who[item][0:]
			csv_writer=csv.writer(s_file)
			csv_writer.writerow(data)

if __name__ == '__main__':
    main()
