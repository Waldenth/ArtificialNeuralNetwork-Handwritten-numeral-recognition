BP反馈神经网络 : numpy搭建
CNN卷积神经网络: tensorflow2.x
-------------------------------------------------
更新：
将模块进行了分类，BP,CNN神经网络分别设立文件夹，调用画板与网络分离，更有条理。
不要随意更改文件名
使用tqdm库完善程序，增加进度条，增强可视效果
导出可视化并标记部分测试数据
文件夹
CNN：	
./CNN_data:	存放卷积神经网络训练后的数据，仅Python程序可以识别
./Retrain_CNN:	对CNN进行数据在训练并得到更新的数据文件
MNIST_test_image_classification:	将测试集的前50份数据转换为图像并标记网络的测试结果	

Draw_Board说明：
绘图后，s键保存图片，i键使用网络进行识别，识别图像复制到./classification/相应文件夹中
必须先保存，再识别
绘图失误,esc键可以清屏
Draw_Board_BP/CNN:使用不同的神经网络进行识别

Analyse_xxx:
对训练和测试数据统计可视化

算力有限，CNN只对1w5张图片进行了训练，可以扩大训练集，CNN池化层可以防止过拟合，训练次数可有效提升精度
-------------------------------------------------
文件夹
Analyse_data:	在确定神经网络的各个主要参数(隐藏层结点数，学习率，训练次数等)时进行分析产生的各种数据
back_roll:		回滚文件夹，当出现意外时回滚上一版本
Identify_handwriting_numbers:	最早的版本和若干文件
my_own_data:	自己画板手写的数字
test_data:		所需要用于测试的图片放入此文件夹，命名x_xx.png x:这个图片代表的数字,xx:为表示这个数字的图片的序号

文件
Analyse_xxxx.py:	可视化各个参数的脚本
get_own_data.py:	若要强化学习,可使用该脚本获得test_data文件夹内各个图片的数据，并转换为标准训练集模式的数据
Identify_handwriting_number.py:	主脚本，包含神经网络和训练
To_Find_xxxx.py:	寻找各个参数不同值效果并保存相应文件的脚本
Neural_Network.py:	调用其来识别图像的神经网络脚本
mnist_train_xxx.csv:	训练集，后缀5000为较小的一个
my_own_test_data.csv:	测试图片文件夹各个图片的数据化文件(可直接加入训练集来强化学习)
Network_data_save.csv:	本次训练后得到的神经网络数据，新一轮训练前应备份
