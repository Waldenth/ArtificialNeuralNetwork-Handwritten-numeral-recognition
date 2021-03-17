import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt

#0黑色,1白色
def picture_tow_valued(image):
    # 把图片转为numpy类型
    image_np=np.array(image)
    # 计算图片离散度
    discretization=image_np.mean()-image_np.std()
    discretization=int(discretization)
    # 循环图片上每一个点
    for i in range(image_np.shape[0]):
        for j in range(image_np.shape[1]):
            # 把大于离散度的数值变为1(白色),小于的变为0(黑色),图片就变成了只有两个数值的黑白图片
            if image_np[i,j]>discretization:
                image_np[i,j]=1
            else:
                image_np[i,j]=0
    '''
    有些图片有瑕疵,边缘没有处理会影响切图,吧边缘数字全部变为0(黑色)
    '''
    for i in range(image_np.shape[0]):
        for j in range(image_np.shape[1]):
            if i == 0 or i == (image_np.shape[0]-1):
                image_np[i,j]=1
                continue
            if j == 0 or j == (image_np.shape[1]-1):
                image_np[i,j]=1
                continue
    #print(image_np)
    return image_np


for i in range(10,21,10):
	image_name=str(i)+'%.jpg'
	img=Image.open(image_name)
	image=img.convert("L")
	image=picture_tow_valued(image)
	plt.imsave("./%s.png" % (str(i)+'%_Processed'), image,cmap ='gray')
