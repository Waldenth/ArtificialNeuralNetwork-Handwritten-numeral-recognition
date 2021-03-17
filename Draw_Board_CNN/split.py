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


#首先行切
def cut_picture_row(image):
    image_0=[]#保存的含有数字的部分
    image_1=[]
    for i in range(image.shape[0]):
        if any(image[i,:]==0):#有黑色代表有数字
            image_1.append(image[i,:])
        else:
            #检查到数字的末尾
            if any(image[i-1,:]==0):
                image_0.append(image_1)
                image_1=[]
    #list转numpy类型
    for i,j in enumerate(image_0):
        image_0[i]=np.array(j)
    return image_0

#列切
def cut_picture_rank(image):
    image_0=[]
    image_1=[]
    for z in image:
        #按照图片列切
        for i in range(z.shape[1]):
            if any(z[:,i]==0):#这一列有黑色，代表有数字
                image_1.append(z[:,i])
            else:
                #检查数字末尾
                if any(z[:,i-1]==0) and len(image_1)>8:
                    image_0.append(image_1)
                    image_1=[]
    for i,j in enumerate(image_0):
        image_0[i]=np.array(j).T
    return image_0


def save_picture(image):
    j=0
    for i in image:
        plt.imsave("./image/%s.png" % (str(j)), i,cmap ='gray')
        j += 1      


def split_word(string):
	img=Image.open(str(string))
	image=img.convert("L")
	image_np=picture_tow_valued(image)#返回一个二维数组
	image_0=[]
	image_0=cut_picture_row(image_np)#返回多个二维数组
	image_0=cut_picture_rank(image_0)
	#print(len(image_0))
	return_valve=len(image_0)
	for i in range(0,len(image_0)):
		maxlen=0
		image_0[i]=np.array(image_0[i])
		j=image_0[i].shape[0]
		k=image_0[i].shape[1]
		if(j<k):
			maxlen=k
			addlen=60
			sub = k - j
			new_img = np.full((maxlen+addlen, maxlen+addlen),1,dtype=float)
			for w in range(0,j):
				for m in range(0,k):
					new_img[int(sub/2 + w+30)][m+30] = image_0[i][w][m]
			image_0[i]=new_img
		else:
			maxlen=j
			sub=j-k
			addlen=60
			new_img=np.full((maxlen+addlen, maxlen+addlen),1,dtype=float)
			for w in range(0,j):
				for m in range(0,k):
					new_img[w+30][int(m+30+sub/2)] = image_0[i][w][m]
			image_0[i]=new_img
	save_picture(image_0)
	return return_valve
def main():
	pass	
if __name__ == '__main__':
    main()
	
	
