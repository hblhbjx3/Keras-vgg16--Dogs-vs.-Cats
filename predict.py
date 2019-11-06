import time
import csv
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import os
import matplotlib.pyplot as plt
# 加载模型
start = time.clock()
model = load_model('dogcatmodel1.0.h5')
print('Warming up took {}s'.format(time.clock() - start))
# 图片预处理
#path='E:/DCIM/P51024-101922.gif'
dir ='test500/'
file_name = os.listdir(dir)
file_name.sort(key=lambda x:int(x[:-4]))
#将照片按文件名进行排序 -4 代表跳过后四位对前面的文字进行排序
n = len(file_name)
y = n * ['']
#print(file_name)
for i in range(n):
    path = dir+file_name[i]
    img_height, img_width = 256, 256
    x = image.load_img(path=path, target_size=(img_height, img_width))
    x = image.img_to_array(x)
    x = x[None]
    #start = time.clock()
    y[i] = int(model.predict(x))
#print(y)
with open('500.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column = [row[1]for row in reader]
    column1 = column[1:]
    #print(column1)
sum = 0
for i in range (n):
    if y[i] == int(column1[i]) :
        sum += 1
    else :
        pass
right = sum/n*100
print('预测为  (0为猫，1为狗)')
print("最后的准确率为{}%".format(right))
#print('预测花费时间： {}s'.format(time.clock() - start))

"""
img = Image.open(path)
plt.imshow(img)   #根据数组绘制图像
plt.show()        #显示图像
# 预测
"""