# 用来测试函数

# 必备工具包
import torch

# 预定义的网络层torch.nn 如卷积层，lstm层，embedding层
import torch.nn as nn

# 数学计算工具包
import math

# torch中变量封装函数Variable
'''
Variable可以把输出的Tensor变成一个输入变量，这样梯度就不会回传了。
detach()也是可以的如果都不加那么就得retain_graph=True了，否则报错
'''
from torch.autograd import Variable

######################################torch.arange(start, end)
# y = torch.arange(1, 6, 2)
# print(y)
# print(y.dtype)

######################################unsqueeze() 升维
# a = torch.arange(0, 6)
# a = a.view(2, 3)
# print(a.shape)  # 2 x 3
# print(a)
# b = a.unsqueeze(1)  # 2 x 1 x 3 因为在第一维上增加了一维（从0开始）
# print(b)
# b = a.unsqueeze(0)  # 1 x 2 x 3 因为在第零维上增加了一维（从0开始）
# print(b)
# b = a.unsqueeze(-1)
# print(b.shape)
# b = a.unsqueeze(-2)
# print(b.shape)
# print(b)

######################################squeeze() 降维
# a = torch.arange(0, 6)
# a = a.view(2, 3)
# print(a.shape)  # 2 x 3
# b = a.unsqueeze(0)  # 1 x 2 x 3
# print(b.shape)
# c = b.squeeze(0)  # 2 x 3
# print(c.shape)
# c = b.squeeze(-3)  # 2 x 3
# print(c.shape)

######################################nn.Dropout
# m = nn.Dropout(p=0.2)
# input1 = torch.randn(4, 5)
# print(input1)
# output = m(input1)
# print(output)

######################################torch.exp
# ans = torch.exp(torch.tensor([0, math.log(2.)]))
# print(ans)

######################################numpy.triu
# import numpy as np
#
# t = np.triu(np.ones(3), k=1)
# print(t)
# print(torch.from_numpy(1-t))

######################################numpy.transpose
# import numpy as np
# arr = np.arange(24).reshape((2, 3, 4))
# print('init arr:------------------')
# print(arr)
# print(arr.shape)
# arr = arr.transpose(1, 0, 2)
# print('change arr:----------------')
# print(arr)
# print(arr.shape)

######################################torch.nn.functional.softmax
# import torch.nn.functional as F
# t = torch.arange(0, 24)
# t = t.view(2, 3, 4).float()
# print(t)
# t = F.softmax(t, 1)
# print(t)

# ######################################torch.view和torch.transpose
# t = torch.arange(0, 24)  # 假设样本数量为1
# a = t.view(2, 12)  # 2个字词:[0-11],[12-23] 词嵌入维度:12,头数:3
# a = a.unsqueeze(0)
# print(a.shape)  # torch.Size([1, 2, 12])
# print(a)  # 输出原数据
# '''
# tensor([[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
#          [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]])
# '''
# # 转置版-------------------------------------
# b = t.view(1, -1, 3, 4)  # 自适应
# c = b.transpose(1, 2)  # 转置 得到3个头
# print(c)
# '''
# tensor([[[[ 0,  1,  2,  3],  # 头1
#           [12, 13, 14, 15]],
#
#          [[ 4,  5,  6,  7],  # 头2
#           [16, 17, 18, 19]],
#
#          [[ 8,  9, 10, 11],  # 头3
#           [20, 21, 22, 23]]]])
# '''
#
# # 非转置版-------------------------------------
# b = t.view(1, -1, 2, 4)  # 自适应 3个头
# print(b)
# '''
# tensor([[[[ 0,  1,  2,  3],  # 头1
#           [ 4,  5,  6,  7]],
#
#          [[ 8,  9, 10, 11],  # 头2
#           [12, 13, 14, 15]],
#
#          [[16, 17, 18, 19],  # 头3
#           [20, 21, 22, 23]]]])
# '''

######################################torch.nn.Linear
# m = nn.Linear(20, 30)
# input = torch.randn(128, 20)
# output = m(input)
# print(output.shape)  # [128 x 30]

######################################smoothing
# from pyitcast.transformer_utils import LabelSmoothing
# import matplotlib.pyplot as plt
#
# # 使用LabelSmoothing实例化一个crit对象.
# # 参数1:size代表目标数据的词汇总数,也是模型最后一层得到张量的最后一维大小 5说明目标词汇总数是5个
# # 参数2:padding_idx表示要将那些tensor中的数字替换成0, 一般padding_idx=0表示不进行替换
# # 参数3:第三个参数smoothing表示标签的平滑程度 如原标签表示值为1, 则平滑后它的值域变为[1-smoothing, 1+smoothing].
# crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)
#
# # 假定一个任意的模型最后输出预测结果和真实结果
# predict = Variable(torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0]]))
#
# # 标签的表示值是0，1，2
# target = Variable(torch.LongTensor([2, 1, 0]))
#
# # 将predict, target传入到对象中
# crit(predict, target)
#
# # 绘制标签平滑图像
# plt.imshow(crit.true_dist)
# plt.savefig('./smoothing.png')
# plt.show()

######################################test
# arr = input("")
# print('arr', arr)
t = [int(n) for n in input().split(' ')]
for i in range(len(t)):
    print(i, t[i])