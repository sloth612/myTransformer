# 绘制词汇向量的特征分布曲线

# 必备工具包
import torch

# 预定义的网络层torch.nn 如卷积层，lstm层，embedding层

# 数学计算工具包

# torch中变量封装函数Variable
'''
Variable可以把输出的Tensor变成一个输入变量，这样梯度就不会回传了。
detach()也是可以的如果都不加那么就得retain_graph=True了，否则报错
'''
from torch.autograd import Variable

# -----------------------------------------------------------------
from transformer.input.positionalEncoding import PositionalEncoding as PositionalEncoding
from transformer.packages import device as device

import matplotlib.pyplot as plt
import numpy as np

# 创建一张15 x 5大小的画布
plt.figure(figsize=(15, 5))

# 实例化PositionalEncoding类得到pe对象, 输入参数是20和0  - d_model, dropout
pe = PositionalEncoding(20, 0)  # d_model = 20, 在cpu上

# 向pe传入Variable封装的tensor, pe会直接执行forward函数,
# tensor张量为 1 x 100 x 20 一共100个词 每个词的词嵌入维度为 20, 假设给tensor以及经过embedding
# 且这个tensor里的数值都是0, 处理后相当于位置编码张量
y = pe(Variable(torch.zeros(1, 100, 20)))

# 定义画布的横纵坐标, 横坐标到100的长度, 纵坐标是某一个词汇中的某维特征在不同长度下对应的值
# 总共有20维(d_model = 20), 这里只查看4, 5, 6, 7维的值.
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())

# 在画布上填写维度提示信息
plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
plt.savefig('./test2.png')
plt.show()

