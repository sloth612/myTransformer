# 掩码张量

# 必备工具包
import matplotlib.pyplot as plt
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

# -----------------------------------------------------------------
import numpy as np
from transformer.packages import device as device


def subsequent_mask(size):
    """
    生成向后遮掩的掩码张量
    :param size:掩码张量最后两个维度的大小, 它的最后两维形成一个方阵
    :return:
    """
    # 在函数中, 首先定义掩码张量的形状
    attn_shape = (1, size, size)  # 1是为了扩充维度 成为三维

    # 使用np.ones和np.triu形成元素为0和1的上三角阵 - 主对角线及以下为0(主对角线为0)
    # 为了节约空间 再使其中的数据类型变为无符号8位整形unit8
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # numpy转化为tensor, 内部做1-操作. 主对角线以上为0(主对角线不为0)
    return torch.from_numpy(1 - subsequent_mask)


# 测试
# size代表生成的掩码张量的最后两维的大小
size = 5
sm = subsequent_mask(size)
print(sm)

plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])  # 三维转化为二维
plt.savefig('./sm.png')
plt.show()
