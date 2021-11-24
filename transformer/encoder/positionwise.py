# 前馈全连接层PositionwiseFeedForward


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
# -----------------------------------------------------------------
import torch.nn.functional as F


# 前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: 词嵌入维度 第一个线性层的输入维度==第二个线性层的输出维度
        :param d_ff: 第一个线性层的输出维度==第二个线性层的输入维度
        :param dropout: 置零比率
        """
        super(PositionwiseFeedForward, self).__init__()

        # 使用nn实例化2个线性层对象:self.w1和self.w2
        self.w1 = nn.Linear(d_model, d_ff)  # [d_model x d_ff]
        self.w2 = nn.Linear(d_ff, d_model)  # [d_ff x d_model]
        # nn.Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: 来自上一层的输出
        :return:
        """
        # 1. 经过第一个线性层 relu激活
        # 2. dropout置零 经过第二个线性层
        return self.w2(self.dropout(F.relu(self.w1(x))))


# 测试
from transformer.encoder.multihead import mha_result as mha_result
# 实例化参数
d_model = 512
# 线性变化的维度
d_ff = 64
dropout = 0.2

# 输入参数
from transformer.packages import device as device

x = mha_result  # 经过多头注意力机制的张量 [2 x 4 x 512]
ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 模型 cpu
ff.to(device)
ff_result = ff(x)
# print(ff_result)  # [2 x 4 x 512]