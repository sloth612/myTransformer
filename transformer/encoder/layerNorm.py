# 规范化层layerNorm


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


# 规范化层
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        :param features: 词嵌入维度
        :param eps: 一个足够小的数 出现在规范化公式的分母中防止除0
        """
        super(LayerNorm, self).__init__()

        # 两个参数张量 a2全1 b2全0 作为调节因子 既能满足规范化要求又能不改变争对目标的表征
        # 使用nn.parameter封装 表示其为模型的参数 会跟着迭代而训练
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        # 把eps传到类中
        self.eps = eps

    def forward(self, x):
        """
        :param x:
        :return:
        """
        """输入参数x代表来自上一层的输出"""
        # 1 求输入变量x的最后一个维度的均值 保持输入输出维度一致
        # 2 求最后一个维度的标准差 根据规范化公式 用x减去均值除以标准差得到规范化结果
        # 3 结果乘以缩放系数a2 * 代表同型点乘 然后加上位移参数b2
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)  # 求标准差
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


# 测试
# 实例化参数
feature = d_model = 512
eps = 1e-6

# 输入参数
from transformer.encoder.positionwise import ff_result as ff_result
from transformer.packages import device as device

# x = ff_result  # 经过前馈全连接层的向量 [2 x 4 x 512]
# ln = LayerNorm(feature, eps)  # 规范化层 cpu上
# ln.to(device)
# ln_result = ln(x)
# print(ln_result)


