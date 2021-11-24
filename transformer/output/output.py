# 输出部分实现


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
# nn.functional工具包装载了网络层中那些只进行计算, 而没有参数的层
import torch.nn.functional as F


# 生成器类 实现了线性层和softmax
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        :param d_model: 词嵌入维度
        :param vocab_size: 词表大小
        """
        super(Generator, self).__init__()
        # 预定义线性层实例化 [d_model x vocab_size]
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        :param x: 上一层的输出张量x
        :return:
        """
        # 1 self.project 线性变化
        # 2 softmax 处理
        #   此处使用log_softmax是因为pytorch版本损失函数实现问题 其他版本中使用softmax
        #   log_softmax对softmax的结果取了对数,因为对数函数是单调递增函数,因此无影响
        return F.log_softmax(self.project(x), dim=-1)


# 测试

# 实例化参数
d_model = 512
vocab_size = 1000  # ci'biao

# 输入参数
from transformer.decoder.decoder import de_result as de_result
from transformer.packages import device as device

x = de_result   # [2 x 4 x 512]
# 调用
gen = Generator(d_model, vocab_size)  # 生成器类实例化对象 cpu
gen.to(device)
# gen_result = gen(x)  # [2 x 4 x 512] x [512 x 1000] = [2 x 4 x 10000]
# print(gen_result)
# print(gen_result.shape)

