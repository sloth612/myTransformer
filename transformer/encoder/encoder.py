# 编码器Encoder


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
from transformer.encoder.multihead import clones as clones
from transformer.encoder.layerNorm import LayerNorm as LayerNorm
# 用于深度拷贝的copy工具包
import copy

# 编码器类
class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer: 编码器层
        :param N: 编码器层的个数
        """
        super(Encoder, self).__init__()
        # clones函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 初始化一个规范化层 将用在编码器的最后面
        self.norm = LayerNorm(layer.size)  # layer.size 词嵌入维度

    def forward(self, x, mask):
        """
        :param x: 上一层的输出
        :param mask: 掩码张量
        :return:
        """
        # 对克隆的N个编码器层进行循环 每次迭代得到一个新的x - 输出的x经过了N个编码器层的处理
        # 最后通过规范化层的对象self.norm进行处理
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 测试
# 实例化参数
from transformer.encoder.multihead import MultiHeadedAttention as MultiHeadedAttention
from transformer.encoder.positionwise import PositionwiseFeedForward as PositionwiseFeedForward
from transformer.encoder.encoderLayer import EncoderLayer as EncoderLayer
from transformer.input.positionalEncoding import pe_result as pe_result
from transformer.packages import device as device

size = d_model = 512
head = 8
d_ff = 64
x = pe_result
dropout = 0.2
c = copy.deepcopy

attn = MultiHeadedAttention(head, d_model)  # 多头自注意力实例化对象cpu
attn.to(device)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 前馈全连接层实例化对象cpu
ff.to(device)
mask = Variable(torch.zeros(8, 4, 4))
if device == torch.device('cuda:0'):
    mask = mask.cuda()

layer = EncoderLayer(size, c(attn), c(ff), dropout)  # 编码器层实例化对象cpu
layer.to(device)
# el_result = layer(x, mask)  # [2 x 4 x 512]
# print(el_result)
N = 8  # 编码器中编码器层的数量


# 调用
en = Encoder(layer, N)  # 编码器 cpu
en.to(device)
en_result = en(x, mask)  # [2 x 4 x 512]
# print(en_result)
# print(en_result.shape)
