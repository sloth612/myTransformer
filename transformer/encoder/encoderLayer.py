# 编码器层EncoderLayer


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
from transformer.encoder.sublayerConnection import SublayerConnection as SublayerConnection
from transformer.encoder.multihead import clones as clones


# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: 词嵌入维度 - 将作为编码器层的大小
        :param self_attn: 多头自注意力子层实例化对象 - 自注意力机制
        :param feed_forward: 前馈全连接层实例化对象
        :param dropout: 置0比率
        """
        super(EncoderLayer, self).__init__()

        # 传入多头自注意力 和 前馈全连接
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 编码器层中有两个子层连接结构, 使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 传入词嵌入维度
        self.size = size

    def forward(self, x, mask):
        """
        :param x: 上一层的输出
        :param mask: 掩码张量mask
        :return:
        """
        # 1 子层连接结构 多头自注意力子层
        #   传入了函数类型的子层 sublayer = lambda x: self_attn(x, x, x, mask)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 2 子层连接结构 前馈全连接层
        return self.sublayer[1](x, self.feed_forward)


# 测试
# 实例化参数
# from transformer.input.positionalEncoding import pe_result as pe_result
# from transformer.encoder.multihead import MultiHeadedAttention as MultiHeadedAttention
# from transformer.encoder.positionwise import PositionwiseFeedForward as PositionwiseFeedForward
# from transformer.packages import device as device

# size = d_model = 512
# head = 8
# d_ff = 64
# x = pe_result
# dropout = 0.2
# self_attn = MultiHeadedAttention(head, d_model)  # 多头自注意力实例化对象 cpu
# self_attn.to(device)
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 前馈全连接实例化对象 cpu
# ff.to(device)
# mask = Variable(torch.zeros(8, 4, 4)).cuda()
#
# # 调用
# el = EncoderLayer(size, self_attn, ff, dropout)  # 编码器层实例化对象 cpu
# el.to(device)
# el_result = el(x, mask)  # [2 x 4 x 512]
# print(el_result)
# print(el_result.shape)
