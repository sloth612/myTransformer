# 解码器Decoder


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
import copy

# 使用类Decoder来实现解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer: 解码器层layer
        :param N: 解码器层的个数N
        """
        super(Decoder, self).__init__()
        # 克隆N个解码器层layer 实例化一个规范化层 - 因为数据经过所有解码器层后需要规范化
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 目标数据的嵌入表示
        :param memory: 编码器层的输出
        :param source_mask: 源数据的掩码张量
        :param target_mask: 目标数据的掩码张量
        :return:
        """
        # 循环让x经过每一个解码器层的处理 然后规范化
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


# 测试
from transformer.encoder.multihead import MultiHeadedAttention as MultiHeadedAttention
from transformer.encoder.positionwise import PositionwiseFeedForward as PositionwiseFeedForward
from transformer.decoder.decoderLayer import DecoderLayer as DecoderLayer
from transformer.packages import device as device

size = d_model = 512
head = 8
d_ff = 64
dropout = 0.2
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
attn.to(device)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff.to(device)
layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
layer.to(device)
N = 8

# 调用
from transformer.input.positionalEncoding import pe_result as pe_result
from transformer.encoder.encoder import en_result as en_result
x = pe_result   # 目标数据的嵌入表示 这里用源数据的嵌入表示代替
memory = en_result
mask = Variable(torch.zeros(8, 4, 4))
if device == torch.device('cuda:0'):
    mask = mask.cuda()
source_mask = target_mask = mask

# 调用
de = Decoder(layer, N)  # 解码器 cpu
de.to(device)
de_result = de(x, memory, source_mask, target_mask)  # [2, 4, 512]
# print(de_result)
# print(de_result.shape)
