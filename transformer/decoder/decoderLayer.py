# 解码器层DecoderLayer


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
from transformer.encoder.sublayerConnection import SublayerConnection as SublayerConnection


# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        :param size: 词嵌入维度 - 解码器层大小
        :param self_attn: 多头自注意力实例化对象 Q=K=V
        :param src_attn: 多头注意力实例化对象 Q!=K=V
        :param feed_forward: 前馈全连接层实例化对象
        :param dropout: 置零比率
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # clones函数克隆三个子层连接对象 - 解码器层有3个残差连接
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 上一层的输入x
        :param memory: 编码器层的语义存储变量memory 编码器的输出
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        :return:
        """
        # 将memory表示成m方便之后使用
        m = memory

        # 第1个子层连接结构: 带mask的多头自注意力self-attn sublayer中自带规范化层 残差连接
        #   mask是对目标数据的遮掩 因为此时模型可能还没有生成目标数据
        #   比如解码器在生成第一个字符或者词汇的时候 不希望模型使用第二个及以后的信息
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # 第2个子层连接结构: 多头注意力子层 sublayer中自带规范化层 残差连接
        #   Q是输入x KV是编码器层输出memory
        #   传入source_mask 对源数据进行遮掩 不是为了抑制信息泄露,而是这笔无意义字符产生的注意力值
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        # 第3个子层连接结构: 前馈全连接子层 sublayer中自带规范化层 残差连接
        return self.sublayer[2](x, self.feed_forward)


# 测试

# 实例化参数
from transformer.encoder.multihead import MultiHeadedAttention as MultiHeadedAttention
from transformer.encoder.positionwise import PositionwiseFeedForward as PositionwiseFeedForward
from transformer.packages import device as device

head = 8
size = d_model = 512
d_ff = 64
dropout = 0.2
# 多头自注意力实例化对象 常规多头注意力 cpu
self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)
self_attn.to(device)
src_attn.to(device)
# 前馈全连接层实例化对象 cpu
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff.to(device)

# 输入参数
# from transformer.input.positionalEncoding import pe_result as pe_result
# from transformer.encoder.encoder import en_result as en_result
# # x是目标数据的词嵌入表示 形式和源数据的词嵌入表示相同 此处用源数据的词嵌入pe_result 充当
# x = pe_result
# # memory是编码器的输出
# memory = en_result
# # 实际中source_mask 和 target_mask并不相同 此处为了方便令两者相同
# mask = Variable(torch.zeros(8, 4, 4)).cuda()
# source_mask = target_mask = mask
#
# # 调用
# dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)  # 解码器层对象 cpug
# dl.to(device)
# dl_result = dl(x, memory, source_mask, target_mask)  # [2 x 4 x 512]
# print(dl_result)
# print(dl_result.shape)


