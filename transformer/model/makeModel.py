# 整体的编码器解码器结构


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
import copy
from transformer.input.positionalEncoding import PositionalEncoding as PositionalEncoding
from transformer.input.embedding import Embeddings as Embeddings
from transformer.encoder.multihead import MultiHeadedAttention as MultiHeadedAttention
from transformer.encoder.positionwise import PositionwiseFeedForward as PositionwiseFeedForward
from transformer.encoder.encoderLayer import EncoderLayer as EncoderLayer
from transformer.encoder.encoder import Encoder as Encoder
from transformer.decoder.decoderLayer import DecoderLayer as DecoderLayer
from transformer.decoder.decoder import Decoder as Decoder
from transformer.output.output import Generator
from transformer.model.encoderDecoder import EncoderDecoder

from transformer.packages import device as device


def make_model(source_vocab, target_vocab, N=6,
               d_model=512, d_ff=2048, head=8, dropout=0.1):
    """
    构建transformer模型
    :param source_vocab: 源数据词表总数
    :param target_vocab: 目标数据词表总数
    :param N: 编码器层和解码器层的层数
    :param d_model: 词嵌入维度
    :param d_ff: 前馈全连接层中变换矩阵的维度
    :param head: 多头注意力机制中的头数
    :param dropout: 置零比率
    :return:
    """
    # 深度拷贝函数
    c = copy.deepcopy

    # 1多头注意力实例化对象
    attn = MultiHeadedAttention(head, d_model)

    # 2前馈全连接实例化对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 3位置编码类实例化对象
    position = PositionalEncoding(d_model, dropout)

    # 放到gpu上
    attn.to(device)
    ff.to(device)
    position.to(device)

    # 编码器-解码器 5个参数: 编码器,解码器,源数据词嵌入函数,目标数据词嵌入函数,生成器
    # 编码器: 多头自注意力层 前馈全连接层
    # 解码器: 多头自注意力层 多头注意力层 前馈全连接层
    # 词嵌入函数 用nn.Sequential装 词嵌入+位置编码
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab))
    # print(next(model.parameters()).is_cuda)
    # 放到gpu上
    model.to(device)

    # 模型结构完成, 接下来初始化模型中的参数,比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# 输入参数
source_vocab = 11
target_vocab = 11
N = 6
# 其他参数默认值
# print(__name__)
# if __name__ == '__main__':
#     res = make_model(source_vocab, target_vocab, N)
#     print(res)

