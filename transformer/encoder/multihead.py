# 多头注意力MultiHeadedAttention


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
from transformer.encoder.attention import attention as attention
# 用于深度拷贝的copy工具包
import copy


# 克隆函数,在多头注意力机制的实现中需要多个结构相同的线性层
# 使用clone函数将它们初始化在一个网络层列表对象中,之后的结构中也会用到该函数.
def clones(module, N):
    """
    用于生成相同网络层的克隆函数
    :param module: 示要克隆的目标网络层
    :param N: 克隆的数量
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 多头注意力
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """
        :param head: 头数
        :param embedding_dim: 词嵌入的维度 d_model
        :param dropout: 置零比率
        """
        super(MultiHeadedAttention, self).__init__()

        # 使用了一个测试中常用的assert语句,判断h是否能被d_model整除
        # 因为多头分的就是词嵌入的维度 每个头分到的词嵌入维度量为 embedding_dim/head
        assert embedding_dim % head == 0

        # 每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head

        # 传入头数h
        self.head = head

        # 线性层对象通过nn的Linear实例化
        # 最后一维线性变换不会改变张量大小->内部变换矩阵为方阵 参数值为embedding_dim
        # 使用clones克隆4个:Q, K, V, 最后拼接concat的矩阵
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # self.attn代表最后得到的注意力张量,初始化时还没有结果所以为None.
        self.attn = None

        # self.dropout对象-nn中的Dropout实例化,置0比率为传进来的参数dropout.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        :param query: 文本 三维向量 第3(2)维的维度为d_model  a x b x d_model
        :param key: 关键词 三维向量 第3(2)维的维度维d_model
        :param value: 思考的答案 三维向量 第3(2)维的维度维d_model
        :param mask: 掩码张量 三维向量 默认为None a x b x b
        :return:
        """
        # 如果存在掩码张量mask 拓展维度后 1 x a x b x b
        if mask is not None:
            mask = mask.unsqueeze(0)

        # batch_size:query尺寸的第1个数字,代表有多少条样本 -- a
        batch_size = query.size(0)

        # ------------多头处理环节------------
        # 1 QKV经过线性层:使用zip将3个线性层组合,然后for循环把QKV传到线性层中
        # 2 多头:分割词嵌入维度d_model,使用view进行维度重塑,多加了一个维度h代表头数，
        #   初始维度为[batch_size x len x d_model]
        #   目标维度为[batch_size x h x len x d_k] 由于len未知 用-1让len大小自适应
        #   自适应第二维转置让len和d_k相邻的目标:
        #       注意力机制使用倒数1、2维得到输出-找到词义和句子位置的关系
        #       同一个字词的词嵌入维度均分到每个头上 避免多份均分摊在一个头上
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]

        # 将四维多头的QKV传到attention中 同时传入mask和dropout
        x, self.attn = \
            attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算之后 得到QKV的四维张量 需要转化为三维
        # 因此执行逆操作: 2、3维转置,然后contiguous - 让转置后的张量应用view
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 3 使用线性层列表中的最后一个线性层concat进行线性变换得到输出.
        return self.linears[-1](x)


# 测试

# 实例化参数
head = 8  # 8个头
embedding_dim = 512
dropout = 0.2

# 输入参数的初始化 如果全部相同 - 自注意力机制
from transformer.packages import device as device
from transformer.input.positionalEncoding import pe_result as pe_result
query = key = value = pe_result  # pe_result 是经过位置编码的x张量 2 x 4 x 512

mask = Variable(torch.zeros(8, 4, 4))  # 8 x 4 x 4
if device == torch.device('cuda:0'):
    mask = mask.cuda()
mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha.to(device)
# print('input mask-shape:', mask.shape)
mha_result = mha(query, key, value, mask)
# print(mha_result)
