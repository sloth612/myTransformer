# 位置编码器类


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


#####################################################1.1位置编码器
from transformer.input.embedding import Embeddings


# 位置编码器类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model: 词嵌入维度
        :param dropout: 置0比率
        :param max_len: 每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 调用nn中预定义的Dropout层 获得Dropout对象self.dropout
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵, 它是一个0阵，[max_len x d_model]
        # 一共 max_len 个词, 每个词映射为 [1 x d_model]
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵, 词汇的绝对位置就是用其索引表示.
        # arange获得连续自然数向量shape:[max_len] 范围为[0, max_len-1]
        # unsqueeze升维使向量成为矩阵shape:[max_len x 1]
        # 最后的矩阵中 max_len每一个代表一个词汇，值为对应的位置？？？？？
        position = torch.arange(0, max_len).unsqueeze(1)  # max_len x 1
        position = position.float()  # 原position是Long div_term是float 需要统一格式
        # 绝对位置矩阵position[max_len x 1]初始化之后加入到位置编码矩阵pe[max_len x d_model]中
        # 最简单思路:先将绝对位置矩阵变换成max_len x d_model形状然后覆盖原来的初始位置编码矩阵
        # 因此需要一个变化矩阵div_term, 跳跃式初始化, 有如下要求:
        # 1. 形状为[1 x d_model] : div_term [max_len x 1] x [1 x d_model] = [max_len x d_model]
        # 2. 能够将自然数的绝对位置编码position缩放成足够小的数字，有助于之后的梯度下降过程中更快的收敛
        # div_term初始化过程:
        # 1. arange 获得自然数向量,范围为[0,d_model] 步长为2. shape:[d_model/2] 注意转化为float,因为exp不支持Long
        # 2. -(math.log(10000.0) / d_model作用在于缩放, 为了梯度下降更快收敛
        # 3. 在第一步中只初始化了一半的向量,因此需要初始化两次
        #   第一次 偶数列 正弦波; 第二次 奇数列 余弦波; div_term shape:[d_model/2]可以自动处理为[1 x d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # max_len x d_model/2
        pe[:, 1::2] = torch.cos(position * div_term)

        # embedding 输出为 ? x ? x d_model 是一个三维向量
        # pe目前shape: max_len x d_model 是一个二维向量, 因此需要升维
        pe = pe.unsqueeze(0)  # pe shape:[1 x max_len x d_model]

        # 最后把pe位置编码矩阵注册成模型的buffer
        # buffer:对模型效果有帮助的,但不是模型结构中超参数或者参数,不需要随着优化步骤进行更新的增益对象.
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数一同被加载.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: 表示文本序列的词嵌入表示, 是一个三维向量,第一维(0)?第二维(1)是句子长度,第三维(2)是d_model
        :return:
        """
        # pe位置编码矩阵 shape: 1 x max_len x d_model
        # 在相加之前我们对pe做一些适配工作， 将这个三维张量的第二维也就是句子最大长度的那一维将切片到与输入的x的第二维相同即x.size(1)，
        # 因为我们默认max_len为5000一般来讲实在太大了,很难有一条句子包含5000个词汇，所以要进行与输入张量的适配.
        # 最后Variable封装，使其与x的样式相同，但位置并不需要进行梯度求解,因此requires_grad为false.
        # self.pe[:, :x.size(1)] == self.pe[:, :x.size(1), :]
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # 最后self.dropout进行'丢弃'操作, 并返回结果.
        return self.dropout(x)


# 测试
from transformer.packages import device as device

vocab = 1000  # 词表大小
d_model = 512
dropout = 0.1
max_len = 60

# 处理经过词嵌入模型的张量
emb = Embeddings(d_model, vocab)  # 是Embedding的模型
# emb = emb.cuda()
emb.to(device)  # 把模型转移到GPU上
# print('model emb: ', next(emb.parameters()).is_cuda)  # 判断模型是否在GPU上
#  myTensor是输入的张量 2 x 4
myTensor = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]], device=device)
myTensor = myTensor.long()  # 转化为LongTensor
x = Variable(myTensor)  # 输入的x张量
if device == torch.device('cuda:0'):
    x = x.cuda()
x = emb(x)  # 经过词嵌入模型处理后的x张量 x-shape:2 x 4 x d_model = 2 x 4 x 512
# print(x.device)  # 判断x张量的位置

# 处理位置编码器
pe = PositionalEncoding(d_model, dropout, max_len)  # 创建位置编码器实例
pe.to(device)  # PositionalEncoding 的模型
pe_result = pe(x)  # 在gpu上
# print(pe_result)
# print(pe_result.shape)  # x张量经过位置编码器处理后的结果 2 x 4 x 512

