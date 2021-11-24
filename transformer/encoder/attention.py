# 注意力attention


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
from transformer.packages import device as device


def attention(query, key, value, mask=None, dropout=None):
    """
    :param query: 文本 最后一维的维度为d_model 三维的情况下 a x b x d_model
    :param key: 关键词 同上
    :param value: 思考的答案 同上
    :param mask: 掩码张量 三维情况下 a x b x b
    :param dropout: nn.Dropout层的实例化对象
    :return: 三维情况下 a x b x d_model
    """
    # 取query的最后一维的大小:一般情况下为词嵌入维度d_model, 命名为d_k
    d_k = query.size(-1)
    # 注意力公式:
    # Attention(Q,K,V)=softmax(\dfrac{QK^T}{\sqrt{d_k}})V
    # 1矩阵转置乘法QK^T 2规范化\sqrt{d_k} 3softmax 4矩阵乘法softmax(...)V
    # 1矩阵转置 key.transpose(-2,-1) key将最后两个维度进行转置,从而让QK^T可以相乘
    # 2矩阵乘法 torch.matmul(query,K^T) 乘法结果第3(2)维的维度为d_model = d_k
    #           torch.matmul永远是最后两维做乘法
    # 3规范化 /math.sqrt(d_k) 除以缩放系数根号下d_k:缩放点积注意力计算
    # 得到注意力得分张量scores
    # [2 x 4 x 512] x [2 x 512 x 4] = [2 x 4 x 4]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('query.shape:', query.shape)  # [2 x 8 x 4 x 64]
    # print('key.shape:', key.shape)  # [2 x 8 x 4 x 64]
    # print('scores.shape:', scores.shape)  # [2 x 8 x 4 x 4]
    # 判断是否使用掩码张量
    if mask is not None:
        # print('mask.shape:', mask.shape)  # [1 x 8 x 4 x 4]
        # 使用tensor的masked_fill方法, 如果掩码张量处为0:对应值用-1e9这个值来替换
        scores = scores.masked_fill(mask == 0, -1e9)

    # softmax, 参数1是softmax对象, 参数2是softmax对象的目标维度
    # 在最后一维上执行 - 输入语句中字词的数量
    p_attn = F.softmax(scores, dim=-1)  # a x b x b

    # 判断是否使用dropout进行随机置0
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 4将p_attn与value张量相乘获得最终的query注意力表示, 并返回注意力张量
    # [a x b x b] x [a x b x d_model] = [a x b x d_model]
    return torch.matmul(p_attn, value), p_attn


# 自注意力测试
from transformer.input.positionalEncoding import pe_result as pe_result

query = key = value = pe_result  # pe_result 是经过位置编码的x张量 2 x 4 x 512

# 带有mask 2 x 4 x 4的零张量
# mask = Variable(torch.zeros(2, 4, 4)).cuda()

# attn, p_attn = attention(query, key, value)  # 不带mask
# attn, p_attn = attention(query, key, value, mask=mask)  # 带mask
# print('attn', attn)  # 2 x 4 x 512
# print('p_attn', p_attn)  # 2 x 4 x 4




