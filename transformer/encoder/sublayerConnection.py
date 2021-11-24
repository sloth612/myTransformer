# 子层连接结构SublayerConnection


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
from transformer.encoder.layerNorm import LayerNorm


# 子层连接结构
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """
        :param size: 词嵌入维度大小
        :param dropout:  置零比率 节点数随机抑制
        """
        super(SublayerConnection, self).__init__()
        # 实例化了规范化层self.norm
        self.norm = LayerNorm(size)
        # nn.Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        :param x: 接收上一个层或者子层的输入
        :param sublayer: 该子层连接中的子层函数
        :return:
        """
        # 规范化 - 子层 - dropout防止过拟合 - add操作
        # 因为是残差连接 所以需要输入x和dropout后的子层输出结构相加作为输出
        return x + self.dropout(sublayer(self.norm(x)))


# # 测试
# size = d_model = 512
# dropout = 0.2
# head = 8
#
# # 输入参数
# from transformer.input.positionalEncoding import pe_result as pe_result
# from transformer.encoder.multihead import MultiHeadedAttention as MultiHeadedAttention
# from transformer.packages import device as device

# x = pe_result  # input位置编码后的数据 此时还没经过多头自注意力
# mask = Variable(torch.zeros(8, 4, 4)).cuda()
#
# # 假设子层中装的是多头自注意力层 实例化
# self_attn = MultiHeadedAttention(head, d_model)  # cpu
# self_attn.to(device)  # gpu
#
# # 使用lambda获得一个函数类型的子层
# sublayer = lambda x: self_attn(x, x, x, mask)  # query, key, value, mask
#
# # 调用
# sc = SublayerConnection(size, dropout)  # cpu
# sc.to(device)
# sc_result = sc(x, sublayer)  # 在残差连接中 规范化 和 相加
# # print(sc_result)

