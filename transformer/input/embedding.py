# 词嵌入模型

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


#####################################################1.1文本嵌入层
from transformer.packages import device as device

# 定义Embedding层实现文本嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        :param d_model: 词嵌入维度
        :param vocab: 词表大小
        """
        super(Embeddings, self).__init__()
        # 调用nn中预定义层Embedding 获得词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        前向传播逻辑，当传给该类的实例化对象参数的时候，自动调用该函数
        :param x: Embedding层为首层，代表输入给模型的文本通过词汇映射后的张量
        :return:
        """
        # 将x传给Embedding层self.lut并与根号下self.d_model相乘作为返回结果
        # 缩放作用，维度越大
        return self.lut(x) * math.sqrt(self.d_model)


# 演示
# embedding = nn.Embedding(10, 3) #vocab d_model
# input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]) # 两行四列
# res = embedding(input)
# print(res)
'''
# 每个数字被映射为三个数字
# 2 * 4 * 3 - 2 * 4 * d_model
tensor([[[ 2.5577,  0.4959, -2.0518], # 代表1
         [ 0.0447,  0.5555,  0.9907], # 代表2
         [ 0.1959, -1.9102,  0.4996], # 代表4
         [ 0.8895,  1.0859,  1.2877]], # 代表5

        [[ 0.1959, -1.9102,  0.4996],
         [-0.2384,  0.1826,  0.1482],
         [ 0.0447,  0.5555,  0.9907],
         [ 1.0254, -1.9821, -1.5096]]], grad_fn=<EmbeddingBackward>)

Process finished with exit code 0
'''

# embedding = nn.Embedding(10, 3, padding_idx=0) # 作用在于让 0 所在值为0
# input = torch.LongTensor([[0, 2, 0, 5]])
# embedding(input)
'''
tensor([[[ 0.0000,  0.0000,  0.0000],
         [ 0.1535, -2.0309,  0.9315],
         [ 0.0000,  0.0000,  0.0000],
         [-0.1655,  0.9897,  0.0635]]])
'''

# 测试
d_model = 512  # 词嵌入维度
vocab = 1000  # 词表大小
# 输入x是一个使用Variable封装的长整型张量, 形状是2 x 4
# x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)  # cpu
# emb = emb.cuda()
# print(next(emb.parameters()).is_cuda)  # False
# embr = emb(x)
# print(embr)
# print(embr.shape)

'''
tensor([[[-15.4299, -17.7916,  -8.9287,  ...,  14.3828,   4.4558, -18.0698],
         [ 40.8252,   1.6852, -26.8057,  ...,  16.4183,  -6.3022,   2.3319],
         [ -8.7622, -41.9216,  -6.2061,  ..., -37.4488, -39.5422, -14.5541],
         [-19.8698, -14.9421,  24.3235,  ..., -44.8080,   9.1618,   3.5722]],

        [[  8.3046,  26.9700,   1.9386,  ..., -15.4103, -19.7201,  19.4218],
         [ 20.7322,  11.4747, -33.0307,  ...,  28.0594, -21.4225, -68.9587],
         [-28.9082,   7.7140,   8.7951,  ...,  -2.4696,  27.7329,   7.1058],
         [  9.8008,  -8.0743,  30.7722,  ...,  15.2633, -24.3229, -14.5709]]],
       grad_fn=<MulBackward0>)
torch.Size([2, 4, 512])
'''
