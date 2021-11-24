# 测试一下


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
# ----------------------------------------------------------------第一步生成数据
from transformer.packages import device as device
from pyitcast.transformer_utils import greedy_decode
from transformer.model.makeModel import make_model

def test(source, model):
    """
    进行测试
    :param source: 假定的输入张量
    :param model: 经过训练的模型
    :return:
    """
    print('begin test----------------------------------------------')
    source_mask = Variable(torch.ones(1, 1, 10))

    # 放到gpu中
    if device == torch.device('cuda:0'):
        source = source.cuda()
        source_mask = source_mask.cuda()
    model.eval()
    # 最后将model, src, src_mask, 解码的最大长度限制max_len, 默认为10
    # 以及起始标志数字, 默认为1, 我们这里使用的也是1
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print('aim', source)
    print('res', result)

# source = Variable(torch.LongTensor([[1, 3, 2, 5, 4, 6, 7, 8, 9, 10]]))
# source = Variable(torch.LongTensor([[1, 4, 1, 1, 6, 6, 7, 9, 9, 6]]))
# source = Variable(torch.LongTensor([[1, 8, 7, 6, 5, 4, 3, 2, 1, 9]]))
source = Variable(torch.LongTensor([[1, 2, 2, 2, 3, 3, 3, 7, 7, 7]]))

# 加载模型
path = './copyModel2'
V = 11
model = make_model(V, V, N=2)  # 返回的是gpu上的模型
model.load_state_dict(torch.load(path))
# test(source, model)
print('real source', source)
for i in range(4):
    t = [int(n) for n in input().split(' ')]
    source = Variable(torch.tensor(t).unsqueeze(0).long())
    test(source, model)
