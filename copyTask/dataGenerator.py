# 数据集生成器


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


import numpy as np

# 导入工具包Batch, 它能够对原始样本数据生成对应批次的掩码张量
# 下载源码后暴力放到project里面了
from pyitcast.transformer_utils import Batch


def data_generator(V, batch, num_batch):
    """
    该函数用于随机生成copy任务的数据
    :param V: 随机生成数字的最大值+1
    :param batch: 每次输送给模型更新一次参数的数据量
    :param num_batch: 输送num_batch次完成一轮
    :return:
    """
    for i in range(num_batch):
        # np.randint 随机生成[1,V)的整数 - 生成[batch x 10]的矩阵
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data = data.long()
        if device == torch.device('cuda:0'):
            data = data.cuda()
        # print('data', i, data)
        # 使矩阵中的第一列(0)数字都为1 - 起始标志列
        # 解码器第一次解码的时候使用起始标志列作为输入
        data[:, 0] = 1

        # copy任务中 source和target是完全相同的 并且数据样本作用变量不需要要求梯度
        # 因此requires_grad为False
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)
        if device == torch.device('cuda:0'):
            source = source.cuda()
            target = target.cuda()
        # 使用Batch对source和target进行对应批次的掩码张量生成, 最后使用yield返回
        yield Batch(source, target)


# 输入参数
# 将生成0-10的整数
V = 11
# 每次给模型20个数据进行参数更新
batch = 20
# 连续30次完成全部数据的遍历, 也就是1轮
num_batch = 30  #

# # 调用
# if __name__ == '__main__':
#     res = data_generator(V, batch, num_batch)
#     print(res)

# ----------------------------------------------------------------第二步transformer
# 导入优化器工具包get_std_opt, 该工具用于获得标准的针对Transformer模型的优化器
# 该标准优化器基于Adam优化器, 使其对序列到序列的任务更有效.
from pyitcast.transformer_utils import get_std_opt
from transformer.model.makeModel import make_model
# 导入标签平滑工具包,标签平滑的作用:小幅度的改变原有标签值的值域
# 因为理论上即使是人工的标注数据也可能并非完全正确, 会受到一些外界因素的影响而产生一些微小的偏差
# 使用标签平滑来弥补这种偏差, 减少模型对某一条规律的绝对认知, 以防止过拟合
from pyitcast.transformer_utils import LabelSmoothing
# 导入损失计算工具包, 该工具能够使用标签平滑后的结果进行损失的计算,
# 损失的计算方法可以认为是交叉熵损失函数.
from pyitcast.transformer_utils import SimpleLossCompute

# 使用make_model获得model
model = make_model(V, V, N=2)  # 返回的是gpu上的模型
# print(next(model.parameters()).is_cuda)

# 使用get_std_opt获得模型优化器
model_optimizer = get_std_opt(model)  # function

# 使用LabelSmoothing获得标签平滑对象
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
criterion.to(device)

# 使用SimpleLossCompute获得利用标签平滑结果的损失计算方法
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)  # function


# ----------------------------------------------------------------第三步训练评估
# 导入模型单轮训练工具包run_epoch, 该工具将对模型使用给定的损失函数计算方法进行单轮参数更新.
# 并打印每轮参数更新的损失结果.
from pyitcast.transformer_utils import run_epoch

def run(model, loss, epochs=10):
    """
    :param model: 将要进行训练的模型
    :param loss: 损失计算方法
    :param epochs: 模型训练的轮数
    :return:
    """

    for epoch in range(epochs):
        # 模型使用训练模式, 所有参数将被更新 反向传播算梯度的
        model.train()
        # 训练时, batch_size是20 参数是 训练数据 模型 损失函数 -- 一个轮次
        run_epoch(data_generator(V, 8, 20), model, loss)

        # 模型使用评估模式, 参数将不会变化
        model.eval()
        # 评估时, batch_size是5
        run_epoch(data_generator(V, 8, 5), model, loss)


# 输入参数
# 进行10轮训练
epochs = 10
# model和loss都是来自上一步的结果
# print('begin training ----------------------------------------')
# run(model, loss, epochs)

# ----------------------------------------------------------------第四步贪婪解码
# 导入贪婪解码工具包greedy_decode, 该工具将对最终结进行贪婪解码
# 贪婪解码的方式是每次预测都选择概率最大的结果作为输出,不一定获得最优答案,但是效率最高
from pyitcast.transformer_utils import greedy_decode
from transformer.packages import device as device

# 修改之前的run函数
def run2(model, loss, epochs=10):
    """
    :param model: 训练的模型
    :param loss: 损失函数的计算方法
    :param epochs: 总共的计算轮次
    :return:
    """
    for epoch in range(epochs):
        model.train()  # 训练模式 参数更新
        # 运行一个轮次 参数为 训练数据 模型 损失函数 V, batch, num_batch
        run_epoch(data_generator(V, 8, 40), model, loss)  # 原batch是 20 修改为40
        model.eval()  # 评估模式 参数不变 V, batch, num_batch  # 原batch是5 修改为10
        run_epoch(data_generator(V, 8, 10), model, loss)

    # 模型进入测试模式
    model.eval()

    # 假定的输入张量
    source = Variable(torch.LongTensor([[1,3,2,5,4,6,7,8,9,10]]))

    # 定义源数据掩码张量, 此处1代表不遮掩, 相当于对源数据没有任何遮掩.
    source_mask = Variable(torch.ones(1, 1, 10))

    # 放到gpu中
    if device == torch.device('cuda:0'):
        source = source.cuda()
        source_mask = source_mask.cuda()

    # 最后将model, src, src_mask, 解码的最大长度限制max_len, 默认为10
    # 以及起始标志数字, 默认为1, 我们这里使用的也是1
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


epochs = 40  # ##################################################!!!!!!!epochs

if __name__ == '__main__':
    print('begin training and eval----------------------------------------')
    run2(model, loss, epochs)


# 保存模型
path = './copyModel2'
torch.save(model.state_dict(), path)



