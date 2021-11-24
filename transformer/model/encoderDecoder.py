# transformer构建代码


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


# 编码器-解码器结构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param source_embed: 源数据嵌入函数
        :param target_embed: 目标数据嵌入函数
        :param generator: 生成器对象 - 线性层和softmax
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """
        :param source: 源数据 - 经过词嵌入位置编码的数据
        :param target: 目标数据
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        :return:
        """

        # 在函数中, 将source, source_mask传入编码函数, 得到结果后,
        # 与source_mask，target，和target_mask一同传给解码函数.
        # 视频中返回的值没有经过generator 我感觉好奇怪 我觉得应该要加上去
        # print('encoderDecoder.py  device')
        # print('source', source.device)
        # print('target', target.device)
        # print('source_mask', source_mask.device)
        # print('target_mask', target_mask.device)
        return self.decode(self.encode(source, source_mask), source_mask,
                           target, target_mask)
        # return self.generator(self.decode(self.encode(source, source_mask), source_mask,
        #                                   target, target_mask))

    def encode(self, source, source_mask):
        """
        编码函数
        :param source: 源数据
        :param source_mask: 源数据掩码
        :return:
        """
        #  词嵌入+位置编码
        # print('encoderDecoder.py')
        # print('self.src_embed', next(self.src_embed.parameters()).is_cuda)
        # print('self.encoder', next(self.encoder.parameters()).is_cuda)

        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        """
        :param memory: 编码器的输出
        :param source_mask: 源数据掩码
        :param target:
        :param target_mask: 目标数据掩码
        :return:
        """
        # 参数:目标数据嵌入表示x 编码器输出memory 源数据掩码 目标数据掩码
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


# 测试
# 实例化参数
from transformer.encoder.encoder import en as en
from transformer.decoder.decoder import de as de
from transformer.output.output import gen as gen
from transformer.packages import device as device


vocab_size = 1000
d_model = 512
encoder = en
decoder = de
source_embed = nn.Embedding(vocab_size, d_model)
source_embed.to(device)
target_embed = nn.Embedding(vocab_size, d_model)
target_embed.to(device)
generator = gen

# 输入参数
# 假设源数据与目标数据相同, 实际中并不相同
content = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]], device=device)
content = content.long()
source = target = Variable(content)
# 假设src_mask与tgt_mask相同，实际中并不相同
source_mask = target_mask = Variable(torch.zeros(8, 4, 4))
if device == torch.device('cuda:0'):
    source = source.cuda()
    target = target.cuda()
    source_mask = source_mask.cuda()
    target_mask = target_mask.cuda()

# 调用
ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
ed.to(device)
# 编码器解码器输出
ed_result = ed(source, target, source_mask, target_mask)  # [2 x 4 x 512]
# print(ed_result)
# print(ed_result.shape)
