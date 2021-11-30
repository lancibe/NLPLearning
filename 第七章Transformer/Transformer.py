import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# torch中变量封装函数Variable
from torch.autograd import Variable
import matplotlib.pyplot as plt


# 定义Embeddings类来实现文本嵌入层，这里s说明有两个一模一样的嵌入层，共享参数
# 继承nn.Module，这样就有标准层的一些形式，我们也可以理解为一种模式，自己实现的所有层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        类初始化函数
        :param d_model:词嵌入维度
        :param vocab:词表大小
        """
        super(Embeddings, self).__init__()
        # 调用预定义层获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        # 将d_model传入类中
        self.d_model = d_model

    def forward(self, x):
        """
        前向传播逻辑
        :param x: 因为Embedding层是首层，所以代表输入给模型的文本通过词汇映射后的张量
        :return: 将x传给self.lut冰与根号下self.d_model相乘作为结果返回
        """
        return self.lut(x) * math.sqrt(self.d_model)


d_model = 512
vocab = 1000

x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)
embr = emb(x)


# print('embr:', embr)
# print(embr.shape)


# 定义位置编码器类，同样把他看作是一个层，因此会继承nn.Module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        位置编码器的初始化函数
        :param d_model: 词嵌入维度
        :param dropout: 置零比率
        :param max_len: 每个句子最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 实例化Dropout层
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵，是一个0阵，大小是max_len*d_model
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵
        # 首先使用arange方法获得一个连续自然数向量，然后扩展维度变成max_len*1
        position = torch.arange(0, max_len).unsqueeze(1)

        # 有了绝对位置矩阵、位置编码矩阵，现在需要进行连接。
        # 根据他们两个的形状，可以创建一个1*d_model形状的变换矩阵div_term
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 此时pe是一个二维矩阵，为得到embedding的输出，需要扩展一个维度
        pe = pe.unsqueeze(0)

        # 最后把pe位置编码矩阵注册成buffer
        # 我们认为buffer是对模型效果有帮助的、但又不是模型结构中超参数或者参数，不需要随着优化步骤迭代
        # 注册之后就可以在模型保存后重加载时和模型结构与参数一同被加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向函数
        :param x:文本序列的词嵌入表示
        :return:经处理的x
        """
        # 我们默认的max_len太大了，一般不会有句子超过5000词汇。所以需要进行与输入张量的适配
        # 最后再使用Variable封装，使其与x的样式相同。
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # 最后再使用self.dropout对象进行“丢弃”操作，它会使某些数值失效，它的参数p表示失效百分比
        return self.dropout(x)


d_model = 512
dropout = 0.1
max_len = 60

x = embr
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)


# # print(pe_result)
# # print(pe_result.shape)


# import matplotlib.pyplot as plt
#
# # 创建画布
# plt.figure(figsize=(15, 5))
# # 实例化对象
# pe = PositionalEncoding(20, 0)
# # 向pe传入被Variable封装的tensor，这样pe会直接执行forward函数
# # 且这个tensor里数值都是0，被处理后相当于位置编码张量
# y = pe(Variable(torch.zeros(1, 100, 20)))
#
# # 定义画布横纵坐标
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
#
# # 在画布上填写维度提示信息
# plt.legend(['dim %d' % p for p in [4, 5, 6, 7]])
# plt.show()


def subsequent_mask(size):
    """
    生成向后的掩码张量
    :param size: 掩码张量后两个维度大小
    :return: 新的张量
    """

    # 首先定义掩码张量的形状
    attn_shape = (1, size, size)
    # 使用np.ones方法向这个形状中加入1元素，形成上三角阵。
    # 为节省空间，再使其中的数据类型变成无符号八位整形unit8
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 最后将numpy类型转为tensor，并做一个1-的操作。
    # 这其实进行了三角阵的反转，每个元素都会被1减。
    return torch.from_numpy(1 - subsequent_mask)


# plt.figure(figsize=(5, 5))
# plt.imshow(subsequent_mask(20)[0])
# plt.show()


def attention(query, key, value, mask=None, dropout=None):
    """
    注意力机制的实现，输入分别是query，key，value，mask，dropout
    :param query: Q
    :param key: K
    :param value: V
    :param mask: 掩码张量
    :param dropout: nn.Dropout层的实例化对象，默认为None
    :return: 返回公式运行的结果和注意力张量
    """
    # 首先取query的最后一维的大小，一般就等于词嵌入维度，命名为d_k
    d_k = query.size(-1)
    # 根据注意力公式进行计算
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 判断是否使用掩码张量
    if mask is not None:
        # 使用tensor的masked_fill方法，将掩码张量和scores张量每个位置一一比较，如果掩码张量处于0
        # 则对应的scores张量用-1e9来替换
        scores.masked_fill_(mask == 0, -1e9)

    # 对scores最后一维进行softmax操作
    p_attn = F.softmax(scores, dim=-1)

    # 之后判断是否使用dropout进行随机置零
    if dropout is not None:
        # 将p_attn传入dropout进行丢弃处理
        p_attn = dropout(p_attn)

    # 最后根据公式将p_attn与V相乘。
    return torch.matmul(p_attn, value), p_attn


# query = key = value = pe_result
# mask = Variable(torch.zeros(2, 4, 4))
# attn, p_attn = attention(query, key, value, mask)
# print('attn:', attn)
# print('p_attn:', p_attn)


# 深度拷贝
import copy


# 克隆函数
def clones(module, N):
    """
    用于生成相同网络层的克隆函数
    :param module:要克隆的目标网络层
    :param N:需要克隆的数量
    :return:存入nn.ModuleList列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 使用一个类来实现多头注意力机制处理
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """
        类初始化函数
        :param head:头数
        :param embedding_dim:词嵌入维度
        :param dropout: 置零比率
        """
        super(MultiHeadedAttention, self).__init__()

        # 在函数中，先使用一个测试中常用的assert语句判断h能否被d_model整除
        # 这是因为我们之后要给每个头分配等量的词特征，也就是embedding_dim/head个
        assert embedding_dim % head == 0

        # 得到每个头获得分割词向量维度d_k
        self.d_k = embedding_dim // head
        # 传入头数
        self.head = head
        # 获得线性层对象，通过nn的Linear实例化。它内部变换矩阵是embedding_dim x embedding_dim
        # 需要四个，因为QKV各需要一个，最后拼接的矩阵还要一个
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # self.attn为None，他代表最后得到的注意力张量，现在还没有结果所以为None
        self.attn = None
        # 最后一个self.dropout对象，通过nn中的Dropout实例化而来
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向逻辑函数
        :param query:Q
        :param key:K
        :param value:B
        :param mask:可能需要的掩码张量
        :return:多头注意力结构的输出
        """
        # 如果存在掩码张量
        if mask is not None:
            # 扩展维度，代表多头中的第n头
            mask = mask.unsqueeze(1)

        # 接着获得一个batch_size变量，他是query尺寸的第一个数字，代表有多少样本
        batch_size = query.size(0)

        # 多头处理环节
        # 首先利用zip将输入QKV与三个线性层组到一起，然后使用for循环，将输入QKV分别传到线性层中，
        # 做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多加了一个维度h，代表头数，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值.然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，
        # 从attention函数中可以看到，利用的是原始输入的倒数第一和第二维.这样我们就得到了每个头的输入.
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]

        # 得到了每个头的输入后，接下来就是传入attention中
        # 直接调用前面的attention函数
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，
        # 因此这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用contiguous方法，
        # 这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用，
        # 所以，下一步就是使用view重塑形状，变成和输入形状相同.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后使用线性列表中最后一个线性层对输入进行线性变换得到最终的多头注意力结构的输出
        return self.linears[-1](x)


# 实例化参数
head = 8
embedding_dim = 512
dropout = 0.2

# 输入参数
query = key = value = pe_result
mask = Variable(torch.zeros(2, 4, 4))

mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)


# print(mha_result)
# print(mha_result.shape)


# 通过类PositionwiseFeedForward来实现前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化函数
        :param d_model:第一个线性层输入维度，也就是第二个线性层输出维度
        :param d_ff: 第二个线性层输入维度
        :param dropout:置零比率
        """
        super(PositionwiseFeedForward, self).__init__()

        # 首先按照预期使用了nn实例化了两个线性层对象
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        # 实例化dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向逻辑函数
        :param x: 来自上一层的输出
        :return: 经过两个线性层，先经过第一个，并使用relu函数激活，然后经过丢弃，进入第二个线性层
        """
        return self.w2(self.dropout(F.relu(self.w1(x))))


d_model = 512
d_ff = 64
dropout = 0.2

x = mha_result
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)


# print(ff_result)
# print(ff_result.shape)


# 通过LayerNorm实现规范化层的类
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        初始化函数
        :param features:词嵌入维度
        :param eps:在规范化公式的分母出现，防止分母为零
        """
        super(LayerNorm, self).__init__()

        # 初始化两个张量，一个全为1一个全为0
        # 最后使用nn.parameter封装，代表他们是模型的参数
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        # 把eps传进类中
        self.eps = eps

    def forward(self, x):
        """
        前向函数
        :param x:来自上一层的输出
        :return:规范化后的参数
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a2 * (x - mean) / (std + self.eps) + self.b2


features = d_model
eps = 1e-6
# 输入x来自前馈全连接层的输出
x = ff_result
ln = LayerNorm(features, eps)
ln_result = ln(x)
# print(ln_result)
# print(ln_result.shape)


# 使用SubLayerConnection来实现子层连接结构的类
class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """
        初始化函数
        :param size:词嵌入维度大小
        :param dropout:置零比率
        """
        super(SubLayerConnection, self).__init__()
        # 实例化规范化对象self.norm
        self.norm = LayerNorm(size)
        # 又使用nn中预定义的dropout实例化一个dropout对象
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, subLayer):
        """
        前向逻辑函数
        :param x: 接受上一层的输入
        :param subLayer: 子层参数
        :return:最终子层连接输出
        """
        # 先规范化，然后将结果传入子层处理，再对子层进行dropout操作。随机停止一些网络中神经元的作用，防止过拟合，
        # 因为存在跳跃连接，所以将输入x与dropout后的子层输出结果相加作为最终子层连接输出
        return x + self.dropout(subLayer(self.norm(x)))
