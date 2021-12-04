# 数学计算工具包
import math

# torch相关
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch中经典文本数据集有关的工具包
import torchtext

# torchtext中数据处理工具，该函数用于英文分词
from torchtext.data.utils import get_tokenizer

# 已经构建完成的TransformerModel
from pyitcast.transformer import TransformerModel


# 创建语料域, 语料域是存放语料的数据结构,
# 它的四个参数代表给存放语料（或称作文本）施加的作用.
# 分别为 tokenize,使用get_tokenizer("basic_english")获得一个分割器对象,
# 分割方式按照文本为基础英文进行分割.
# init_token为给文本施加的起始符 <sos>给文本施加的终止符<eos>,
# 最后一个lower为True, 存放的文本字母全部小写.
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

# 然后使用torchtext的数据集方法导入数据
# 并切分为训练文本，验证文本，测试文本，并对这些文本施加刚刚创建的语料域
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)


# 将训练集文本数据构建一个vocab对象
# 可以用vocab对象的stoi方法统计文本共包含的不重复词汇总数
TEXT.build_vocab(train_txt)

# 然后选择设备
device = torch.device("cuda")


def batchify(data, bsz):
    """
    该函数用于将文本数据映射成连续数字，并转换指定的样式，指定的样式可参考图片
    :param data: 之前得到的文本数据(train_txt, val_txt, test_txt)
    :param bsz: batch_size，每次模型更新参数的数据量
    :return: 处理之后的数据
    """
    # 先将单词映射成连续对应的数字
    data = TEXT.numericalize([data.examples[0].text])

    # 接着用数据词汇总数除bsz并取整得到一个nbatch代表需要多少次batch后遍历所有数据
    nbatch = data.size(0) // bsz

    # 使用narrow方法对不规整剩余数据进行删除
    # 第一个参数代表横轴删除还是纵轴删除，0为横，1为纵
    # 第二个和第三个参数代表保留开始轴到结束轴的数值，类似于切片
    data = data.narrow(0, 0, nbatch*bsz)

    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


# 用batchify来处理训练数据，验证数据以及测试数据
# 训练数据的bsz
batch_size = 20

# 验证和测试数据（统称为评估数据）的bsz
eval_batch_size = 10

# 获得处理后的数据
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)




