from pyitcast.transformer_utils import LabelSmoothing
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


# 实例化一个crit对象
# 第一个参数size代表目标数据词汇总数，也是模型最后一层得到张量的最后一位大小
# 第二个参数表示要将那些tensor中的数字替换成0，padding_idx=0表示不替换
# 第三个参数smoothing表示标签平滑程度，如原来标签的表示值为1，则平滑后值域为[1-smoothing, 1+smoothing]
crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)

# 假定一个任意模型最后输出预测结果和真实结果
predict = Variable(torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                      [0, 0.2, 0.7, 0.1, 0],
                                      [0, 0.2, 0.7, 0.1, 0]]))

# 标签的表示值为0，1，2
target = Variable(torch.LongTensor([2, 1, 0]))

# 将predict，target传入对象中
crit(predict, target)

# 绘制标签平滑图像
plt.imshow(crit.true_dist)
plt.show()
