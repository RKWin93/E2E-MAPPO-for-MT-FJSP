import torch
import numpy as np
import torch.nn as nn



"""
正交初始化网络参数：
1、我们一般在初始化actor网络的输出层时，会把gain设置成0.01，
2、actor网络的其他层和critic网络都使用Pytorch中正交初始化默认的gain=1.0。

（1）用均值为0，标准差为1的高斯分布初始化权重矩阵，
（2）对这个权重矩阵进行奇异值分解，得到两个正交矩阵，取其中之一作为该层神经网络的权重矩阵。
"""
# orthogonal init
def orthogonal_init(model, gain=1.0):
    for layer in model.modules():  # 将`Sequential对象中的层对象给遍历出来
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.orthogonal_(layer.weight, gain=gain)  # 权重
            nn.init.constant_(layer.bias, 0)  # 偏置

"""
学习率衰减可以一定程度增强训练后期的平稳性，提高训练效果。这里我们采用线性衰减学习率的方式，使lr从初始的3e-4，随着训练步数线性下降到0
total_steps是不间断累加的，记录总共跑了多少step
max_train_steps = episode * n_total_task
"""
def lr_decay(total_steps, optimizer_actor, optimizer_critic):
    max_train_steps = configs.episode_num * configs.n_total_task  # 我的场景。每一个子任务都要选一次m，所以总任务数 * episode数量
    lr_a_now = configs.LR * (1 - total_steps / max_train_steps)
    lr_c_now = configs.LR * (1 - total_steps / max_train_steps)
    for p in optimizer_actor.param_groups:
        p['lr'] = lr_a_now
    for p in optimizer_critic.param_groups:
        p['lr'] = lr_c_now


"""
新增动态的归一化方法：
1、首先定义一个动态计算mean和std的class，名为RunningMeanStd。
    这个class中的方法的核心思想是已知n个数据的mean和std，如何计算n+1个数据的mean和std
2、定义一个名叫Normalization的类，其中实例化上面的RunningMeanStd，需要传入的参数shape代表当前环境的状态空间的维度。
    训练过程中，每次得到一个state，都要把这个state传到Normalization这个类中，然后更新mean和std，再返回normalization后的state。
"""
"""
在一般的强化学习训练中，`self.n` 通常不会在每个 episode 中重置为0。相反，它会在整个训练过程中持续增加，以跟踪处理的总数据点数量。

`self.n` 的目的是在动态计算均值和标准差时，用于计算移动平均值（running mean）和移动标准差（running standard deviation）。这意味着它会持续更新，以反映在所有 episode 中处理的数据点。通过保持 `self.n` 不为0，算法可以继续跟踪累积的统计信息，以更好地适应变化的数据分布。

当你在连续放缩奖励时，保持 `self.n` 不为0通常是更准确的，因为它反映了更多的样本数据，而不仅仅是当前 episode 的数据。这有助于算法更好地适应奖励分布的变化，特别是在训练过程中可能会遇到不同的任务或环境。

在训练期间，`self.n` 会持续增加，当整个训练结束时，如果需要重新开始新的训练，可以选择将 `self.n` 重置为0，以便开始新的训练阶段。
"""

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0     #记录已经处理的数据点数量，初始值为0。  todo （样本ENV一样，reward不会离谱！）只有在main运行时初始化，然后所有的episode都跑完，也不会重新清0；目的是随着episode越来越多，反应更多的样本数据，而不是一个episode的数据！！！
        self.mean = np.zeros(shape) #存储均值的数组，初始值为形状为`shape`的全零数组
        self.S = np.zeros(shape)  #存储方差的数组，初始值为形状为`shape`的全零数组: 二阶中心矩?
        self.std = np.sqrt(self.S)  #存储标准差的数组，初始值为`self.S`数组的平方根。

    """
    在 update 方法中，当你传入一个数据点 x 时，它执行以下操作：

    将 x 转换为 NumPy 数组。
    增加计数器 n。
    如果 n 等于1，将 mean 和 std 初始化为 x 的值。
    如果 n 大于1，更新 mean 为旧的均值加上新数据点 x 与旧均值之差除以 n。
    更新 S 为旧 S 加上 (x - 旧均值) * (x - 新均值)。
    更新 std 为 S 除以 n 的平方根，这将计算新的标准差。
    这个类用于计算均值和标准差，通常在强化学习中用于正则化数据，特别是在使用 PPO（Proximal Policy Optimization）等算法时，有助于处理不同范围的输入数据。
    """
    def update(self, x):
        x = np.array(x) # `x`转换为NumPy数组
        self.n += 1
        if self.n == 1:  #
            self.mean = x  # 怪不得要和x的shape相同，不然怎么赋值
            self.std = np.abs(x) # 是否是x无关紧要，因为已经分子变成0了（归一化时）todo 防止bug，std都是正数，如果x为负数，std就是负数了！相除符号都变了，改为x的绝对值
        else:
            old_mean = self.mean.copy()  #如果已处理的数据点数量为1（即第一个数据点），则直接将均值和标准差设置为输入数据`x`。
            self.mean = old_mean + (x - old_mean) / self.n  #新均值 = 旧均值 + (输入数据 - 旧均值) / 已处理的数据点数量
            self.S = self.S + (x - old_mean) * (x - self.mean)  #新方差 = 旧方差 + (输入数据 - 旧均值) * (输入数据 - 新均值)
            self.std = np.sqrt(self.S / self.n)  #新标准差 = 新方差的平方根 / 已处理的数据点数量的平方 # todo 平方根就是开平方？

"""
在Normalization的初始化中会对计算mean和std的类进行初始化，代替清0的作用
1、每个episode都需要对其重新实例化（除非你要一直记录这样子的mean。不断累加）
2、归一化：最直接的方法就是放缩，直接除以一个最大值，放缩到0和1之间
"""
class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

"""
reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
r = reward_scaling(r)   # 这里好比就是调用__call__，调用一次就累加一次

这个类的目的是在训练中动态地缩放奖励，以适应奖励值的变化，并确保算法的稳定性。这种缩放可以有助于处理不同范围的奖励值，以更好地训练强化学习模型。
"""
class RewardScaling:  # todo 需要每次的episode的时候，就进行初始化，连带RunningMeanStd初始化，相当于清0；因为其中有self_count的累加值等等在记录！！！
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1 todo 表示只有一个数值？
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x  # 折扣因子，计算折扣即时r的总和 todo 和计算折扣累加GT有什么不同？（边放缩边进行了折扣累计奖励的计算，然后拿过来用在gae的计算！！！不是哦，这里累加了只是为了更新mean和std！return的还是r本身啊！）
        self.running_ms.update(self.R)  # 算一个总和，计算总和的均值和标准差
        x = x / (self.running_ms.std + 1e-8)  # Only divided std  将奖励 x 除以 running_ms 的标准差，以进行奖励缩放。这将确保奖励在缩放后具有标准差为1。todo 不减去mean是为了防止出现0，也代表第一个的就是自身放缩成1；这是动态的mean和std！！！！
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


"""
写一个打印出adj邻居矩阵的函数
标注（列，行），对应方向
"""
import matplotlib.pyplot as plt
def plot_adj_with_index(sparse_data, spilt_id, use_split=False):
    fig = plt.figure(figsize=(32, 32))
    b = sparse_data.to_dense().cpu()
    if use_split:
        plt.imshow(b[:spilt_id,:spilt_id], cmap='gray')
    else:
        plt.imshow(b, cmap='gray')

    # 标记行列编号
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if b[i, j] == 1:
                # plt.text(j, i, f"({i}{j})", color='red', ha='center', va='center')
                plt.text(j, i, f"({j}{i})", color='red', ha='center', va='center', fontsize=8)
    plt.colorbar()
    plt.show()