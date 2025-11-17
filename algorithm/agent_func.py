
from torch.distributions.categorical import Categorical
import torch

"""
agent的选择动作：
1、传入：可选动作的概率 + 候选的节点信息
2、返回：选择action的id + 对应的task的id + 对应action的log_prob

candidate = tensor   (batch,job)
            ([[ 0,  4,  8, 12],
            [ 0,  4,  8, 12],
            [ 0,  4,  8, 12],
            [ 0,  4,  8, 12]], device='cuda:0')
prob = 4,4 = batch*job  都是可选择job的概率！

选择action的时候：
1、网络没有.train
2、用网络没有加with torch.no_grad(): 没有加，提高计算效率
3、采样之后的action_id没有.to(device) 
"""
def select_operation_action(prob, candidate):
    pi = Categorical(prob)  # 维度：4，4
    action_idx = pi.sample()  # 采样得到动作a：4 (batch)
    task_index = []
    log_a = pi.log_prob(action_idx)  # 对应动作的logprob：4 (batch)

    for i in range(action_idx.size(0)):  # 有几个batch
        a = candidate[i][action_idx[i]]  # 从候选task的节点id中选择，i=第几个样本，s[i]当前样本选择的动作！遍历a，然后从候选candidate中选择a：就是我的mask的感觉！！！！！！
        task_index.append(a)  # 按照候选选择的可用a，返回一个list，储存每一个样本的选择action
    task_index = torch.stack(task_index, 0)  # stack重新堆叠下，变成一个大张量：1维4元素
    return task_index, action_idx, log_a

"""
greedy方式：
1、只选择最大概率的action
2、用于test/eval
3、prob = candidate = （batch,job）
"""
# select action method for test
def greedy_select_action(prob, candidate): # p是网络输出的概率prob
    pi = Categorical(prob)  # 去除维度=1的维度，建立离散概率分布  = （batch,job）
    _, index = prob.max(1)  # greedy就是直接选取最大的概率！  沿着维度1（即类别维度）找到最大概率的索引  =(batch_size,)
    log_a = pi.log_prob(index)  # 返回已选action的对应的对数概率   = (batch_size,)

    taskidx = []
    for i in range(index.size(0)):   # 遍历batch！
        a = candidate[i][index[i]]  # 可选task的id，从1开始编码，这里是job的index转为task的id，
        taskidx.append(a)   
    taskidx = torch.stack(taskidx, 0)  # # (batch_size,)
    return taskidx, index, log_a   # (batch_size,)

# 选择machine的action函数
def select_machine_action(prob):  # 传入概率分布pi_mch = batch * m
    pi = Categorical(prob)  # 去除维度=1的维度，建立离散概率分布
    action_idx = pi.sample()  # 直接采样，得到动作索引：张量 tensor（batch个元素）

    #if memory is not None: log_prob.append(dist.log_prob(s).cpu().tolist())

    log_a = pi.log_prob(action_idx) # 对应采样索引的log_prob  tensor（batch个元素）

    return action_idx, log_a  # tensor（batch个元素

# 选择machine的action函数: 只选择最大概率的动作
def greedy_select_machine_action(prob):  # 传入概率分布pi_mch = batch * m
    pi = Categorical(prob)  # 去除维度=1的维度，建立离散概率分布
    action_idx = torch.argmax(prob)  # 直接采样，得到动作索引：张量 tensor（batch个元素）

    #if memory is not None: log_prob.append(dist.log_prob(s).cpu().tolist())

    log_a = pi.log_prob(action_idx) # 对应采样索引的log_prob  tensor（batch个元素）

    return action_idx, log_a  # tensor（batch个元素