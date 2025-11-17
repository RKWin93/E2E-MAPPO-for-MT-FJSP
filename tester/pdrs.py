import copy
import gym
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
import numpy as np
import random
from random import shuffle
import pandas as pd
import torch.nn.utils as nn_utils
import torch.nn.functional as F

from trainer.fig_kpi import result_box_plot
from graph_jsp_env.disjunctive_graph_jsp_env_singlestep import DisjunctiveGraphJspEnv_singleStep
from instance.generate_allsize_mofjsp_dataset import Instance_Dataset



def PDRsSETTINGError(param):
    pass


class FJSP_Rules(object):
    def __init__(self, n_job, n_machine):
        self.n_j = n_job
        self.n_m = n_machine
        self.total_task = self.n_j * self.n_m

    # """
    # 1、输入到Env里边的是reshape之后的m_order
    # """
    # def chose_rule_m(self):
    #     pass
    #
    # def chose_rule_e(self):
    #     pass


    """
    定义选择m的rules: return action
    1、SPT_m: shortest processing time 选择最短的加工时间t  （常见的）
    2、SEC_m: shortest energy consumption 最小的能耗：t*e1
    3、machine in same edge 只选同一个边中的m（有随机） e2最小
    4、ratio=0：all machine using 所有设备都选上，即也是不同边（有随机）
    """
    def SPT_m(self, t_ability):  # 16task * 4 machine
        t_instance = copy.deepcopy(t_ability)  # task*m的加工时间能力矩阵
        # TODO：对应Minus版本的数据，防止最小值找到负数；np.min找到最小值，np.argmin找到最小值的索引
        t_instance[t_instance < 0] = float("inf")  # 将小于等于0的元素替换为无穷大
        # indices_of_min = np.argmin(t_ability, axis=1)  # argmin必须用二维矩阵 + 输出每行的最小值索引: 所有子任务的最短加工时间的m索引
        indices_of_min = np.argmin(t_instance, axis=1)  # 沿着axis=1的方向找到每一行的最小值, 返回的是一维数组啊！总共task个元素+输出每行的最小值索引: 所有子任务的最短加工时间的m索引
        return indices_of_min  # 返回的是选好的所有m, 从task1开始的，m从0开始

    # def SEC_m(self, e1_ability): # 16task * 4 machine
    def SEC_m(self, t_ability, p_ability): # 16task * 4 machine

        # EC = np.multiply(t_ability, e1_ability) # 对应元素相乘
        # indices_of_min = np.argmin(EC, axis=1)  # 输出每行的最小值索引  先暂时不乘以t了！！！！！！！！！！！
        # indices_of_min = np.argmin(e1_ability, axis=1)  # 输出每行的最小值索引  先暂时不乘以t了！！！！！！！！！！！
        # 计算元素的绝对值一一对应相乘得到矩阵 c
        e1_ability = np.multiply(t_ability, np.abs(p_ability)) # 保留负号
        pt_instance = copy.deepcopy(e1_ability)  # task*m的加工能耗能力矩阵
        # TODO：对应Minus版本的数据，防止最小值找到负数; 不对，t和p都有一一对应负数，能耗就没有负数的！
        pt_instance[pt_instance < 0] = float("inf")  # 将小于等于0的元素替换为无穷大
        indices_of_min = np.argmin(pt_instance, axis=1) # 输出每行的最小值索引
        return indices_of_min  # 返回的是选好的所有m

    """
    选择m的时候可以遍历所有的m，然后查看谁的新增的运输和空闲时间少，选谁，是可以选，但是不是常用规则；先不写，比较麻烦，写是可以写的，但不是我的关键啊！
    最左侧的一列task，运输和空闲都要随机选m，因为都是0；运输后续可以找min的；但是空闲遍历所有，可能好多都不新增的m，还要随机！！！
    还有一个原因：上述的随机选取出现的比较多，有的可能新增一样，那不如直接写一个随机的方法！！！
    
    统一一下：从task1开始遍历选择m！！！
    """
    def Random_m(self, ability_t):
        indices_of_min = []
        for task_index in range(self.total_task):
            positive_indices = np.where(ability_t[task_index] > 0)[0]
            # print(positive_indices)
            if len(positive_indices) > 0:  # 存在大于0的索引，一般都存在，除非我的生成样本代码写错了！
                random_index = random.choice(positive_indices)
                indices_of_min.append(random_index)
                # print(random_index)
            else:
                print("No positive elements found in the row.")
        return indices_of_min


    def MISE_m(self, edge_info):  # [[0, 1], [2, 3]]
        rand_edge = edge_info[random.randint(0,len(edge_info)-1)] # [a,b]=[0,1],随机选一个edge，只用该边里边的m
        machines = []
        for i in range(self.total_task):
            random_num = random.choice(rand_edge)  # 从列表中随机选择一个元素
            machines.append(random_num)  # 将选择的元素添加到结果列表中
        return machines # 返回的事所有m的id（task个）

    """默认就是每个job的operation=machine"""
    def AMU_m(self):
        m_list = [i for i in range(self.n_m)] # 生成m的列表[0,1,2,3]
        machines = []
        for i in range(self.n_j): # 每个job都用了所有的m，但是是随机的排序
            random.shuffle(m_list)  # 随机排列列表
            machines.extend(m_list)  # 将排列后的列表展开并添加到结果列表中
        return machines # 返回的事所有m的id（task个）

    """
    定义选择o的rules: return action = task_id - 1 列表
    1、FIFO: first in first out 先进先出
    2、MOR：most operation number remaining 最大剩余operation的job (随机) or 一列一列选，不会随机，都是从小到大？
    3、LWKR_T：least work remaining-processing time 余下加工时间最短的job
    4、LWKR_PT：least work remaining-energy consumption 余下加工能耗最少的job
    5、LWKR_IT：least work remaining-idle time 余下新增空闲时间最少的job (麻烦，写了，跑起来估计会很慢) 
    6、LWKR_TT：least work remaining-trans time 余下运输时间最少的job  
    
    镜像：
    3、MWKR_T：Most work remaining-processing time 余下加工时间最短的job
    4、MWKR_PT：Most work remaining-energy consumption 余下加工能耗最少的job
    5、MWKR_IT：Most work remaining-idle time 余下新增空闲时间最少的job (麻烦，写了，跑起来估计会很慢) 
    6、MWKR_TT：Most work remaining-trans time 余下运输时间最少的job  
    
    1028-不再是只看candidate，而应该是余下的所有同job的task的总和，剩下min/max！！！！
    """
    def FIFO_o(self):
        operations = [i+1 for i in range(self.total_task)] # 生成task的列表[1,2,3,...,n]
        return operations # 返回task_id=1 开始的list

    # TODO: 因为所有job的子任务数量相同，所以这个类似于随机选。哪个job剩余的task数量多，选哪个job的工序
    def MOR_o(self): # 每一列选完了，才能开始下一列（从左到右）
        task_list = [i+1 for i in range(self.total_task)]  # 生成task的列表[1,2,3,...,16]
        task_list = np.array(task_list).reshape(self.n_j, self.n_m) # 转成4*4的12.。。16矩阵
        # [[1,5,9,13],[2,6,10,14],......]
        result_list = [list(col) for col in zip(*task_list)] # 列表转置后按列取出，存成列表。col指的是原task_list的列，提取出来
        operations = []
        for i in range(self.n_m):  # 每次都是同一列选完，operation的个数
            random.shuffle(result_list[i])  # 随机排列列表 TODO 1205-这边的MOR就是按照剩余子任务数最多的顺序，但是面对一样的数量，不会随机选了，就是按照顺序走的！（还是打乱吧）
            operations.extend(result_list[i])  # 将排列后的列表展开并添加到结果列表中
        return operations # 返回task_id的list

    def Random_task(self):
        task_list = [i+1 for i in range(self.total_task)]  # 生成task的列表[1,2,3,...,16]
        task_list = np.array(task_list).reshape(self.n_j, self.n_m)  # 转成4*4的12.。。16矩阵
        candidate = task_list[:,0].tolist() # 第一列：1,5，9,13  转成列表才能删除元素
        operation_list = []
        for i in range(self.total_task): # 循环task次数
            select_task = random.choice(candidate)  # 从都是candidate中随便选一个，即为所选择的job的index
            operation_list.append(select_task)  # 记录所选task_id

            # 找到指定元素的索引
            index = candidate.index(select_task)
            # 将指定元素加1
            candidate[index] += 1

            if (candidate[index]-1) % self.n_m == 0: # 说明此时的某一行已经选完，
                # 从矩阵 a 中删除已经选取过的元素
                del candidate[index]
        return operation_list

    """
    剩余task的总和最小值（某一指标的总和最小）：
    1、我还没有分配m，怎么知道每个task的指标大小呢？（关注的是least最小，指标就用最小，或者mean！）
    """
    def LWKR_T_o_jointActor(self, ability_t, data_type, rule_type):
        # 按照task1234.。16的顺序找到每一个task的minT
        t = copy.deepcopy(ability_t) # task*m
        if data_type == "min":
            t[t<0] = float("inf")  # 小于0的都设置成无穷大
            t_selected = np.min(t, axis=1) # 找到每一行中的最小值！一维数组形式
        elif data_type == "mean":
            t_mean = []
            for row in t:
                positive_elements = [element for element in row if element > 0] # list
                if positive_elements:  # 防止某一个task都是负数，不太可能出现！！
                    mean = sum(positive_elements) / len(positive_elements)
                    t_mean.append(mean)
                else:
                    t_mean.append(0)
            t_selected = np.array(t_mean)
        # TODO 有负数找max没有影响
        elif data_type == "max":
            # t[t < 0] = float("-inf")  # 小于0的都设置成负无穷大
            t_selected = np.max(t, axis=1)  # 找到每一行中的最大值！（负数不影响）list形式

        t_new = t_selected.reshape(self.n_j, self.n_m)  # 整形成二维矩阵
        refer_value = np.sum(t_new, axis=1)  # 求得每一行的sum，作为参考值，返回一维数组

        """按照每一个job剩余task的总和的大小，选择job"""
        job_index_lst = []
        # task_index = [0] * self.n_m  # 作为每一个job的task的index
        task_index = [0] * self.n_j  # 作为每一个job的task的index TODO 1218-这边代表当前job的可选的task的index，数量应该有j个才对！
        for i in range(self.total_task):
            # TODO 如果选的是剩余task的总和最小，越选那么这个job的剩余的task的总和越小，就是选完这个job再开始其他的！！！！
            if rule_type == "least":
                job_index = np.argmin(refer_value)  # 所选job的index，找到min的索引，代表此job的剩余task的总t比较小
            elif rule_type == "most":
                job_index = np.argmax(refer_value)  # 所选job的index,找到max的索引

            job_index_lst.append(job_index)  # 记录索引值

            # print("--------------",refer_value, job_index,t_new,task_index)
            # print("--------------",refer_value[job_index])
            # print("--------------", t_new[job_index][task_index[job_index]])
            refer_value[job_index] -= t_new[job_index][task_index[job_index]]  # 对应的判断指标值 - 已经选择的task的t
            task_index[job_index] += 1 # 更新下一次可选的task的index（对应job中的）
            if refer_value[job_index] == 0 or task_index[job_index] > self.n_m-1: # 表明这个job选完了
                if rule_type == "least":
                    refer_value[job_index] = float("inf")  # （找min）不能选的设为最大
                elif rule_type == "most":
                    refer_value[job_index] = float("-inf") # （找max）不能选的设为最小

        """转换job_index变为task的id=1开始"""
        # print(c) # 按照余下最小选择的每次的job先后顺序
        task_list = [i + 1 for i in range(self.total_task)]  # 生成task的列表[1,2,3,...,16]
        task_list = np.array(task_list).reshape(self.n_j, self.n_m)  # 转成4*4的1，2，。。16矩阵
        task_list = task_list.tolist()  # 转成列表才能删除元素
        # 初始化结果列表 result
        operations = []
        for i in job_index_lst:
            # 选取相应行中的第一个元素
            chosen_val = task_list[i][0]
            operations.append(chosen_val)
            # 从矩阵 a 中删除已经选取过的元素
            del task_list[i][0]
        # print(operations)
        return operations  # 返回的是选好的task_id的顺序

    def LWKR_PT_o_jointActor(self, ability_t, ability_p, data_type, rule_type):
        # 按照task1234.。16的顺序找到每一个task的minPT
        e1_ability = np.multiply(ability_t, np.abs(ability_p))  # 保留负号
        pt = copy.deepcopy(e1_ability)  # task*m的加工时间能力矩阵 # task*m
        if data_type == "min":
            pt[pt<0] = float("inf")  # 小于0的都设置成无穷大
            t_selected = np.min(pt, axis=1) # 找到每一行中的最小值！一维数组形式
        elif data_type == "mean":
            t_mean = []
            for row in pt:
                positive_elements = [element for element in row if element > 0] # list
                if positive_elements:  # 防止某一个task都是负数，不太可能出现！！
                    mean = sum(positive_elements) / len(positive_elements)
                    t_mean.append(mean)
                else:
                    t_mean.append(0)
            t_selected = np.array(t_mean)
        # TODO 有负数找max没有影响
        elif data_type == "max":
            # t[t < 0] = float("-inf")  # 小于0的都设置成负无穷大
            t_selected = np.max(pt, axis=1)  # 找到每一行中的最大值！（负数不影响）list形式

        t_new = t_selected.reshape(self.n_j, self.n_m)  # 整形成二维矩阵
        refer_value = np.sum(t_new, axis=1)  # 求得每一行的sum，作为参考值，返回一维数组

        """按照每一个job剩余task的总和的大小，选择job"""
        job_index_lst = []
        # task_index = [0] * self.n_m  # 作为每一个job的task的index
        task_index = [0] * self.n_j  # 作为每一个job的task的index TODO 1218-这边代表当前job的可选的task的index，数量应该有j个才对！
        for i in range(self.total_task):
            # TODO 如果选的是剩余task的总和最小，越选那么这个job的剩余的task的总和越小，就是选完这个job再开始其他的！！！！
            if rule_type == "least":
                job_index = np.argmin(refer_value)  # 所选job的index，找到min的索引，代表此job的剩余task的总t比较小
            elif rule_type == "most":
                job_index = np.argmax(refer_value)  # 所选job的index,找到max的索引

            job_index_lst.append(job_index)  # 记录索引值

            refer_value[job_index] -= t_new[job_index][task_index[job_index]]  # 对应的判断指标值 - 已经选择的task的t
            task_index[job_index] += 1 # 更新下一次可选的task的index（对应job中的）
            if refer_value[job_index] == 0 or task_index[job_index] > self.n_m-1: # 表明这个job选完了
                if rule_type == "least":
                    refer_value[job_index] = float("inf")  # （找min）不能选的设为最大
                elif rule_type == "most":
                    refer_value[job_index] = float("-inf") # （找max）不能选的设为最小

        """转换job_index变为task的id=1开始"""
        # print(c) # 按照余下最小选择的每次的job先后顺序
        task_list = [i + 1 for i in range(self.total_task)]  # 生成task的列表[1,2,3,...,16]
        task_list = np.array(task_list).reshape(self.n_j, self.n_m)  # 转成4*4的1，2，。。16矩阵
        task_list = task_list.tolist()  # 转成列表才能删除元素
        # 初始化结果列表 result
        operations = []
        for i in job_index_lst:
            # 选取相应行中的第一个元素
            chosen_val = task_list[i][0]
            operations.append(chosen_val)
            # 从矩阵 a 中删除已经选取过的元素
            del task_list[i][0]
        # print(operations)
        return operations  # 返回的是选好的task_id的顺序

    # TODO 空闲和运输时间是需要调度之后才知道，选择task时不知道；可以尝试放在选m？

    def LWKR_T_o(self, jsp_instance, type): # 需要返回已经选好的m的对应的t

        """
        :param jsp_instance: np.array = 2*j*m：选择的m_id + m对应的t
        :return:
        """
        """
        先按照candidate中谁t小，记录job的选择index
        row_mins = [(5, 1), (1, 3), (1, 4), (2, 6)] 元组列表
        min_row = min(row_mins)：返回的是（1,3）上述元组中第一个元素最小的元组，有相同数值的只返回第一个！！
            现在修改：防止出现bug，出现相同最小值，因此从其中随机选一个（只有1个元组也能运行的）
            `key=lambda x: x[0]`指定了比较的键值为元组的第一个元素。
        """
        # 初始化结果列表 c
        acton = []
        jsp_t_chose = jsp_instance[1].tolist() # 转成列表才能删除元素！
        for i in range(self.total_task):
            # 记录jsp_t_chose每行第一个元素的值和所在的行数
            row_mins = [(row[0], idx) for idx, row in enumerate(jsp_t_chose) if len(row) > 0]  # out：[(5, 1), (1, 3), (1, 4), (2, 6)]
            # 如果所有行都是空的，就退出循环：全删完了
            if not row_mins:
                break
            # 选出最小值所在的行并记录其行数
            if type == "min":
                min_first = min(row_mins, key=lambda x: x[0])  # 找到第一个数最小的元组, 只会找到其中首位最小的第一个元组 = (1, 3)
            elif type == "max":
                min_first = max(row_mins, key=lambda x: x[0])  # 找到第一个数最小的元组, 只会找到其中首位最小的第一个元组 = (1, 3)

            tuples = [t for t in row_mins if t[0] == min_first[0]]  # 使用列表推导式筛选出第一个数与最小值相等的元组 = [(1, 3), (1, 4)]
            index_temp = []
            for k in tuples:
                index_temp.append(k[-1])  # 找出都是最小首位的元组的对应的末位的index
            min_row = random.choice(index_temp)  # 从都是min的索引中随便选一个，即为所选择的job的index
            # min_row = min(row_mins)[1] # 反馈的是行数，BUG：相同数值只返回第一个最小值的index！！！！！

            acton.append(min_row)   # 记录的是选择job的index
            # 从所选行中删除第一个元素，删完该行，变成[],依旧存在
            del jsp_t_chose[min_row][0]  # 类似于我的每次更新candidate列表，第一行的第一个元素被选，下一次可选该行的第二个元素，要来和其他的比较的！
            # jsp_instance[1][min_row] = np.delete(jsp_instance[1][min_row], 0)

        """
        将job的index序号转成task_id的顺序
        """
        # print(c) # 按照余下最小选择的每次的job先后顺序
        task_list = [i + 1 for i in range(self.total_task)]  # 生成task的列表[1,2,3,...,16]
        task_list = np.array(task_list).reshape(self.n_j, self.n_m)  # 转成4*4的1，2，。。16矩阵
        task_list = task_list.tolist() # 转成列表才能删除元素
        # 初始化结果列表 result
        operations = []
        for i in acton:
            # 选取相应行中的第一个元素
            chosen_val = task_list[i][0]
            operations.append(chosen_val)
            # 从矩阵 a 中删除已经选取过的元素
            del task_list[i][0]
        # print(operations)
        return operations  # 返回的是选好的task_id的顺序

    """
    同理上述的最小t选择法，选择最小p*t
    """
    def LWKR_PT_o(self, jsp_e1, type): # 被选之后的p*t的j*m矩阵

        # EC = np.multiply(jsp_instance[1], jsp_e1)  # 对应元素相乘
        EC = jsp_e1  # 对应元素相乘  先暂时不乘以t了！！！！！！！！！！！

        # 初始化结果列表 c
        acton = []
        EC_lst = EC.tolist()  # 转成列表才能删除元素
        for i in range(self.total_task):
            # 记录每行第一个元素的值和所在的行数
            row_mins = [(row[0], idx) for idx, row in enumerate(EC_lst) if len(row) > 0]
            # 如果所有行都是空的，就退出循环
            if not row_mins:
                break
            # 选出最小值所在的行并记录其行数
            # `key=lambda x: x[0]`指定了比较的键值为元组的第一个元素。
            if type == "min":
                min_first = min(row_mins, key=lambda x: x[0])  # 找到第一个数最小的元组, 只会找到其中首位最小的第一个元组 = (1, 3)
            elif type == "max":
                min_first = max(row_mins, key=lambda x: x[0])  # 找到第一个数最小的元组, 只会找到其中首位最小的第一个元组 = (1, 3)
            tuples = [t for t in row_mins if t[0] == min_first[0]]  # 使用列表推导式筛选出第一个数与最小值相等的元组 = [(1, 3), (1, 4)]
            index_temp = []
            for k in tuples:
                index_temp.append(k[-1])  # 找出都是最小首位的元组的对应的末位的index
            min_row = random.choice(index_temp)  # 从都是min的索引中随便选一个，即为所选择的job的index
            # min_row = min(row_mins)[1]   #反馈的是行数，相同数值只返回第一个最小值的index！！！！！

            acton.append(min_row)
            # 从所选行中删除第一个元素
            del EC_lst[min_row][0]

        # print(c) # 按照余下最小选择的每次的job先后顺序
        task_list = [i + 1 for i in range(self.total_task)]  # 生成task的列表[1,2,3,...,16]
        task_list = np.array(task_list).reshape(self.n_j, self.n_m)  # 转成4*4的1，2，。。16矩阵
        task_list = task_list.tolist()  # 转成列表才能删除元素
        # 初始化结果列表 result
        operations = []
        for i in acton:
            # 选取相应行中的第一个元素
            chosen_val = task_list[i][0]
            operations.append(chosen_val)
            # 从矩阵 a 中删除已经选取过的元素
            del task_list[i][0]
        # print(operations)
        return operations  # 返回的是选好的task_id的顺序

    def LWKR_IT_o(self, env, type): # 选择剩下action可选中的，idle_t最小的子任务

        pool_task_list = [1 + self.n_m * i for i in range(self.n_j)] # candidate可选task_id：[1,5.9.13]
        pool_task_dict = {}  # 用作更新的任务池，可选candidate的dict
        for i in range(self.n_j):
            pool_task_dict[i] = pool_task_list[i]    # 每个job对应的可选的task的id
        chosen_action_list = []  # 最终的动作的序列列表
        pool_task_list_init = pool_task_list  # 用作判断是否超出索引
        index_list = [i for i in range(self.n_j)] # 可以选择的job，即行数

        """
        关键是我怎么判断it的大小？
        1、step返回的idle = prev - current，因为是累加，所以代表选择了当前task造成的it更新
        2、it一般都是负数，因为是不断累加的！所以，找的是最大值！
        """
        # 随机选取的动作
        for i in range(self.total_task):

            # if index_list:  # 判断index，list没有被删完
            #     index_chose = random.choice(index_list)  # 从job中随便选一个，3行，对应012！！！！！！！！！随机选取！

            it_temp = []

            """
            1、每一次判断当前的candidate的时候，前边所选的action都要跑一边。保证是在同一个基础上开始的
            2、所以，每次判断当前candidate的时候，需要reset
                之前reset里边的 self.selected_action 和 self.it_s没有初始化，直接读初始化后的nodes节点，开始、结束时间是None，报错！ 
            """
            for index in index_list: # 剩下有啥子job可选
                for step in range(len(chosen_action_list)):  # 现今选了几个动作了，env走完那几步
                    _, _, _, _, _, r_idle, *_ = env.step((chosen_action_list[step] - 1))  # action = task_id - 1， DGenv的step需要传入task的index = id -1
                    # print("sssssssssssssss  step:", step, chosen_action_list)  # 0 1
                # print("sssssssssssssss:", index_list, index, pool_task_dict[index])  # 0 1
                _, _, _, _, _, r_idle, *_ = env.step((pool_task_dict[index] - 1))
                it_temp.append(r_idle)

                env.reset()  # 重置，在看下一个可选action的返回值（判断当前action的it，需要重复走完前置的动作！好麻烦啊！）
            if type == "min":
                max_val = max(it_temp)
            elif type == "max":  # 返回的 deltaIT = prev - curr 负数，所以大小取反
                max_val = min(it_temp)
            indices = [j for j, x in enumerate(it_temp) if x == max_val] # 返回都是最大值的索引

            index_temp = []
            for k in indices:
                index_temp.append(index_list[k])  # 找出max对应的具体可以选择的index
            index_chose = random.choice(index_temp)  # 从都是max的索引中随便选一个，即为所选择的job的index

            chosen_action_list.append(pool_task_dict[index_chose])  # 按照随机产生的index，从pool里边选择动作，添加到chosen_action_list
            pool_task_dict[index_chose] = pool_task_dict[index_chose] + 1  # 按照从左往右，一次+1；更新可供选择的子任务（子任务有先后，不能直接选后续的子任务）  不断累加，导致最后一次选完，task_id会超限！！！（下文解决）


            if (index_chose + 1) < self.n_j:  # 判断选的是否是最后一行？
                if pool_task_dict[index_chose] >= pool_task_list_init[index_chose + 1]:  # 说明选到下一行了，不行
                    pool_task_dict[index_chose] = 0  # 如果当前job的子任务选完了，就置0
                    index_list.remove(index_chose)  # 从可选的index中去掉当前job
                    # if_print(0, "pool_task_dict update:", pool_task_dict)
                else:
                    pass
            else:  # 最后一行，超出索引
                if pool_task_dict[index_chose] > self.total_task:  # index_chose只能是最后一行，超过矩阵索引
                    pool_task_dict[index_chose] = 0
                    index_list.remove(index_chose)  # 可选的job的个数变少，因为其他的选完了！
                    # if_print(0, "pool_task_dict update:", pool_task_dict)
                else:
                    pass
        return chosen_action_list # 返回的是选好的task_id的顺序

    def LWKR_IT_o_jointActor(self, env, type, m_list): # 选择剩下action可选中的，idle_t最小的子任务

        # m_list = m_list.tolist()
        m_list = m_list
        pool_task_list = [1 + self.n_m * i for i in range(self.n_j)] # candidate可选task_id：[1,5.9.13]
        pool_task_dict = {}  # 用作更新的任务池，可选candidate的dict
        for i in range(self.n_j):
            pool_task_dict[i] = pool_task_list[i]    # 每个job对应的可选的task的id
        chosen_action_list = []  # 最终的动作的序列列表
        pool_task_list_init = pool_task_list  # 用作判断是否超出索引
        index_list = [i for i in range(self.n_j)] # 可以选择的job，即行数

        """
        关键是我怎么判断it的大小？
        1、step返回的idle = prev - current，因为是累加，所以代表选择了当前task造成的it更新
        2、it一般都是负数，因为是不断累加的！所以，找的是最大值！
        """
        # 随机选取的动作
        for i in range(self.total_task):

            # if index_list:  # 判断index，list没有被删完
            #     index_chose = random.choice(index_list)  # 从job中随便选一个，3行，对应012！！！！！！！！！随机选取！

            it_temp = []

            """
            1、每一次判断当前的candidate的时候，前边所选的action都要跑一边。保证是在同一个基础上开始的
            2、所以，每次判断当前candidate的时候，需要reset
                之前reset里边的 self.selected_action 和 self.it_s没有初始化，直接读初始化后的nodes节点，开始、结束时间是None，报错！ 
            """
            for index in index_list: # 剩下有啥子job可选
                for step in range(len(chosen_action_list)):  # 现今选了几个动作了，env走完那几步
                    """
                    chosen_action_list里边存的是已经选择的task_id
                    """
                    task_index = chosen_action_list[step] - 1  # 找到对应的task_index
                    joint_actor = [task_index, m_list[task_index]] # 传入=[task_index. m_id]
                    _, _, _, _, _, r_idle, *_ = env.step(joint_actor)  # action = task_id - 1， DGenv的step需要传入task的index = id -1
                    # print("sssssssssssssss  step:", step, chosen_action_list)  # 0 1
                # print("sssssssssssssss:", index_list, index, pool_task_dict[index])  # 0 1
                task_index1 = pool_task_dict[index] - 1  # 找到对应的task_index
                joint_actor1 = [task_index1, m_list[task_index1]] # 传入=[task_index. m_id]
                _, _, _, _, _, r_idle, *_ = env.step(joint_actor1)
                it_temp.append(r_idle)

                env.reset()  # 重置，在看下一个可选action的返回值（判断当前action的it，需要重复走完前置的动作！好麻烦啊！）
            if type == "min":
                max_val = max(it_temp)
            elif type == "max":  # 返回的 deltaIT = prev - curr 负数，所以大小取反
                max_val = min(it_temp)
            indices = [j for j, x in enumerate(it_temp) if x == max_val] # 返回都是最大值的索引

            index_temp = []
            for k in indices:
                index_temp.append(index_list[k])  # 找出max对应的具体可以选择的index
            index_chose = random.choice(index_temp)  # 从都是max的索引中随便选一个，即为所选择的job的index

            chosen_action_list.append(pool_task_dict[index_chose])  # 按照随机产生的index，从pool里边选择动作，添加到chosen_action_list
            pool_task_dict[index_chose] = pool_task_dict[index_chose] + 1  # 按照从左往右，一次+1；更新可供选择的子任务（子任务有先后，不能直接选后续的子任务）  不断累加，导致最后一次选完，task_id会超限！！！（下文解决）


            if (index_chose + 1) < self.n_j:  # 判断选的是否是最后一行？
                if pool_task_dict[index_chose] >= pool_task_list_init[index_chose + 1]:  # 说明选到下一行了，不行
                    pool_task_dict[index_chose] = 0  # 如果当前job的子任务选完了，就置0
                    index_list.remove(index_chose)  # 从可选的index中去掉当前job
                    # if_print(0, "pool_task_dict update:", pool_task_dict)
                else:
                    pass
            else:  # 最后一行，超出索引
                if pool_task_dict[index_chose] > self.total_task:  # index_chose只能是最后一行，超过矩阵索引
                    pool_task_dict[index_chose] = 0
                    index_list.remove(index_chose)  # 可选的job的个数变少，因为其他的选完了！
                    # if_print(0, "pool_task_dict update:", pool_task_dict)
                else:
                    pass
        return chosen_action_list # 返回的是选好的task_id的顺序


    def LWKR_TT_o(self, ability_prev_transT, type):  # 传入j*m的所需运输时间的矩阵，有运输的存在prev中

        """
        :param ability_prev_transT: np.array = j*m：传入j*m的所需运输时间的矩阵，有运输的存在prev中
        :return:
        """
        """
        先按照candidate中谁需要运到下一个m的tranT小，记录job的选择index
        """
        # 初始化结果列表 c
        action = []
        transT_chose = ability_prev_transT.tolist()  # 转成列表才能删除元素！
        for i in range(self.total_task):
            # 记录transT_chose每行第一个元素的值和所在的行数
            row_mins = [(row[0], idx) for idx, row in enumerate(transT_chose) if len(row) > 0]
            # 如果所有行都是空的，就退出循环：全删完了
            if not row_mins:
                break
            """
            # 选出最小值所在的行并记录其行数
            # transT不像是t和p一般不会出现完全相同的数值，运输好多都是=0，那么就要有随机选择
            # `key=lambda x: x[0]`指定了比较的键值为元组的第一个元素。
            """
            if type == "min":
                min_first = min(row_mins, key=lambda x: x[0])  # 找到第一个数最小的元组, 只会找到其中首位最小的第一个元组 = (1, 3)
            elif type == "max":
                min_first = max(row_mins, key=lambda x: x[0])  # 找到第一个数最小的元组, 只会找到其中首位最小的第一个元组 = (1, 3)

            tuples = [t for t in row_mins if t[0] == min_first[0]]  # 使用列表推导式筛选出第一个数与最小值相等的元组 = [(1, 3), (1, 4)]
            index_temp = []
            for k in tuples:
                index_temp.append(k[-1])  # 找出都是最小首位的元组的对应的末位的index
            min_row = random.choice(index_temp)  # 从都是min的索引中随便选一个，即为所选择的job的index

            action.append(min_row)  # 记录的是选择job的index
            # 从所选行中删除第一个元素，删完该行，变成[],依旧存在
            del transT_chose[min_row][0]  # 类似于我的每次更新candidate列表，第一行的第一个元素被选，下一次可选该行的第二个元素，要来和其他的比较的！
            # transT_chose[1][min_row] = np.delete(transT_chose[1][min_row], 0)

        """
        将job的index序号转成task_id的顺序
        """
        # print(c) # 按照余下最小选择的每次的job先后顺序
        task_list = [i + 1 for i in range(self.total_task)]  # 生成task的列表[1,2,3,...,16]
        task_list = np.array(task_list).reshape(self.n_j, self.n_m)  # 转成 4 * 4 的1，2，。。16矩阵
        task_list = task_list.tolist()  # 转成列表才能删除元素
        # 初始化结果列表 result
        operations = []
        for i in action:
            # 选取相应行中的第一个元素
            chosen_val = task_list[i][0]
            operations.append(chosen_val)
            # 从矩阵 a 中删除已经选取过的元素
            del task_list[i][0]
        # print(operations)
        return operations  # 返回的是选好的task_id的顺序




"""
此为循环一次，Rules所产生的cost
1、按照最新的JointAction的思路，先选择task，然后再分配machine（尽量去掉我之前写的machine_ENV）
2、注意现在的t和p的能力值是有负数的，代表不能被选择

注意：原先的it和transT都必须是选好m之后，每一次都是遍历所有candidate查看选择之后新增的it或transT，进行比较之后确定task的选择顺序、
（现在没有预先选好的m + m能力有限制 + 不再是只看candidate，而应该是余下的所有同job的task！！！！）
"""
def run_Rules_jointActions_withMinus_1217(args, o_rule, m_rule, data, data_index, data_type, MLWKR_type):
    """

    :param args:   传入的配置参数
    :param m_rule:   选择采用哪个选m策略
    :param o_rule:   选择采用哪个选o策略
    :param data:     eval采用的数据样本（暂定100个seed=1）
    :param data_index:  单步运行，此时使用data的index
    :param data_type:   处理LWKR的时候采用的是min，max，mean的数据处理方式： least对应min和mean，most对应max和mean
    :param MLWKR_type： 选择least还是most的L/MWKR规则算法
    :return:
    """

    n_job = args['n_job']  # 传进来的是scene场景大小，j和m
    n_machine = args['n_machine']
    n_total_task = n_job * n_machine
    reward_dict = args['reward_scaling']  # direction选择中的reward放缩比例,TODO-1217-已经不放缩了，有动态的放缩

    # 记录当前次数的cost以及一些Gt的返回值
    cost = {
        "mch_Gt": None,
        "mch_pt": None,
        "mch_transT": None,
        "mch_useRatio": None,
        "opr_Gt": None,
        "opr_mk": None,
        "opr_idleT": None,
        "opr_pt": None,
        "opr_transT": None
    }

    """
    加载eval的数据；
    1、第一维度是env_samples = 100 （暂时）
    2、t and p: shape = (samples, total_task, m)
    3、transT: shape = samples * m * m
    4、edge: shape = samples * edge_num * m/edge_num(均分)
    """
    sample_t = data.t[data_index]
    sample_p = data.p[data_index]
    sample_transT = data.transT[data_index]
    sample_edge_info = data.edge[data_index]

    """
    不同的m的规则选择
    """
    myrules = FJSP_Rules(n_job, n_machine)  # 实例化Rules类

    # 调度环境初始化
    # m选择不同了，所以每次都要初始化
    """初始化DGFJSPEnv，反馈state和训练用reward"""
    jsp_instance = np.array([sample_t, sample_p])  # (记录能力矩阵：加工时间t和加工能耗p)
    DGenv = DisjunctiveGraphJspEnv_singleStep(jps_instance=jsp_instance,
                                              reward_function_parameters=reward_dict,
                                              # makespan of the optimal solution for this instance
                                              default_visualisations=["gantt_console", "graph_console"],
                                              reward_function='wrk', 
                                              ability_tr_mm=sample_transT,  # 运输能力矩阵
                                              perform_left_shift_if_possible=False,  # !TODO PDRs的时候：关闭左移的机制！！！！！！！！！！！！
                                              # TODO -1109-我现在觉得，PDRs就乖乖的自己的方法，本来就没有左移插空这一说
                                              # perform_left_shift_if_possible=True  # PDRs的时候：关闭左移的机制！！！！！！！！！！！！
                                              configs=args
                                              )

    DGenv.reset(Random_weight_type="eval")  # TODO 1108-验证的时候，采用的不是随机权重，而是固定的权重,PDRs也一样，这样子才有权重的初始化！

    """
    确定选择的task_id的顺序(从1开始的)
    """
    if o_rule == 0:
        operation_lst = myrules.FIFO_o()  # 先进先出 [1234...]
    elif o_rule == 1:
        operation_lst = myrules.MOR_o()  # 最大剩余子任务数量，按列
    # elif o_rule == 2:
    #     operation_lst = myrules.Random_task()  # 完全随机选取的task！

    # TODO 1109-以下方式在选择task的逻辑顺序上边并没有本质区别，无非是一些判断条件不同，带来的结果大差不差，可以去掉（1217-重新启用！）
    elif o_rule == 2:
        # operation_lst = myrules.LWKR_T_o_jointActor(sample_t, "min", rule_type="least")  # 余下加工时间最小
        operation_lst = myrules.LWKR_T_o_jointActor(sample_t, "mean", rule_type="least")  # 余下加工时间最小
    elif o_rule == 3:
        # operation_lst = myrules.LWKR_PT_o_jointActor(sample_t, sample_p, "min", rule_type="least")  # 余下能耗e1最小
        operation_lst = myrules.LWKR_PT_o_jointActor(sample_t, sample_p, "mean", rule_type="least")  # 余下能耗e1最小
    elif o_rule == 4:
        # operation_lst = myrules.LWKR_IT_o(DGenv, MLWKR_type)  # 余下新增空闲时间最少
        # operation_lst = myrules.LWKR_IT_o_jointActor(DGenv, MLWKR_type, machine_lst)  # 余下新增空闲时间最少
        # operation_lst = myrules.LWKR_T_o_jointActor(sample_t, "max", rule_type="most")  # 余下加工时间最大
        operation_lst = myrules.LWKR_T_o_jointActor(sample_t, "mean", rule_type="most")  # 余下加工时间最大
    elif o_rule == 5:
        # operation_lst = myrules.LWKR_TT_o(jsp_need_transT, MLWKR_type)  # 余下当前task需要运输的transT最少
        # operation_lst = myrules.LWKR_PT_o_jointActor(sample_t, sample_p, "max", rule_type="most")  # 余下加工能耗最大
        operation_lst = myrules.LWKR_PT_o_jointActor(sample_t, sample_p, "mean", rule_type="most")  # 余下加工能耗最大
    elif o_rule == 6: # TODO 1217-完全随机，注意处理一下，不要用到这个！最后只跑一次
        operation_lst = myrules.Random_task()  # 完全随机选取的task！

    # print("operation_lst = ", operation_lst)

    """
    不同的m的规则选择：m_index 从0开始的，按照task从1234.。。16开始选择的！！！
    1、选择同一个边的m，本质上是选择transT比较小的m（现在有MInus，不能保证同一个边中的m能做所有task）
    2、一个job的所有m都要用到，这里就不合适了，原因同上，有Minus，不能保证同一job的m都能用上
    3、写2个，选择当前m，导致新增transT和idleT小的！！！ or  EET，选择当前m之后，总的完工时间小的
    """
    if m_rule == 0:
        machine_lst = myrules.SPT_m(sample_t)  # 最小加工时间
    elif m_rule == 1:
        # sample_e1 = np.multiply(sample_t, sample_p)  # 对应元素相乘
        # machine_lst = myrules.SEC_m(sample_e1)  # 最小加工能耗
        machine_lst = myrules.SEC_m(sample_t, sample_p)  # 最小加工能耗
        # machine_lst = myrules.SEC_m(sample_p)  # 最小加工功率
    elif m_rule == 2:
        # machine_lst = myrules.MISE_m(sample_edge_info)  # 同边m，最小运输t
        machine_lst = myrules.Random_m(sample_t)  # 随机选取m，类似于按照运输和空闲！这个不是主要的，先有一个好结果再说啊！！！
    # elif m_rule == 3:
        # machine_lst = myrules.AMU_m()  # 每个job所有m，最大利用率

    # print("machine_lst = ", machine_lst)

    m_rule_name = ['SPT', 'SEC', 'Random']
    o_rule_name = ['FIFO', 'MOR', 'LWKR_T_o', 'LWKR_PT_o', 'MWKR_T_o', 'MWKR_PT_o', 'Random']
    # o_rule_name = ['FIFO', 'MOR', 'Random']
    print("\n ====================rule episode: %d, o_rule: %s + m_rule: %s======================================="
          % (data_index, o_rule_name[o_rule], m_rule_name[m_rule]))
    print("operation_lst = ", operation_lst)
    print("machine_lst = ", machine_lst)

    """开始按照最新的DGFJSPEnv的环境进行选择"""
    step = 0  # 记录选择m的step，也作为m选列表的index

    mch_r_lst, mch_pt_lst, mch_transT_lst, mch_ratio_lst = [], [], [], []  # 记录当前选m的即时r
    opr_r_lst, opr_mk_lst, opr_it_lst, opr_pt_lst, opr_transT_lst = [], [], [], [], []  # 记录当前选o的即时r
    while True:
        """
        参考按照eval的顺序：
        1、选择task_index + m_index = m_id
        2、env.step返回r
        3、单单关注reward，也不需要state，那么不用用到m的env
        """
        # print("*******************************Operation step:{} *******************************************".format(step_cout))

        o_action = operation_lst[step]  # 选择一个action: 是task id列表! - 1 = task的index
        """# 选择当前执行task_index的对应的m_index:0,1,2,3"""
        m_action = machine_lst[o_action - 1]  # 选择当前执行task_index的对应的m_index:0,1,2,3
        joint_actions = [o_action - 1, m_action]  # 都要从0开始！！！

        # print(f"============PDRs Minus: [{joint_actions[0]},{joint_actions[1]}], t={sample_t[joint_actions[0]][joint_actions[1]]}, p= {sample_p[joint_actions[0]][joint_actions[1]]}")
        if sample_t[joint_actions[0]][joint_actions[1]] < 0 :
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        _, reward, done, _, r_t, r_idle, r_energy_m, r_energy_transM, *_ = DGenv.step(joint_action=joint_actions)  # action = task_id - 1 = task_index
        # env.render()

        """
        记录一个episode内的：delta = prev - current
        1、带权重的delta r：makespan + idleT*1
        2、不带权重的delta r：mk，idleT, pt, transT 
        """
        opr_r_lst.append(reward)  # 带权重的cost
        opr_mk_lst.append(r_t)
        opr_it_lst.append(r_idle)
        opr_pt_lst.append(r_energy_m)
        opr_transT_lst.append(r_energy_transM)

        step += 1  # 记录当前是第几步，总共是job*machine步
        # if_print(1, "current: step:{}, reward: {}, done:{}".format(step_cout_m, m_r, m_done))

        """
        1、全部调度完了，有一个reward，不然都是0
        2、每一步reward = -cost？
        3、每一步reward = last - current？ 累加：total = 0 - final_reward
        """
        if done:
            # print("查看所有的reward： rewards = ", rewards)
            # DGenv.render()
            step = 0  # 以便下次使用，但是开头有初始化，以防万一

            """
            记录选择operation的累加Gt
            """
            cost["opr_Gt"] = sum(opr_r_lst)  # 总共的reward，清0之前保存
            cost["opr_mk"] = sum(opr_mk_lst)
            cost["opr_idleT"] = sum(opr_it_lst)
            cost["opr_pt"] = sum(opr_pt_lst)
            cost["opr_transT"] = sum(opr_transT_lst)
            opr_r_lst, opr_mk_lst, opr_it_lst, opr_pt_lst, opr_transT_lst = [], [], [], [], []

            """至今为止：一定要在done之后reset之前读取：全选完的当前this step的cost，没有权重，不用累加，不用相减"""
            mk = DGenv.makespan_previous_step  # 上次的makespan
            pt = DGenv.total_e1_previous_step / n_total_task   # 上次的加工能耗之和（已分配设备的）  TODO 别忘了这里是平均能耗！
            transT = DGenv.trans_t_previous_step  # 至今为止的运输时间t （暂时没有乘以运输设备的e）
            idleT = DGenv.idle_t_previous_step  # 至今为止的空闲时间之和
            untilNow = [mk, pt, transT, idleT]

            # print("**************************************************************************************************")

            # print("mch_Gt: %.5f, mch_pt: %f, mch_transT: %.5f, mch_useRatio: %.5f" % (cost["mch_Gt"],
            #                                                                           cost["mch_pt"],
            #                                                                           cost["mch_transT"],
            #                                                                           cost["mch_useRatio"])
            #       )

            # print("**************************************************************************************************")

            """
            一旦done了，就要重置下环境！！！！ 
            一定记得加上load_instance的重置！浪费了一晚上！！！
            """
            # DGenv.render()  # 看看最终的结果 TODO 1130-没必要render画出来，浪费时间。m有能力缺陷的！
            # print("opr_Gt: %.5f, opr_mk: %.5f, opr_idleT: %.5f, opr_pt: %f, opr_transT: %.5f" % (cost["opr_Gt"],
            #                                                                                      cost["opr_mk"],
            #                                                                                      cost["opr_idleT"],
            #                                                                                      cost["opr_pt"],
            #                                                                                      cost["opr_transT"]))
            print("Real Value: Cost_sum_NoWeight: %.5f, MK: %.5f, PT: %.5f, TT: %f, IT: %.5f" % (sum(untilNow),
                                                                                                 mk,
                                                                                                 pt,
                                                                                                 transT,
                                                                                                 idleT))

            DGenv.reset()
            print("=======================================================reset===============================================================")
            break

    # 这里我只关注最终的EC = e1 + transT*1 + idleT*1
    # totalCost_ec = [a + b + c for a, b, c in zip(net1_totalCost_e, net1_totalCost_trans, totalCost_idle)]  # 画出我关注的指标
    # total_cost = cost["mch_Gt"] + cost["opr_Gt"]  # 带权重的！！！
    total_cost = cost["opr_Gt"]  # 带权重的！！！

    return total_cost, cost, untilNow  # 返回一组样本跑完一个rules的累加即时r（加权后） + 即时r（加权后）和即时4指标的累加dict + 真实4指标值



"""
循环运行rules，记录关注的指标，并画出结果
1、画箱型图
2、一图数据存在一个大list中：每种组合都是其中的一个小list
3、依次循环遍历所有Rules的组合即可
"""
if __name__ == '__main__':   #直接运行时，执行以下代码；导入该模块时，不会运行以下代码

    configs_pwd = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/test/config_test.json"
    # configs_pwd = "E:\PY-code\MO-FJSP-DRL\config.json"  # 这个是本地文件，上述是本地文件的在服务器的投影，内容是一样的！
    
    if os.path.exists(configs_pwd):
        # Load config and init objects
        with open(configs_pwd, 'r') as load_f:
            conf_dict = json.load(load_f)  # json方式来加载，形成conf的字典！
            
    # parse_dict 函数使用了 json.loads(s) 来将一个 JSON 格式的字符串 s 解析为一个 Python 字典。如果传递给 parse_dict 的字符串不符合 JSON 格式，将会引发 json.JSONDecodeError 异常。
    def parse_dict(s):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            raise argparse.ArgumentTypeError("Invalid dictionary format")

    parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象，这个对象将用于解析命令行参数
    
    # ！  TODO：记得同步修改device  = /remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/algorithm/ppo_algorithm.py
    parser.add_argument('--device', type=str, default="cuda:4", help='training device type') # !BUG'cuda'是否不指定gpu？跨GPU? 原先"cuda:0"
    
    parser.add_argument('--n_job', type=int, default=6, help='instance')
    parser.add_argument('--n_machine', type=int, default=6, help='instance')
    parser.add_argument('--n_edge', type=int, default=2, help='instance')
    
    # [[6,6,2],[10,6,2],[20,6,3],[10,10,2],[15,10,2],[20,10,5]],0=传入[6，6，2] + 对应的epi的数值
    parser.add_argument('--mappo_scene', type=parse_dict, default=conf_dict["mappo_id"]["Exist_jme"][0], help='hyper_paras') # iotj模型场景
    parser.add_argument('--mappo_id', type=parse_dict, default=conf_dict["mappo_id"]["Exist_epi"][0], help='hyper_paras') # iotj模型id
    parser.add_argument('--esa_scene', type=parse_dict, default=conf_dict["esa_id"]["Exist_jme"][0], help='hyper_paras') # eswa模型场景
    parser.add_argument('--esa_id', type=parse_dict, default=conf_dict["esa_id"]["Exist_epi"][0], help='hyper_paras') # eswa模型id
    
    parser.add_argument('--weight_mk', type=float, default=0.4, help='instance')
    parser.add_argument('--weight_ec', type=float, default=0.4, help='instance')
    parser.add_argument('--weight_tt', type=float, default=0.2, help='instance')
    
    parser.add_argument('--mask_value', type=float, default=1, help='hyper_paras')  # job的候选节点的mask的赋值
    parser.add_argument('--m_scaling', type=int, default=1, help='hyper_paras') # m选择中的reward放缩比例
    parser.add_argument('--reward_scaling', type=parse_dict, default=conf_dict["reward_scaling"], help='hyper_paras') # DG图中direction选择中的reward放缩比例
    
    args = parser.parse_args()   # 解析命令行参数，并将结果存储在args变量中。
    
    
    total_rules = 12
    pt_lst = [[] for _ in range(total_rules + 1)]  # 初始化为空列表: 最后一位是留给ppo的
    mk_lst = [[] for _ in range(total_rules + 1)]  # 初始化为空列表
    transT_lst = [[] for _ in range(total_rules + 1)]  # 初始化为空列表
    idle_lst = [[] for _ in range(total_rules + 1)]  # 初始化为空列表
    cost_list = [[] for _ in range(total_rules+1)]  # 初始化为空列表

    test_data_pth = '/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/instance/'
    test_dataset_path = test_data_pth + f"test_Instance_J{n_job}M{n_machine}E{n_edge}.pkl"
    rules_data = Instance_Dataset(  # TODO def的所有形参都默认值，这里传入可以只选择要改的值！
            generate_true=0,    # TODO generate_true=0直接读取文件的地址，不需要其他参数
            dataset_pth=test_dataset_path)
    
    m_rules = 2
    o_rules = 6
    rules_epi = 100
    for i_episode in range(rules_epi):   # 实验次数 = eval的次数

        # 遍历所有的Rule的组合
        for m_i in range(m_rules):
            for o_i in range(o_rules):
                # 返回单次episode的Gt + 对应的cost的dict
                cost_temp, cost_dict_episode, untilNow = run_Rules_jointActions_withMinus_1217(args=args,
                                                                                 m_rule=m_i,
                                                                                 o_rule=o_i,
                                                                                 data=rules_data,
                                                                                 data_index=i_episode,
                                                                                 MLWKR_type="max")  # 循环4*6的PDRs，并记录,返回的是列表
                """
                存储比较绕，但是就是对应rule组合存在对应位置
                1、按照0123来进行组合
                2、返回的是单次episode的累加奖励Gt
                3、临时，应该末位留给我的ppo
                """
                pt_lst[o_rules * m_i + o_i].append(cost_dict_episode["opr_pt"])
                mk_lst[o_rules * m_i + o_i].append(cost_dict_episode["opr_mk"])
                transT_lst[o_rules * m_i + o_i].append(cost_dict_episode["opr_transT"])
                idle_lst[o_rules * m_i + o_i].append(cost_dict_episode["opr_idleT"])
                cost_list[o_rules * m_i + o_i].append(cost_temp)
                


    list_fig = []
    list_fig.append(pt_lst)  # 最大完工时间mk + 空闲时间it
    list_fig.append(mk_lst)  # 最大完工时间mk + 空闲时间it
    list_fig.append(transT_lst)  # 最大完工时间mk + 空闲时间it
    list_fig.append(idle_lst)  # 最大完工时间mk + 空闲时间it
    list_fig.append(cost_list)  # 总cost，加权和
    list_fig.append([])  # 防止报错
    # 画图
    result_box_plot(list_fig,2,3)  # 画图1行3列

