import os
import numpy as np
import random
import copy
from collections import deque
from datetime import datetime
from torch.utils.data import Dataset
import pickle
import argparse
import json
import pandas as pd
import csv

def if_print(index, *args):
    if index == 1:
        print(*args)

"""用于记录你运行时的log"""
class StatusLogger:
    def __init__(self, max_size=20000):
        self.log_queue_k = deque(maxlen=max_size)   # 双端队列，再添加前便会被移除
        self.log_queue_v = deque(maxlen=max_size)   # 双端队列，再添加前便会被移除
        self.log_timestamp = deque(maxlen=max_size)   # 双端队列，再添加前便会被移除


    # 记录进log + 同时打印出来，直接省事
    def log(self, key, message, print_true=0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message_k = f"{key}"
        formatted_message_v = f"{message}"
        
        self.log_timestamp.append(f"[{timestamp}]")
        self.log_queue_k.append(formatted_message_k) 
        self.log_queue_v.append(formatted_message_v) 
        
        if print_true:
            print(f'[{timestamp}]  {formatted_message_k}: {message}')  # 时间戳提前，打印
        

    def get_logs(self, pth="logs.txt"):  
        
        logs = dict(zip(self.log_queue_k, self.log_queue_v)) # 作为字典进行输出，方便调用和查看  
        timestamps = list(self.log_timestamp)  # 转为list进行保存和return进行打印. 为空也可以转为list
        i = 0
        with open(pth, "a") as file:    # 追加内容到txt，不会删除原文件。“w”是会直接覆盖
            file.write(f'\n') 
            file.write(f'=' * 250) 
            file.write(f'\n') 
            for key, value in logs.items():  
                if not timestamps:  # 时间列表为空
                    file.write(f'{key}: {value} \n')  # 每个键值对后面添加换行符 
                else:
                    file.write(f'{timestamps[i]} - {key}: {value} \n')  # 每个键值对后面添加换行符 
                i+=1
            # for log in logs:  
            #     file.write(log + "\n")  
        
        self.log_timestamp.clear()  
        self.log_queue_k.clear()  
        self.log_queue_v.clear()  
        return logs
    
    # 返回dict输出
    def log_not_str(self, key, message):
        
        formatted_message_k = f"{key}"
        formatted_message_v = message
        
        self.log_queue_k.append(formatted_message_k) 
        self.log_queue_v.append(formatted_message_v) 
        
    """
    tag=0，清空整个文件
    tag=1,存到excel，需要.xlsx
    tag=2,存到csv
    tag=3,自定义字符串
    """
    def _out_Excel_CSV(self, tag, arr_in=None, Fpath:str= 'default_path.xlsx.csv'):
        if tag ==0 :   #初始化csv，空白csv
            data_df2 = pd.DataFrame([])
            data_df2.to_csv(Fpath, index=False, header=False)  #写一个覆盖的csv，输入空list，相当于新建
        
        if tag == 1:
            data_df = pd.DataFrame(arr_in)
            # data_df.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # 更改每一列的列标识
            # data_df.index = ['1', '2', '3', '4']  #每一行的行标识，不用就注释掉
            writer = pd.ExcelWriter(Fpath)  # 创建名称为test的excel表格 = 'test.xlsx'
            data_df.to_excel(writer, sheet_name='page_1', float_format='%.2f')  # float_format 精度，将data_df写到test表格的第一页中。若多个文件，可以在page_2中写入
            # writer.save()  # 保存    会warning，该功能可能以后没有了
            writer.close()   # 保存，最新版改成这个
        if tag == 2:
            if len(np.array(arr_in).shape) > 2:   # 防止传入的arr_in是一个list，没有shape属性；就先给转成np.array
                for i in range(arr_in.shape[0]):
                    data_df1 = pd.DataFrame(arr_in[i])
                    data_df1.to_csv(Fpath, index=False, header=False, mode='a')
                space = []
                data_df1 = pd.DataFrame(space)
                data_df1.to_csv(Fpath, index=False, mode='a')  # 不要行名，不要列名，mode=a，追加不覆盖
            else:
                data_df1 = pd.DataFrame(arr_in)
                data_df1.to_csv(Fpath, index=False, header=False, mode='a')
                space = []
                data_df1 = pd.DataFrame(space)
                data_df1.to_csv(Fpath, index=False, mode='a')   #不要行名，不要列名，mode=a，追加不覆盖

        if tag ==3 :   #自定义一个list来输入
            data_df2 = pd.DataFrame(arr_in)
            data_df2.to_csv(Fpath, index=False, header=False, mode='a')  #写一个追加的csv，输入自定义list，作为标识符
    
Logger = StatusLogger()  
Result_Logger = StatusLogger()  





"""
DATA很关键的，我只能说，避免以下情况：
1、不能同task，不同m之间加工时间差距很大：会出现都选最小的t时，其余方案的单个m的加工时间比sum最小时间的总和还要大，那还调度个屁！
2、不能人为定义规律：之前我是t的权重0.2，对应p的权重1.8，同一个task中会出现t小的，p*t还是最小的。
3、试验过：只要定义seed之后，就会有固定的随机数序列，每次用np.random.uniform不会重头开始，而是继续生成；除非你重新seed一下，就是重新开始！！！

generate_true=1此时是生成数据并保存，=0直接从pkl文件读取！
"""
# 继承自 Dataset 的类，方便后续使用DataLoader进行数据加载
class Instance_Dataset(Dataset):

    def __init__(self, samples=12800, n_job=6, n_machine=6, n_edge=2, 
                 ability_scope=None, use_PT=0, seed=None, generate_true=1, 
                 csv_pth=None, pkl_pth=None, dataset_pth=None):
        super(Instance_Dataset, self).__init__()

        if generate_true:
            # 初始变量
            j_num = n_job  
            m_num = n_machine
            eachjob_task = m_num  # 默认工序数量等于设备数量
            total_task = j_num * eachjob_task  # 每个job的子任务数量是固定的，不固定的（未来工作来做！）
            Logger.log("instances/scenario", f"job number={j_num}, machine number={m_num}, tasks number={total_task}", print_true=1)
            # if_print(0, "job number: {}, machine number: {}, tasks number: {}".format(j_num, m_num, total_task))

            if seed != None:  # 设置相同的种子，保证生成相同的随机数序列！用于复现和结果验证
                np.random.seed(seed)
                Logger.log("instances/seed", seed, print_true=1)

            """
            保证同一个task，all machine的加工时间和加工能耗是差别不大的！！！
            采用 np.random  保证np.random.seed(seed)是有用的
            """
            t_low = ability_scope["t_low"]  # 平均加工时间
            t_high = ability_scope["t_high"]
            p_low = ability_scope["p_low"]  # 平均功率
            p_high = ability_scope["p_high"]
            e1_low = ability_scope["e1_low"] # 平均能耗（暂未用！）
            e1_high = ability_scope["e1_high"]

            weight_low = ability_scope["weight_low"]  # 从针对每一task的平均t和p开始随机化的权重
            weight_high = ability_scope["weight_high"]
            Logger.log("instances/ability_t_p_scope", f"t_low={t_low}, t_high={t_high}, p_low={p_low}, p_high={p_high}, e1_low={e1_low}, e1_high={e1_high}, weight_low={weight_low}, weight_high={weight_high}", print_true=0)

            tasks_avg_t = np.random.uniform(low=t_low, high=t_high,size=(samples, total_task))  # samples个样本，其中生成每一个task的平均加工时间
            tasks_avg_p = np.random.uniform(low=p_low, high=p_high,size=(samples, total_task))  # samples个样本，其中生成每一个task的平均加工功率

            t_weight = np.random.uniform(low=weight_low, high=weight_high,size=(samples, total_task, m_num))  # samples个样本，其中生成每一个task对应m个设备的平均加工时间
            p_weight = np.random.uniform(low=weight_low, high=weight_high,size=(samples, total_task, m_num))  # samples个样本，其中生成每一个task对应m个设备的平均加工能耗
            Logger.log("instances/avg_t_p", f"tasks_avg_t[-1]={tasks_avg_t[-1]}, tasks_avg_p[-1]={tasks_avg_p[-1]}, size={tasks_avg_t.shape}", print_true=0)
            Logger.log("instances/weight_t_p", f"t_weight[-1]={t_weight[-1]}, p_weight[-1]={p_weight[-1]}, size={t_weight.shape}", print_true=0)
        
            # TODO 1113-新增了p2，但是不用也可以，因为DGENV中有全1的初始值！（否则需要重新训练PPO和MIP）
            m_p2 = np.random.uniform(low=1, high=5,size=(samples, 1, m_num))  # samples个样本，其中生成每一个m的等待功率

            """如果实在不放心，直接一次生成够"""
            ability_t_lst = []  # Ability加工时间的列表
            ability_p_lst = []  # Ability加工能耗的列表
            for i_sample in range(tasks_avg_t.shape[0]):  # 总共samples个样本
                for i_task in range(tasks_avg_t.shape[1]):  # 获取其第2个维度的信息，每个样本有多少个tasks
                    one_task_w_t = t_weight[i_sample][i_task]
                    one_task_t = tasks_avg_t[i_sample][i_task] * one_task_w_t  # 平均时间*权重 = 随机的每一个m的加工时间pt
                    ability_t_lst.append(one_task_t)

                    # # 计算weight的对称值
                    # weight_symm = (ability_scope["weight_low"] + ability_scope["weight_high"]) / 2  # 除以2，求取中间对称数
                    # weight_symm = np.full(shape=m_num, fill_value=weight_symm)
                    # one_task_w_p = (weight_symm - one_task_w_t) + weight_symm  # 取其反向对称的功率的权重

                    one_task_w_p = p_weight[i_sample][i_task]  # 取其功率的权重
                    if not use_PT:  # False：用功率p
                        one_task_p = tasks_avg_p[i_sample][i_task] * one_task_w_p  # 平均功率 * 权重（和pt对应） = 随机的每一个m的加工功率
                        ability_p_lst.append(one_task_p)
                    elif use_PT:  # True: 用能耗p*t
                        one_task_e1 = tasks_avg_p[i_sample][i_task] * one_task_w_p * one_task_t  # 平均功率 * 权重（和pt对应）* 加工时间 = 随机的每一个m的加工能耗e1
                        ability_p_lst.append(one_task_e1)

            ability_t = np.vstack(ability_t_lst).reshape(samples, total_task, m_num)  # 按照垂直方向堆叠成： 16*4; 再按照samples的个数重新整形：(samples, total_task, m_num)
            ability_p = np.vstack(ability_p_lst).reshape(samples, total_task, m_num)  # 按照垂直方向堆叠成： 16*4；再按照samples的个数重新整形
            Logger.log("instances/ability_t_p", f"ability_t[-1]={ability_t[-1]}, ability_p[-1]={ability_p[-1]}, size={ability_t.shape}", print_true=0)
            
            """
            为了防止出现：
            1、大场景会选到同一个machine，干脆和别人统一，都是machine不可能做所有的task，是存在功能差异的！
            2、选择每个task中不能干的m的个数，然后随机选择相同数量的m_index取负数
            3、对应m不能干的t的对应位置，p中也取负号
            """
            ability_t_minus = np.copy(ability_t)  # sample * task * m
            # 对每一行进行随机取负数操作
            for i_sam in range(samples):
                for row in ability_t_minus[i_sam]:
                    num_elements_to_negate = np.random.randint(0, m_num)  # 不包含m_num！ 随机选择0到4个元素进行取负数 随机选择0到4之间的整数，表示几个负数：01234..m_num，不包含5=m_num+1
                    indices_to_negate = np.random.choice(m_num, size=num_elements_to_negate, replace=False)  # 随机选择要取负数的元素索引:范围0123..m_num-1，没有m_num.选择几个？不会有重复！
                    row[indices_to_negate] *= -1  # 取负数操作
            ability_p_minus = np.copy(ability_p)
            # 获取 a 中小于 0 的元素的位置索引
            indices = np.where(ability_t_minus < 0)
            # 将 b_minus 中对应位置的元素取负数
            ability_p_minus[indices] = -ability_p[indices]
            Logger.log("instances/ability_t_p_minus", f"ability_t_minus[-1]={ability_t_minus[-1]}, ability_p_minus[-1]={ability_p_minus[-1]}, size={ability_t_minus.shape}", print_true=0)
            
            # machine in edge's situation
            edge_num = n_edge  # 边的个数
            # machine_list = list(range(m_num))  # 生成machine的列表
            # 生成 s 个列表
            machine_list = [list(range(m_num)) for _ in range(samples)]  # 生成machine的列表,samples个 [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
            Logger.log("instances/ability_edge_init", f"edge_num={edge_num}, initial_machine_list[-1]={machine_list[-1]}, size={np.array(machine_list).shape}", print_true=0)
            # 拆成edge_num个子列表，且长度随机  TODO 暂时是均分，还没有随机！！！！！！！！！！！！！！！
            # output: [[0, 1], [2, 3]]  ability_scope["equal_edge"] = true, 均分
            ability_edge = []
            for i in range(samples):
                edge_machine_info = self.split_list_random(machine_list[i],
                                                    edge_num,
                                                    ability_scope["equal_edge"])  # edge_machine_info是设备的列表，分布在不同的边[0,1,2,....,m_num-1]
                ability_edge.append(edge_machine_info)
            ability_edge = np.array(ability_edge)  # shape = samples * edge_num * m/edge_num
            Logger.log("instances/ability_edge", f"ability_edge[-1]={ability_edge[-1]}, size={np.array(ability_edge).shape}", print_true=0)
            
            """
            machine的部署是已知不变的，所以运输时间ability矩阵也是固定的，查表可得
            """
            # After chosen machine, we need to calculate the weight of edge
            # (transportation time between machine/machine， that is edge/edge)
            # transportation e2 ability  传输时间的m*m的矩阵，表示了m-->m的传输时间
            tr_in_low = ability_scope["transT_in_low"]  # 边内运输t的范围
            tr_in_high = ability_scope["transT_in_high"]  # 边内运输t的范围
            tr_out_low = ability_scope["transT_out_low"]  # 边边运输t的范围
            tr_out_high = ability_scope["transT_out_high"]  # 边边运输t的范围
            Logger.log("instances/ability_trans_scope", f"tr_in_low={tr_in_low}, tr_in_high={tr_in_high}, tr_out_low={tr_out_low}, tr_out_high={tr_out_high}", print_true=0)

            ability_transT = []
            for i_samele in range(samples):
                ability_tr = []
                for i in range(m_num):  # m*m的矩阵
                    for j in range(m_num):
                        flag, d = self.decide_tr_inner_or_inter(i, j, edge_num, ability_edge[i_samele])  # 判断两个m是否在一个边内，边外时计算相对距离
                        if i == j:  # 设备传给设备自身，tr = 0
                            ability_tr.append(0)
                        elif flag == 1:  # i 和 j 是在边内传输
                            ability_tr.append(np.random.uniform(low=tr_in_low, high=tr_in_high, size=1).item())  # [a, b)
                        elif flag == 2:  # i 和 j 是在边边之间传输
                            ability_tr.append(np.random.uniform(low=tr_in_high * d,  # 如果边边离得远，直接方法很多的运输时间
                                                                high=tr_out_high * d,
                                                                size=1).item())  # edge的序号相差太大，我们认为是离得更远，随意用d来放大(Bug:相差1就是6啊，改成从tr_inner_range开始到tr_inter_range*d)
                ability_tr = np.array(ability_tr)  # list转array
                ability_tr = ability_tr.reshape(m_num, m_num)  # 一维变二维   

                # 两个m之间正向和反向的时间是一样的
                U = np.triu(ability_tr, k=1)  # 变成上三角矩阵了
                L = U.T  # 上三角的变成下三角矩阵了
                D = np.diag(np.diag(ability_tr))  # 取对角矩阵
                ability_tt = U + L - D  # 变成上下三角矩阵元素对应相等的矩阵了(先注释)

                ability_transT.append(ability_tt)  # 按照样本数量进行记录生成的m*m的运输时间能力矩阵

            ability_transT = np.array(ability_transT)  # list转array，shape = sample * m * m
            Logger.log("instances/ability_transT", f"ability_transT[-1]={ability_transT[-1]}, size={ability_transT.shape}", print_true=0)
            
            # 找到最后一个'/'的位置,则获取从开始到最后一个'/'的内容
            if os.path.exists(csv_pth[:csv_pth.rfind('/')]):
                # 0清空 + 1 xlsx + 2 csv
                Logger._out_Excel_CSV(0, [], csv_pth)  # 先清空，在保存
                Logger._out_Excel_CSV(3, ['ability_t_minus'], csv_pth)  # 自定义
                Logger._out_Excel_CSV(2, ability_t_minus, csv_pth)
                Logger._out_Excel_CSV(3, ['ability_p_minus'], csv_pth)  # 自定义
                Logger._out_Excel_CSV(2, ability_p_minus, csv_pth)
                Logger._out_Excel_CSV(3, ['ability_transT'], csv_pth)  # 自定义
                Logger._out_Excel_CSV(2, ability_transT, csv_pth)
                Logger._out_Excel_CSV(3, ['ability_edge'], csv_pth)  # 自定义
                Logger._out_Excel_CSV(2, ability_edge, csv_pth)
                Logger._out_Excel_CSV(2, ability_p_minus, csv_pth)
                Logger._out_Excel_CSV(3, ['m_p2'], csv_pth)  # 自定义
                Logger._out_Excel_CSV(2, m_p2, csv_pth)
                
                Logger.log("instances/log_csv", f"save instance in {csv_pth} path", print_true=1)
                
            instance = [ability_t_minus, ability_p_minus, ability_transT, ability_edge] # TODO 250418-产生的instance（lst）用来保存
            with open(pkl_pth, 'wb') as f:
                pickle.dump(instance, f)
            Logger.log("instances/log_pkl", f"save instance in {pkl_pth}.pkl'", print_true=1)
            
        else:
            
            # load dataset   
            with open(dataset_pth, 'rb') as f:
                instances = pickle.load(f)    # 加载数据集，一个list，里边多个轨迹=dict，每个dict包含《st，at，st+1,rt,terminalt》
            Logger.log("instances/load_pkl", f"load instance from {dataset_pth}.pkl'", print_true=1)
            
            """"
            t and p: shape = (samples, total_task, m)
            transT: shape = samples * m * m
            edge: shape = samples * edge_num * m/edge_num(均分)
            """
            # self.t = ability_t
            self.t = instances[0]
            if not use_PT:
                # self.p = ability_p
                self.p = instances[1]
            elif use_PT:
                # self.p = tasks_e1
                pass
            # self.p2 = m_p2
            self.transT = instances[2]
            self.edge = instances[3]
            self.size = len(self.t)  # 就是样本samples的大小

            # 用于测试数据: 倒数后100个数据
            self.t_100_last = self.t[-100:]
            self.p_100_last = self.p[-100:]
            # self.p2_100_last = self.p2[-100:]
            self.transT_100_last = self.transT[-100:]
            self.edge_100_last = self.edge[-100:]
    
    # chatgpt写的
    # [[0, 1], [2, 3]]
    def split_list_random(self, lst, num_splits, use_equal_edge=True): # 默认是均分，除非是你修改了！
        """将列表随机拆分成指定数量的子列表"""
        # random.shuffle(lst)  # 先打乱列表 TODO 250418-后续可以m的分配更加随机
        total = len(lst)
        results = []
        avg_num_m = total // num_splits # 取整数部分！
        for i in range(num_splits):
            if i == num_splits - 1:
                size = total
            else:
                if use_equal_edge:  # machine平均分配在不同的边
                    size = avg_num_m  #设为均分，取整数部分！eg暂时将每个拆分的列表的长度设为2（4个设备2个边）
                else:
                    size = random.randint(1, total - (num_splits - i - 1))  # 随机切分，确保每个子列表至少有一个元素

            results.append(lst[:size])
            lst = lst[size:]
            total -= size
        return results
    
    def decide_tr_inner_or_inter(self, row, col, edgenum, edgemachine_info):
        """判断两个m之间是边内还是边外"""
        in_flag = 0   #边内 or 边外
        distance = 0

        for i in range(edgenum):    # 几个边
            if row in edgemachine_info[i]:   # 先从矩阵的行开始，找到之后开始找列
                for j in range(edgenum):
                    if col in edgemachine_info[j]:  #矩阵的列
                        if i == j:
                            in_flag = 1    # 同一个边内
                        else:
                            in_flag = 2    # 不同边内
                            distance = abs(i-j)  #边与边之间的远近，可能决定了时间的长短

        return in_flag, distance
    
    def __len__(self):
        return self.size

    """"
    t and p: shape = (samples, total_task, m)
    transT: shape = samples * m * m
    edge: shape = samples * edge_num * m/edge_num(均分)
    """
    def __getitem__(self, idx):  # TODO 这一步很关键，返回的是"一个样本"的数据，用来dataloader的调用！！！
        t = self.t[idx]
        p = self.p[idx]
        # p2 = self.p2[idx]
        transT = self.transT[idx]
        edge = self.edge[idx]

        self.ability_dict = {"t": t,  # 记录字典，传递每一个样本的数据：一个factory的能力
                             "p": p,
                            #  "p2": p2,
                             "transT": transT,
                             "edge": edge}

        return self.ability_dict
    
    
    

if __name__ == '__main__':
    
    configs_pwd = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/instance/config_ins.json"
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
    parser.add_argument('--train_samples', type=int, default=12800, help='instance')
    parser.add_argument('--eval_samples', type=int, default=100, help='instance')
    parser.add_argument('--n_job', type=int, default=6, help='instance')
    parser.add_argument('--n_machine', type=int, default=6, help='instance')
    parser.add_argument('--n_edge', type=int, default=2, help='instance')
    parser.add_argument('--use_PT', type=int, default=0, help='hyper_paras') # 是否在生成factory信息采用e1=pt作为能力输出
    parser.add_argument('--ability_scope', type=parse_dict, default=conf_dict["ability_scope"], help='instance')
    parser.add_argument('--train_seed', type=int, default=0, help='instance')
    parser.add_argument('--eval_seed', type=int, default=1, help='instance') 
    parser.add_argument('--test_seed', type=int, default=3, help='instance') 
    args = parser.parse_args()   # 解析命令行参数，并将结果存储在args变量中。
    config=vars(args)  # vars内置函数，转为dict，表示参数和其值
    
    print('=' * 250)
    
    ins_size = [(6,6,2),(10,6,2),(20,6,3),(10,10,2),(15,10,2),(20,10,5)]
    pth = '/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/instance'
    # with open(log_file, 'w') as file:
    #     # 不需要写入任何内容，直接打开文件就会清空它 # 打开文件以写入模式（'w'），这将清空文件内容
    #     pass
    
    for ins_n in range(len(ins_size)-5):  # s TODO 250418-暂时只生成662的数据
        
        csv_pth = pth + "/ABILITY_J%sM%sE%s_instances.csv" % (ins_size[ins_n][0], ins_size[ins_n][1], ins_size[ins_n][2])
        pkl_pth = pth + f'/Instance_J{ins_size[ins_n][0]}M{ins_size[ins_n][1]}E{ins_size[ins_n][2]}.pkl'  
        log_file = pth + f"/log_instance_generate_J{ins_size[ins_n][0]}M{ins_size[ins_n][1]}E{ins_size[ins_n][2]}.txt" 
        
        """ 
        新的Dataset：（一口气生成所有样本）
        1、用于生成作为训练数据依据的factory能力信息
        2、可以用seed来固定相同的随机序列了
        3、工厂的4个指标：加工时间t，加工能耗p，运输时间t，m在edge的部署情况
        4、作为环境env_batch，可以采用dataloader进行数据选取，保证seed有用！
        """
        train_instance = Instance_Dataset(  # TODO def的所有形参都默认值，这里传入可以只选择要改的值！
            samples=config['train_samples'],
            n_job=ins_size[ins_n][0],
            n_machine=ins_size[ins_n][1],
            n_edge=ins_size[ins_n][2],
            ability_scope=config['ability_scope'],
            use_PT=config['use_PT'],
            seed=config['train_seed'],
            generate_true=1,
            csv_pth=csv_pth,
            pkl_pth=pkl_pth)  # 固定seed，samples相同时，生成一样；samples不同，生成数据不同 TODO generate_true=1不用写读取文件的地址
        
        logg = Logger.get_logs(log_file)  # TODO 返回所有保存的log，log是dict形式，储存的txt文件是"追加"模式，会不断很长
        
        eval_csv_pth = pth + "/eval_ABILITY_J%sM%sE%s_instances.csv" % (ins_size[ins_n][0], ins_size[ins_n][1], ins_size[ins_n][2])
        eval_pkl_pth = pth + f'/eval_Instance_J{ins_size[ins_n][0]}M{ins_size[ins_n][1]}E{ins_size[ins_n][2]}.pkl'  
        eval_log_file = pth + f"/eval_log_instance_generate_J{ins_size[ins_n][0]}M{ins_size[ins_n][1]}E{ins_size[ins_n][2]}.txt"
        
        eval_instance = Instance_Dataset(
            samples=config['eval_samples'],
            n_job=ins_size[ins_n][0],
            n_machine=ins_size[ins_n][1],
            n_edge=ins_size[ins_n][2],
            ability_scope=config['ability_scope'],
            use_PT=config['use_PT'],
            seed=config['eval_seed'],
            generate_true=1,
            csv_pth=eval_csv_pth,
            pkl_pth=eval_pkl_pth)  # 固定seed，samples相同时，生成一样；samples不同，生成数据不同 TODO generate_true=1不用写读取文件的地址
        
        logg = Logger.get_logs(eval_log_file)  # TODO 返回所有保存的log，log是dict形式，储存的txt文件是"追加"模式，会不断很长
        
        test_csv_pth = pth + "/test_ABILITY_J%sM%sE%s_instances.csv" % (ins_size[ins_n][0], ins_size[ins_n][1], ins_size[ins_n][2])
        test_pkl_pth = pth + f'/test_Instance_J{ins_size[ins_n][0]}M{ins_size[ins_n][1]}E{ins_size[ins_n][2]}.pkl'  
        test_log_file = pth + f"/test_log_instance_generate_J{ins_size[ins_n][0]}M{ins_size[ins_n][1]}E{ins_size[ins_n][2]}.txt"
        
        test_instance = Instance_Dataset(
            samples=config['eval_samples'],   # test和eval的数据集的大小是一样的
            n_job=ins_size[ins_n][0],
            n_machine=ins_size[ins_n][1],
            n_edge=ins_size[ins_n][2],
            ability_scope=config['ability_scope'],
            use_PT=config['use_PT'],
            seed=config['test_seed'],
            generate_true=1,
            csv_pth=test_csv_pth,
            pkl_pth=test_pkl_pth)  # 固定seed，samples相同时，生成一样；samples不同，生成数据不同 TODO generate_true=1不用写读取文件的地址

        logg = Logger.get_logs(test_log_file)  # TODO 返回所有保存的log，log是dict形式，储存的txt文件是"追加"模式，会不断很长
        # f'data/{env_name}-{dataset}-v2.pkl'
        
          



