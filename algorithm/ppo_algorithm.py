import copy
import os
import random
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from torch.distributions import Categorical
import torch.nn.functional as F

from instance.generate_allsize_mofjsp_dataset import Logger, Result_Logger
from model.actor_critic import Operation_Actor_JointAction_selfCritic, Machine_Actor_JointAction_selfGAT_selfCritic,Global_Critic_JointAction_GAT, esa_Operation_Actor_Critic, esa_Machine_Actor, ablation_Machine_Actor_JointAction_selfGAT_selfCritic
from trainer.fig_kpi import get_GPU_usage
from model.gcn_mlp import Encoder, aggr_obs, g_pool_cal

from trainer.train_device import device

"""
当调用这整个py文件时：
1、除非用了if __name__ == '__main__': 不会被执行代码
2、不然，所有的代码都会被执行的，包括变量（在其他py中只有import了该变量，才能直接使用）会被用在class的变量中，赋值会改变

`torch.device` 类本身并不直接指定在哪个 GPU 上运行，它只用于指定设备类型（GPU 或 CPU）
"""

model_path = ('./model/ppo_actor.pth', './model/ppo_critic.pth')

# n_actionM, n_actionO, lr, jobnum, machinenum, in_size, hide_size, m_out_size, o_out_size,
#                  v_out_size, factory_in_size,
#                  load_pretrained, model_func,
#                  batch_size

class PPOAlgorithm(object):
    def __init__(self, args, load_pretrained):
        
        self.n_job = args['n_job']   # 后续想调用的，可以用self.的形式，其他没必要
        self.n_machine = args['n_machine']
        self.n_total_task = self.n_job * self.n_machine
        self.batch_size = args['env_batch']

        self.GAMMA = args['GAMMA'] # reward折扣率
        self.LAMDA = args['LAMDA'] # GAE参数
        self.epsilon = args['epsilon']  # 重要性采样的裁剪
        self.ENTROPY_BETA = args['ENTROPY_BETA']  # 偏向于出现0001，那就调小！！！！  平衡熵正则化
        # self.a_update_steps = args['']a_update_steps  # actor网络在第N步训练多少次
        # self.c_update_steps = args['']c_update_steps  # critic网络在第N步训练多少次

        # self.w_v_loss = args['']w_m_v_loss

        """
        CNN、MLP、新版合并action和critic的网络参考/remote-home/iot_wangrongkai/FJSP-LLM-250327/MO-FJSP-DRL/ppo_algorithm.py  TODO 250422-删除为了简便，别纠结，魔改！
        """
        
        """
        初始化GCN的网络：
        1、分别初始化：选择machine + 选择operation + 全局critic网络（单独列出，不像eswa的合并在选择job中，会用task节点+全图节点+m的全图节点来训练）
        2、因为我想的全局critic分别用了选m和选o的数据，优化器不好选择
        3、不用old网络，直接单个网络采集+记录作为old。ReplayBuffer方式
        """
        self.job_actor = Operation_Actor_JointAction_selfCritic(configs=args).to(device)  # 别忘了上GPU.初始化无参数传入
        self.job_actor_optimizer = torch.optim.Adam(self.job_actor.parameters(),
                                                    lr=args['LR'],
                                                    eps=args['lr_eps'])  # 修改adam优化器默认的eps，提高训练性能
        # 学习率的调度器，逐渐衰减！（每过step_size就乘以一个系数衰减）
        self.job_actor_lr_decay = torch.optim.lr_scheduler.StepLR(self.job_actor_optimizer,
                                                                  step_size=args['decay_step_size'],
                                                                  gamma=args['decay_ratio'])

        self.machine_actor_gcn = Machine_Actor_JointAction_selfGAT_selfCritic(configs=args).to(device)  # m节点的池化聚合 + MLPActor的网络
        # TODO 0108-新增消融实验，只用1个m的状态，然后只能用MLP！
        # # self.machine_actor_gcn = ablation_Machine_Actor_JointAction_selfGAT_selfCritic(configs=args).to(device)  # m节点的池化聚合 + MLPActor的网络
        self.machine_actor_optimizer_gcn = torch.optim.Adam(self.machine_actor_gcn.parameters(),
                                                            lr=args['LR'],
                                                            eps=args['lr_eps'])  # 修改adam优化器默认的eps，提高训练性能
        # 学习率的调度器，逐渐衰减！（每过step_size就乘以一个系数衰减）
        self.machine_actor_lr_decay_gcn = torch.optim.lr_scheduler.StepLR(self.machine_actor_optimizer_gcn, 
                                                                  step_size=args['decay_step_size'],
                                                                  gamma=args['decay_ratio'])

        
        self.global_critic = Global_Critic_JointAction_GAT(configs=args).to(device)  # m节点的池化聚合 + MLPActor的网络
        self.global_critic_optimizer = torch.optim.Adam(self.global_critic.parameters(),
                                                        lr=args['LR'],
                                                        eps=args['lr_eps'])  # 修改adam优化器默认的eps，提高训练性能
        # 学习率的调度器，逐渐衰减！（每过step_size就乘以一个系数衰减）
        self.global_critic_lr_decay = torch.optim.lr_scheduler.StepLR(self.global_critic_optimizer,
                                                                      step_size=args['decay_step_size'],
                                                                      gamma=args['decay_ratio'])
        
        Logger.log("Training/ppo/network_architecture", f"Operation_Actor: self.job_actor={self.job_actor}, Machine_Actor: self.machine_actor_gcn={self.machine_actor_gcn}, Global_Critic: self.global_critic={self.global_critic}", print_true=1)
        
        

        # TODO 1223-ESA的方式，没有第三个单独的critic网络，而是直接嵌入在job网络中，看看效果！
        """
        复现ESA方法：
        1、啥我都写过，无非就是复习下重新写下，不用担心
        2、就好比又在重新写一个新的想法，一天写完然后有结果就完事了！
        """
        self.esa_job_actor = esa_Operation_Actor_Critic(configs=args).to(device)  # 别忘了上GPU.初始化无参数传入
        self.esa_job_actor_optimizer = torch.optim.Adam(self.esa_job_actor.parameters(),
                                                    lr=args['LR'],
                                                    eps=args['lr_eps'])  # 修改adam优化器默认的eps，提高训练性能
        # 学习率的调度器，逐渐衰减！（每过step_size就乘以一个系数衰减）
        self.esa_job_actor_lr_decay = torch.optim.lr_scheduler.StepLR(self.esa_job_actor_optimizer, 
                                                                      step_size=args['decay_step_size'],
                                                                      gamma=args['decay_ratio'])

        self.esa_machine_actor_gcn = esa_Machine_Actor(configs=args).to(device)  # m节点的池化聚合 + MLPActor的网络
        self.esa_machine_actor_optimizer_gcn = torch.optim.Adam(self.esa_machine_actor_gcn.parameters(),
                                                            lr=args['LR'],
                                                            eps=args['lr_eps'])  # 修改adam优化器默认的eps，提高训练性能
        # 学习率的调度器，逐渐衰减！（每过step_size就乘以一个系数衰减）
        self.esa_machine_actor_lr_decay_gcn = torch.optim.lr_scheduler.StepLR(self.esa_machine_actor_optimizer_gcn,
                                                                              step_size=args['decay_step_size'],
                                                                              gamma=args['decay_ratio'])
       
        Logger.log("Training/ppo/esa/network_architecture", f"Operation_Actor: self.esa_job_actor={self.esa_job_actor}, Machine_Actor: self.esa_machine_actor_gcn={self.esa_machine_actor_gcn}", print_true=1)
        
        
        """
        loss函数：.backward 计算梯度
        优化器：.step 参数更新
        """
        self.global_critic_loss_func = torch.nn.MSELoss()  # 均方误差，公式写的是这个
        
        """
        因为是done之后才更新网络
        记录需要的参数: v,r,action对数概率,
        记得清0！
        """
        self.state_values, self.rewards, self.log_a, self.softmax_a, self.log_actions = [], [], [], [], []
        self.m_state_values, self.m_rewards, self.m_log_a, self.m_softmax_a, self.m_log_actions = [], [], [], [], []
        # self.grad_norm = []

        """
        remaining_m剩余任务数量池：每个job的剩余任务数量:
        初始化某些变量 | 一定记得要清0，或者重新赋值
        """
        self.remaining_m = {}  # 创建一个字典作为每个job的剩余任务数量，用作掩码，排除不能选择的job（即行数） TODO eval的时候使用
        for i in range(self.n_job):
            self.remaining_m[i] = self.n_machine  # machine默认就是每个job的子任务数量
        """
        不能用self.remaining_m循环append在list，同一个引用，改一个值其他都变化了！
        每次都有一个新的初始化变量，用来append；内存里变回自动分配新的索引，否则就是同一个索引
        """
        self.remaining_m_batch = []
        for _ in range(self.batch_size):
            remain_m = {}  # 初始化字典，用来添加到列表!一定要在这里重新初始化！！！！！！！！！！！！！！！！！！！！！！！！！
            for j in range(self.n_job):
                remain_m[j] = self.n_machine  # machine默认就是每个job的初始的子任务数量
            # remain_m = {0:4,1:4,2:4,3:4}
            self.remaining_m_batch.append(remain_m)  # 记录batch_size个的各个job的剩余task个数
        # print("--test: init self.remaining_m_batch = ", self.remaining_m_batch, len( self.remaining_m_batch))

        """
        pool_task_dict任务池：每个job可选的task的id（从1开始）
        """
        self.pool_task_list = [1 + self.n_machine * i for i in range(self.n_job)]  # 每一行的首个task的id，TODO 一直不会变，也不会修改

        self.pool_task_dict = {}  # 用作更新的任务池，可选择task还有哪些？  TODO eval的时候使用
        for i in range(self.n_job):
            self.pool_task_dict[i] = self.pool_task_list[i]
        """
        不能用self.pool_task_dict循环append在list，同一个引用，改一个值其他都变化了！
        每次都有一个新的初始化变量，用来append；内存里变回自动分配新的索引，否则就是同一个索引
        """
        self.pool_task_dict_batch = []  # 用作更新的任务池的batch版本，可选择task还有哪些？
        for _ in range(self.batch_size):
            task_dict = {}  # 初始化字典，用来添加到列表!一定要在这里重新初始化！！！！！！！！！！！！！！！！！！！！！！！！！
            for i in range(self.n_job):
                task_dict[i] = self.pool_task_list[i]
            self.pool_task_dict_batch.append(task_dict)  # 记录batch_size个的各个job的可以选择的task的id = candidate的候选task的id
        # print("--test: init self.pool_task_dict_batch = ", self.pool_task_dict_batch, len(self.pool_task_dict_batch))

        """
        选operation的logits掩码，记得清0初始化 （job的mask）
        """
        lst = [0.0] * self.n_job  # ! TODO job_mask就是屏蔽当前可以选择的job（一列一列来选择，保证紧凑，所有指标都好！不然就是我旧方法，完全随机选择！）
        self.mask_new = torch.tensor(lst).cuda()  # job选择时的mask机制（第一列完全随机，后续按照candidate最小的先选）     TODO eval的时候使用
        
        self.mask_new_batch = []
        for j in range(self.batch_size):
            self.mask_new_batch.append(lst)  # 用于和batch版本的logit进行相加，注意转成tensor
        self.mask_new_batch = torch.tensor(self.mask_new_batch).cuda()  # 转成tensor用于和net中的logit相加， torch.tensor([0.0, 0.0],[0.0, 0.0],...)
        
        self.chosen_taskID_list = []  # 最终选择的task的id  TODO eval的时候使用
        self.chosen_taskID_list_batch = [[] for _ in range(self.batch_size)]  # ! TODO 所有batch中所选择的task的id的, 按照step顺序在对应bs的list中进行append 原chosen_action_list_batch
        
        """
        加载trained model，默认是False：先不加载
        1、可能会在双层学习中用到！
        """
        # if load_pretrained and os.path.exists(model_path[0]):
        #     print('------------load the model----------------')
        #     self.actor_model.load_state_dict(torch.load(model_path[0]))
        #     self.critic_model.load_state_dict(torch.load(model_path[1]))
        # if load_pretrained and os.path.exists(d_model_path):
        #     print('------------load the model----------------')
        #     self.model.load_state_dict(torch.load(d_model_path))

    def esa_update_chosenTaskID_CandidateTaskIDx_JobMask(self, paralenv, action_batch, mask_value):

        """更新用于记录各个job剩余子任务数 + 当前可执行taskID + 所选择的TaskID + job是否可选择的mask = 用于记录当前env中DG的状态，也可用于更新DG图"""
        for i_batch in range(self.batch_size):

            index_a = action_batch[i_batch].data.item()  # 每一个batch的对应所选择的action的值，会重新赋值被覆盖，不用清0

            """
            self.remaining_m_batch, 大list存bs个dict，(job:剩余task个数) = 选择job数量之后就相应-1，到0之后会在选择action的时候就不再选择的
                每个job（每行）剩余可以选择的子任务数（初始=m数）
            pool_task_dict_batch 记录batch_size个的各个job的可以选择的task的id，从1开始的
            chosen_action_list_batch 每个batch中选择的task的id，从1开始的, 按照step顺序在对应bs的list中进行append
            
            1、转化：我选的是job的index，需要转换成当前job的可执行的task的id！！！
                保证我选的都是对的，所以不会存在+1超过下一行首位数值的情况！！
            """
            if self.remaining_m_batch[i_batch][index_a] != 0:  # job还能选，TODO 注意传进来的action_batch是tensor，要用标量值要.data.item()
                self.remaining_m_batch[i_batch][index_a] -= 1                
            
            self.chosen_taskID_list_batch[i_batch].append(self.pool_task_dict_batch[i_batch][index_a])  # 按照job的action_index，从pool里边选择当前job的可选择的taskID，添加到chosen_taskID_list_batch
            
            """
            pool_task_dict_batch可选的task的id不能超限：
            1.超限：当前job的最后一个可选的task被选，下列会无脑+1，就会超限
            2、虽然超限，但是有mask在，会对当前可选节点打分，用的还是当前job的最后一个task的节点嵌入，会一样打分，但是会用mask来屏蔽打分，剩下的才有概率
            3、上述方法：打分会每次多用job的最后一个task，能否改成candidate的个数减少？？？
            """
            if self.remaining_m_batch[i_batch][index_a] != 0:  # 都减到0了，说明选完了没有剩余m，可选task的id就固定在同job的最后一个task，不再更新
                self.pool_task_dict_batch[i_batch][index_a] += 1  # 按照从左往右，一次+1；更新可供选择的子任务（子任务有先后，不能直接选后续的子任务）

            """
            不能对action直接修改，因为是网络训练出来的参数，会记录数据用来更新梯度的！！
            mask_new大小：batch_size * job
            mask要直接加到softmax的输出层之前，logit层那里！
            """
            # 处理概率为0的情况, 遍历！
            for key, value in self.remaining_m_batch[i_batch].items():
                if value == 0:
                    self.mask_new_batch[i_batch][key] = mask_value  # 遍历各个job的剩余task个数，等于0的相应job不可选 TODO 这是我原始的mask的判断，只判断没有task可选！！！
        
        """ Log分别对应 = [batch*dict(job:m)], [batch*[123..]], [batch*dict(job:cur_taskID)], [batch * [job01]] """
        # Logger.log("Training/while/ppo/update_RemainingM_AvailableTaskID_chosenTaskID_jobMask", f"self.remaining_m_batch={self.remaining_m_batch}, self.chosen_taskID_list_batch={self.chosen_taskID_list_batch}, self.pool_task_dict_batch={self.pool_task_dict_batch}, self.mask_new_batch={self.mask_new_batch}", print_true=1)  # = [batch*dict(job:m)], [batch*[123..]], [batch*dict(job:cur_taskID)], [batch * [job01]]
        
        """------------------------------------ 以上是我原版的jobMask=只关注没有剩余task可选，训练好了效果可以有很大提升----------------------------------------------"""
       
        # TODO 1223-按照ENV中更新之后的节点的“finish time”进行mask的选取！min才能选 + candidate可以不用管，虽然会更新到下一列，但是mask限制了，选不到，所以不用改！
        """
        self.G.nodes[task_id]['finish_time'] = taskid=123。。。
        paralenv.paral_env_DG[i_bs] = 当前的并行的env
        
        remaining_m_batch: list,里边存了bs个dict，每个dict记录所有job=0123的剩余子任务个数！ = [batch*dict(job:m)]
        pool_task_dict_batch：list,里边存了bs个dict，每个dict记录所有job的当前可以选择的子任务的id（从1开始） = [batch*dict(job:cur_taskID)]
        mask_new_batch: bs*j的张量，记录T和F，T就不能选 = [batch * [job01]] 
        
        self.pool_task_list是永远不变的 = 1，4, 7（3*3场景）
        
        傻瓜式遍历，有用就行！ + 替换我的mask的值！！！
        思路：
        每一个bs输出的都是job个T/F，先判断job是否选完，然后判断是否被选过了，剩下的第一列要随机选择，之后每一列都判断每个job行中最大预估ft时间最短的，选择该job！
            其中，DG中没选择的task的ft是None（虽然会预估但是不会更新在DG的node中，只会更新各个node的state向量），然后初始化j*m全为0，只有被选择的task的ft才有值且都>0，所以找每一job行的max，就是当前被选过节点的最大ft（指的是当前每一个job的可选task的前置task的完工时间，从中对比找最小的）
        行列=job*m，方正的矩阵，每次选择都是有candidate的！不是按照一列一列的，而是每个job的前置task结束最早的，就选这个！
        """
        mask_operation_batch = self.mask_new_batch.bool()  # 01float张量转成true和false TODO 1223-原始的mask的判断的方式，01张量转成bool张量 = bs*j，原版基础上改
        for i_bs in range(self.batch_size):
            
            eachbs_mask = [0] * self.n_total_task  # 初始化都可以选择, 所有task的mask，但是我们只用某一列的
            eachbs_ft = [0] * self.n_total_task  # 初始化每一个task的ft都是0

            for i_task in range(self.n_total_task): # 遍历所有的task
                if paralenv.paral_env_DG[i_bs].G.nodes[i_task+1]['finish_time'] is not None: # 表明此时已经选择了task
                    eachbs_mask[i_task] = 1  # 对已经选择task的mask置1
                    eachbs_ft[i_task] = paralenv.paral_env_DG[i_bs].G.nodes[i_task+1]['finish_time'] # 被选的task的ft更新进list
            eachbs_mask_np = np.array(eachbs_mask).reshape(self.n_job, self.n_machine) # 存有每一个task的是否被选的mask，被选之后，就不能再选这个task了
            eachbs_ft_np = np.array(eachbs_ft).reshape(self.n_job, self.n_machine) # 存有每一个task的完工时间，没选的为0

            eachcol_mask_sum = np.sum(eachbs_mask_np,axis=0)  # 对二维矩阵的每一列进行求和，得到一维的nparray，1*m
            # 找到二维nparray每一行的最大值，返回的是list
            """! TODO 关键：每一行中最大ft=被选task，这些上一step被选的task，谁ft结束的早，先选谁！不管是不是同一列！"""
            max_values_in_each_row = [max(row) for row in eachbs_ft_np] # 每一行=每一个job的最大的ft，就是当前判断的依据，其中最小的就是除第一列之外可以选择的task！！！
            for i_col in range(self.n_machine): # 遍历每一列！
                if i_col != 0:  # 非第一列。若第一列，就是随机选！(不用mask)，
                    if eachcol_mask_sum[i_col-1] == self.n_job and eachcol_mask_sum[i_col] != self.n_job:  # 遍历：前一列=self.n_job，当前列！=job，说明没有选完，需要判断ft的大小
                        # 根据张量self.mask_new_batch[i_bs]的取值情况，在max_values_in_each_row的对应位置赋值inf，永远不会选到（inf最大）
                        for i in range(len(self.mask_new_batch[i_bs])):  # 遍历当前bs的所有job
                            if self.mask_new_batch[i_bs][i] == 1:  # 说明该job选完了
                                max_values_in_each_row[i] = float("inf")
                        # 找到所有最小值的索引
                        min_value = min(max_values_in_each_row)
                        min_indexes = [index for index, value in enumerate(max_values_in_each_row) if value == min_value]  # list ：当前拥有最小值ft的作为可选的当前列的task（如果ft相同且最小，那就都随机选）
                        mask_temp = [1] * self.n_job  # ESA：后续都是每一列只有min的才可以选择，所以初始全1，表示都不能选！
                        for iii in range(len(min_indexes)):  # 有几个最小的ft，这些最小ft对应的index即为可选择的job的index
                            mask_temp[min_indexes[iii]] = 0 # 0表示是可以选择的！！！！
                        mask_operation_batch[i_bs] = torch.tensor(mask_temp).bool()  # 转成tensor，然后直接01转bool
                elif i_col == 0: # 第一列的特殊情况
                    if eachcol_mask_sum[i_col] != self.n_job:  # 不等于job说明没选完
                        mask_operation_batch[i_bs] = torch.tensor(eachbs_mask_np[:, i_col]).bool()  # 第一列的01转成bool，被选过不能再选，从该列其他的开始选
                    else: # =job 说明第一列选完了，啥也不做，下一个判断
                        pass

        """
        我用的candidate和mask都是取于原先的程序 
        1、可选的task的池子pool_task_dict_batch== [batch*dict(job:cur_taskID)]---减1作为candidate = env_batch * job （np.array）
        2、更新选完action的mask---转为true和false作为mask = env_batch * job （tensor.bool（）） TODO 即使有candidate，也会被mask屏蔽
        """
        value_list = []
        for dict in self.pool_task_dict_batch:  # 遍历list中的所有dict，储存的是每一batch中可选的task的id
            value_list.append(list(dict.values()))  # 将列表中的字典的所有值转换为列表
        candidate_batch = np.array(value_list) - 1  # list转np.array，然后taskid-1转成对应的索引index
        # Logger.log("Training/while/ppo/update_Candidate_newjobMask", f"candidate_batch={candidate_batch}, mask_operation_batch={mask_operation_batch}", print_true=1)   # env_batch * job, mask采用bool形式

        """
        1、candidate：batch * job (注意：需要记录的是可选的task的索引值，为了方便gather提取特征！)
        2、mask：batch * job (注意：原有的是float01的张量，这里转成false和true的张量，mask_value=1来赋值！！)
        """
        return candidate_batch, mask_operation_batch  # candidate没有变化 + mask重新替代了！！！！！！
        # return candidate_batch, self.mask_new_batch.bool()  # candidate没有变化 + TODO 250508-mask改为旧版本的只判断是否选择完了(收敛但非最优)
    
    
    """evaluation模式下的更新candidate和job_mask的函数"""
    def Eval_esa_update_chosenTaskID_CandidateTaskIDx_JobMask(self, env, action_batch, mask_value):

        """更新用于记录各个job剩余子任务数 + 当前可执行taskID + 所选择的TaskID + job是否可选择的mask = 用于记录当前env中DG的状态，也可用于更新DG图"""
        index_a = action_batch.data.item()  # 每一个batch的对应所选择的action的值，会重新赋值被覆盖，不用清0
        
        # 选择job数量之后就相应-1，到0之后会在选择action的时候就不再选择的 = 每个job（每行）剩余可以选择的子任务数（=m数）
        if self.remaining_m[index_a] != 0:  # 注意传进来的action_batch是tensor，要用标量值要.data.item()
            self.remaining_m[index_a] -= 1
            
        # 转化：我选的是job的index，需要转换成当前job的可执行的task的id！！！证我选的都是对的，所以不会存在+1超过下一行首位数值的情况！！
        self.chosen_taskID_list.append(self.pool_task_dict[index_a])  # 按照随机产生的index，从pool里边选择动作，添加到chosen_action_list
        
        """
        可选的task的id不能超限：
        1.超限：当前job的最后一个可选的task被选，下列会无脑+1，就会超限
        2、虽然超限，但是有mask在，会对当前可选节点打分，用的还是当前job的最后一个task的节点嵌入，会一样打分，但是会用mask来屏蔽打分，剩下的才有概率
        3、上述方法：打分会每次多用job的最后一个task，能否改成candidate的个数减少？？？
        """
        if self.remaining_m[index_a] != 0:  # 都减到0了，说明选完了没有剩余m，可选task的id就固定在同job的最后一个task，不再更新
            self.pool_task_dict[index_a] += 1  # 按照从左往右，一次+1；更新可供选择的子任务（子任务有先后，不能直接选后续的子任务）

        """
        不能对action直接修改，因为是网络训练出来的参数，会记录数据用来更新梯度的！！
        mask_new大小：batch_size * out_size
        mask要直接加到softmax的输出层之前，logit层那里！
        """
        # 处理概率为0的情况, 遍历！
        for key, value in self.remaining_m.items():
            if value == 0:
                self.mask_new[key] = mask_value  # 遍历各个job的剩余task个数，等于0的相应job不可选 TODO 这是我原始的mask的判断，只判断没有task可选！！！
                
        # Logger.log("Evaluation/update_RemainingM_AvailableTaskID_chosenTaskID_jobMask", f"self.remaining_m={self.remaining_m}, self.chosen_taskID_list={self.chosen_taskID_list}, self.pool_task_dict={self.pool_task_dict}, self.mask_new={self.mask_new}", print_true=1)  # = [batch*dict(job:m)], [batch*[123..]], [batch*dict(job:cur_taskID)], [batch * [job01]]
               
        """------------------------------------ 以上是我原版的jobMask=只关注没有剩余task可选，训练好了效果可以有很大提升----------------------------------------------"""

        # todo 1223-按照ENV中更新之后的节点的“finish time”进行mask的选取！min才能选 + candidate可以不用管，虽然会更新到下一列，但是mask限制了，选不到，所以不用改！
        """
        self.G.nodes[task_id]['finish_time'] = taskid=123。。。
        paralenv.paral_oenv[i_bs] = 当前的并行的env

        remaining_m_batch: list,里边存了bs个dict，每个dict记录所有job=0123的剩余子任务个数！
        pool_task_dict_batch：list,里边存了bs个dict，每个dict记录所有job的当前可以选择的子任务的id（从1开始）
        mask_new_batch: bs*j的张量，记录T和F，T就不能选

        self.pool_task_list是永远不变的 = 1，4, 7（3*3场景）

        傻瓜式遍历，有用就行！ + 替换我的mask的值！！！
        思路：
        每一个bs输出的都是job个T/F，先判断job是否选完，然后判断是否被选过了，剩下的第一列要随机选择，之后每一列都判断每个job行中最大预估ft时间最短的，选择该job！
            其中，DG中没选择的task的ft是None（虽然会预估但是不会更新在DG的node中，只会更新各个node的state向量），然后初始化j*m全为0，只有被选择的task的ft才有值且都>0，所以找每一job行的max，就是当前被选过节点的最大ft（指的是当前每一个job的可选task的前置task的完工时间，从中对比找最小的）
        行列=job*m，方正的矩阵，每次选择都是有candidate的！不是按照一列一列的，而是每个job的前置task结束最早的，就选这个！
        """
        mask = self.mask_new.bool()  # 01float张量转成true和false TODO 1223-原始的mask的判断的方式，01张量转成bool张量 = bs*j，原版基础上改

        eachbs_mask = [0] * self.n_total_task  # 初始化都可以选择, 所有task的mask，但是我们只用某一列的
        eachbs_ft = [0] * self.n_total_task  # 初始化每一个task的ft都是0

        for i_task in range(self.n_total_task):  # 遍历所有的task
            if env.G.nodes[i_task + 1]['finish_time'] is not None:  # 表明此时已经选择了task
                eachbs_mask[i_task] = 1  # 对已经选择task的mask置1
                eachbs_ft[i_task] = env.G.nodes[i_task + 1]['finish_time']  # 被选的task的ft更新进list
        eachbs_mask_np = np.array(eachbs_mask).reshape(self.n_job,self.n_machine)  # 存有每一个task的是否被选的mask，被选之后，就不能再选这个task了
        eachbs_ft_np = np.array(eachbs_ft).reshape(self.n_job,self.n_machine)  # 存有每一个task的是否被选的mask，被选之后，就不能再选这个task了

        eachcol_mask_sum = np.sum(eachbs_mask_np, axis=0)  # 对二维矩阵的每一列进行求和，得到一维的nparray，1*m
        # 找到二维nparray每一行的最大值，返回的是list
        """! TODO 关键：每一行中最大ft=被选task，这些上一step被选的task，谁ft结束的早，先选谁！不管是不是同一列！"""
        max_values_in_each_row = [max(row) for row in eachbs_ft_np]  # 每一行=每一个job的最大的ft，就是当前判断的依据，其中最小的就是除第一列之外可以选择的task！！！
        for i_col in range(self.n_machine):  # 遍历每一列！
            if i_col != 0:  # 第一列，就是随机选！
                if eachcol_mask_sum[i_col - 1] == self.n_job and eachcol_mask_sum[i_col] != self.n_job:  # 遍历：前一列=3=self.n_job，当前列！=3，说明没有选完，需要判断ft的大小
                    # 根据张量self.mask_new_batch[i_bs]的取值情况，在max_values_in_each_row的对应位置赋值inf，永远不会选到（inf最大）
                    for i in range(len(self.mask_new)):
                        if self.mask_new[i] == 1:  # 说明该job选完了
                            max_values_in_each_row[i] = float("inf")
                    # 找到所有最小值的索引
                    min_value = min(max_values_in_each_row)
                    min_indexes = [index for index, value in enumerate(max_values_in_each_row) if value == min_value]  # list ：当前拥有最小值ft的作为可选的当前列的task（如果ft相同且最小，那就都随机选）
                    mask_temp = [1] * self.n_job  # ESA：后续都是每一列只有min的才可以选择，所以初始1
                    for iii in range(len(min_indexes)):
                        mask_temp[min_indexes[iii]] = 0  # 0表示是可以选择的！！！！
                    mask = torch.tensor(mask_temp).bool()  # 转成tensor，然后直接01转bool shape = j, 只有一个env！
            elif i_col == 0:  # 第一列的特殊情况，不等于3说明没选完
                if eachcol_mask_sum[i_col] != self.n_job:
                    mask = torch.tensor(eachbs_mask_np[:, i_col]).bool()  # 第一列的01转成bool，被选过不能再选，从该列其他的开始选
                else:  # =job 说明第一列选完了，啥也不做，下一个判断
                    pass
           
        """
        我用的candidate和mask都是取于原先的程序 
        1、可选的task的池子---减1作为candidate = env_batch * job （np.array）
        2、更新选完action的mask---转为true和false作为mask = env_batch * job （tensor。bool（））
        """
        candidate = np.array(list(self.pool_task_dict.values())) - 1  # 将字典的value列表转为np，然后-1作为task的索引值
        # Logger.log("Evaluation/update_Candidate_newjobMask", f"candidate={candidate}, mask={mask}", print_true=1)   # env_batch * job, mask采用bool形式

        return candidate, mask
        # return candidate, self.mask_new.bool()     # eval阶段，返回候选列表和mask TODO 250508-mask改为旧版本的只判断是否选择完了(收敛但非最优)
    
    
    
    def step_for_net_out_Critic_GAT(self, net_model, task_fea, graph_pool_avg, adj, candidate, machine_fea1, machine_fea2):
        out_temp_lst = []
        # BUG：小改进，是不是直接全输入到net_model中去，直接就是buffer_step * env_batch * 1的输出？
        for i in range(task_fea.shape[0]):  # 求取step的个数: buffer_step
            v = net_model(task_fea[i],
                          graph_pool_avg,
                          adj[i],
                          candidate[i],
                          machine_fea1[i],
                          machine_fea2[i]
                          )  # batch * 1 mlp只有一个state
            out_temp_lst.append(v)
        out = torch.stack(out_temp_lst, dim=0)  # 沿着第一个维度堆叠，list[tensor_1, tensor_2, ..., tensor_32]变成一个大tensor
        return out  # buffer_step * env_batch * 1
    
    """分别计算job和machine的局部的针对特定reward的GAE"""
    def cal_local_job_machine_reward_GAE(self, mk_r, pt_r, tt_r, it_r, jv, jv_, mv, mv_, done_operation):
        """"""
        """注意，这里更新折扣r，然后v采样，然后计算adv，再归一化，固定之后作为同一K_epoch中的loss权重。"""

        """
        jv = jv_ = buffer_step * env_batch * 2, 挑选对应指标mk + it的维度元素，然后squeeze（-1）去掉最后一个维度
        mv = mv_ =  buffer_step * env_batch * 2, 挑选对应指标pt + tt的维度元素，然后squeeze（-1）去掉最后一个维度
        deltas = 32*16 (step * batch)  dones = 32* 16 (step * batch)
        [mk_r, pt_r, tt_r, it_r] = 4个（buffer_step * env_batch）的元素
        """
        adv_lst = []
        differ_r_lst = [mk_r, pt_r, tt_r, it_r]
        differ_v_lst = [jv[:, :, 0], mv[:, :, 0], mv[:, :, 1], jv[:, :, 1]]  # 选取 mk + pt + tt + it 的元素，都是step*bs*1 TODO 听说会自动3维变2维度，不用squeeze
        differ_v__lst = [jv_[:, :, 0], mv_[:, :, 0], mv_[:, :, 1], jv_[:, :, 1]]  # 下一时刻的value值
        for i in range(4):  # 4个指标

            """
            # deltas 将 deltas 的张量直接进行计算，配对：一行对应一行： 16 对应 16
            # zip(reversed()) 函数将两个逆序的数组进行配对，即对应位置的元素一一配对;delta 和 d 分别表示 deltas 和 done 数组中对应位置的元素。
            # *是逐元素相乘,rewards.shape[-1]的大小是batch！
            # adv = torch.Size([32, 16]) 还是一样的大小
            
            # TODO: 是不是还是要加上win_done？防止job选择的时候学到废物信息？比如0001，奇怪：ESWA学的大部分都是废物信息0001？
            # deltas = rewards + GAMMA * (ones_ - dones) * value[1:] - value[:-1] # TD-error的基本形式，加了batch（后续修改done的判断条件）
            # deltas = global_r + self.GAMMA * v_ - v  # TD-error的基本形式，加了batch（后续修改done的判断条件）
            
            1-done: 只是为了防止结束时没有下一时刻的v_（我的场景都有下一时刻！）
            global_r = r + r_operation  # 全局的每一step的即时r shape = buffer_step * env_batch 
            """
            
            global_r = differ_r_lst[i]  # 选择具体指标  buffer_step * env_batch
            v = differ_v_lst[i]  # 选择具体指标    step*bs*1
            v_ = differ_v__lst[i]  # 选择具体指标  step*bs*1

            gae = 0
            adv = []
            # TODO 这里的每一个r的shape = step * env_bs
            deltas = global_r + self.GAMMA * v_.squeeze(-1) - v.squeeze(-1)  # TD-error的基本形式，加了batch（后续修改done的判断条件）
            for delta, d in zip(reversed(deltas), reversed(done_operation)):
                # TODO 每次选取的张量的shape = env_bs，一维度，来处理
                """保证每次只从win_done = 1开始迭代"""
                gae = delta + self.GAMMA * self.LAMDA * gae * (1.0 - d)
                adv.insert(0, gae)  # 插入到list的首位，相当于存到list然后逆序
                # print(f"------GAE: delta = {delta},shape = {delta.shape},\n d = {d}, shape = {d.shape}")
                
            adv = torch.stack(adv)  # adv是list，其中元素是张量，堆叠一下，直接变成一个大张量： 32* 16 (step * batch)--（buffer_step * env_batch）！

            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))  # ! TODO PPO重要trick之一！

            adv_lst.append(copy.deepcopy(adv))

        return adv_lst # 大的list，包含4个reward的对应的adv，adv=(step * batch)
    
    def separate_cal_4_reward_GAE(self, mk_r, pt_r, tt_r, it_r, v, v_, done_operation):
        """"""
        """注意，这里更新折扣r，然后v采样，然后计算adv，再归一化，固定之后作为同一K_epoch中的loss权重。"""

        """
        v = buffer_step * env_batch * 4, 挑选对应指标的第三维度元素，然后squeeze（-1）去掉最后一个维度
        deltas = 32*16 (step * batch)  dones = 32* 16 (step * batch)
        [mk_r, pt_r, tt_r, it_r] = 4个（buffer_step * env_batch）的元素
        """
        adv_lst = []
        differ_r_lst = [mk_r, pt_r, tt_r, it_r]
        for i in range(4):  # 4个指标

            """
            # deltas 将 deltas 的张量直接进行计算，配对：一行对应一行： 16 对应 16
            # zip(reversed()) 函数将两个逆序的数组进行配对，即对应位置的元素一一配对;delta 和 d 分别表示 deltas 和 done 数组中对应位置的元素。
            # *是逐元素相乘,rewards.shape[-1]的大小是batch！
            # adv = torch.Size([32, 16]) 还是一样的大小
            
            1-done: 只是为了防止结束时没有下一时刻的v_（我的场景都有下一时刻！）
            global_r = r + r_operation  # 全局的每一step的即时r shape = buffer_step * env_batch 
            
            # TODO: 是不是还是要加上win_done？防止job选择的时候学到废物信息？比如0001，奇怪：ESWA学的大部分都是废物信息0001？
            # deltas = rewards + GAMMA * (ones_ - dones) * value[1:] - value[:-1] # TD-error的基本形式，加了batch（后续修改done的判断条件）
            # deltas = global_r + self.GAMMA * v_ - v  # TD-error的基本形式，加了batch（后续修改done的判断条件）
            """
            
            global_r = differ_r_lst[i]  # 选择具体指标
            gae = 0
            adv = []
            # TODO 这里的每一个r的shape = step * env_bs，[:, :, 0:1]取的形式还是3维度
            deltas = global_r + self.GAMMA * v_[:, :, i:(i+1)].squeeze(-1) - v[:, :, i:(i+1)].squeeze(-1)  # TD-error的基本形式，v=buffer_step * env_batch * 4
            for delta, d in zip(reversed(deltas), reversed(done_operation)):
                # TODO 每次选取的张量的shape = env_bs，一维度，来处理
                """保证每次只从win_done = 1开始迭代"""
                gae = delta + self.GAMMA * self.LAMDA * gae * (1.0 - d)
                adv.insert(0, gae)  # 插入到list的首位，相当于存到list然后逆序
                # print(f"------GAE: delta = {delta},shape = {delta.shape},\n d = {d}, shape = {d.shape}")
                
            adv = torch.stack(adv)  # adv是list，其中元素是张量，堆叠一下，直接变成一个大张量： 32* 16 (step * batch)--（buffer_step * env_batch）！

            adv = ((adv - adv.mean()) / (adv.std() + 1e-5))  # ！TODO PPO重要trick之一！

            adv_lst.append(copy.deepcopy(adv))

        return adv_lst # 大的list，包含4个reward的对应的adv，adv=(step * batch)

    
    def global_update_JointActions_GAT_selfCritic(self, replay_buffer, all_steps, graph_pool_avg, args, mini_bs):

        """
        GCN网络
        """
        self.job_actor.train()  # 训练的时候切换成训练模式：启用 Dropout 层和批归一化层的训练行为，以及保持计算图用于反向传播和参数更新
        self.machine_actor_gcn.train()
        self.global_critic.train()

        """
        提取经验池数据（张量，GPU）
        1、注意：都会多一个buffer_step的维度，表示存到buffer里边的有多少步
        2、numpy_to_tensor反馈的(都是选择m)： s = machine_fea！！！！

        return:
        adj, tasks_fea, candidate, mask_operation, a_operation, a_logprob_operation,\
        adj_, tasks_fea_, candidate_, mask_operation_, r_operation, done_operation, \
        machine_fea2, a, a_logprob, machine_fea2_, mask_machine_, \
        mk, pt, tt, it, machine_fea1, rw, job_v, machine_v, job_v_, machine_v_
        """

        # print("222:{}".format(get_GPU_usage()[1]))
        
        # Get training data, 转成张量并返回，经验池数据
        adj, tasks_fea, candidate, mask_operation, a_operation, a_logprob_operation, \
            adj_, tasks_fea_, candidate_, mask_operation_, reward, done, \
                machine_fea2, a_machine, a_machine_logprob, machine_fea2_, mask_machine_batch_, \
                    mk_r, pt_r, tt_r, it_r, machine_fea1, random_weight, \
                        job_v, machine_v, job_v_, machine_v_ = replay_buffer.numpy_to_tensor_operation()
        # print("333:{}".format(get_GPU_usage()[1]))
        
        # print("---ReplayBuffer记录的： random_weight = {}， shape={}".format(random_weight, random_weight.shape))
        # TODO 1224-直接发现了我的重复训练的代码有点小bug，那这一个月相当于白白训练了？
        # print("---ReplayBuffer记录的： job_v = {}， shape={}".format(job_v, job_v.shape))
        # print("---ReplayBuffer记录的： machine_v = {}， shape={}".format(machine_v, machine_v.shape))
        # print("---ReplayBuffer记录的： job_v_ = {}， shape={}".format(job_v_, job_v_.shape))
        # print("---ReplayBuffer记录的： machine_v_ = {}， shape={}".format(machine_v_, machine_v_.shape))
        #
        # print("---ReplayBuffer记录的： adj = {}， shape={}".format(adj, adj.shape))
        # print("---ReplayBuffer记录的： adj_ = {}， shape={}".format(adj_, adj_.shape))
        # print("---ReplayBuffer记录的： tasks_fea = {}， shape={}".format(tasks_fea, tasks_fea.shape))
        # print("---ReplayBuffer记录的： tasks_fea_ = {}， shape={}".format(tasks_fea_, tasks_fea_.shape))
        # print("---ReplayBuffer记录的： candidate = {}， shape={}".format(candidate, candidate.shape))
        # print("---ReplayBuffer记录的： candidate_ = {}， shape={}".format(candidate_, candidate_.shape))
        # print("---ReplayBuffer记录的： mask_operation = {}， shape={}".format(mask_operation, mask_operation.shape))
        # print("---ReplayBuffer记录的： mask_operation_ = {}， shape={}".format(mask_operation_, mask_operation_.shape))
        # print("---ReplayBuffer记录的： a_operation = {}， shape={}".format(a_operation, a_operation.shape))
        # print("---ReplayBuffer记录的： reward = {}， shape={}".format(reward, reward.shape))
        # print("---ReplayBuffer记录的： a_logprob_operation = {}， shape={}".format(a_logprob_operation, a_logprob_operation.shape))
        # print("---ReplayBuffer记录的： done = {}， shape={}".format(done, done.shape))
        # print("---ReplayBuffer记录的： machine_fea2 = {}， shape={}".format(machine_fea2, machine_fea2.shape))
        # print("---ReplayBuffer记录的： machine_fea2_ = {}， shape={}".format(machine_fea2_, machine_fea2_.shape))
        # print("---ReplayBuffer记录的： a_machine = {}， shape={}".format(a_machine, a_machine.shape))
        # print("---ReplayBuffer记录的： a_machine_logprob = {}， shape={}".format(a_machine_logprob, a_machine_logprob.shape))
        # print("---ReplayBuffer记录的： mask_machine_batch_ = {}， shape={}".format(mask_machine_batch_, mask_machine_batch_.shape))
        # print("---ReplayBuffer记录的： mk_r = {}， shape={}".format(mk_r, mk_r.shape))
        # print("---ReplayBuffer记录的： pt_r = {}， shape={}".format(pt_r, pt_r.shape))
        # print("---ReplayBuffer记录的： tt_r = {}， shape={}".format(tt_r, tt_r.shape))
        # print("---ReplayBuffer记录的： it_r = {}， shape={}".format(it_r, it_r.shape))
        # print("---ReplayBuffer记录的： machine_fea1 = {}， shape={}".format(machine_fea1, machine_fea1.shape))
        # print("---ReplayBuffer记录的： random_weight = {}， shape={}".format(random_weight, random_weight.shape))

        """
        计算全局变量：
        1、reward = 仅仅使用单独一个DGFJSPEnv环境产生的reward，和task和machine都是对应的
        2、done = done 都是在最后一步停止，即task选完 （暂时没用win_done，所以没有区别！）
        """
        global_r = reward  # 全局的每一step的即时r shape = buffer_step * env_batch

        """
        Calculate the advantage using GAE
        'dw=True' means dead or win, there is no next state s'  （我的场景不存在说没有下一次的state）
        'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        """
        # v = torch.Size([32, 16, 1])  必然是3维度，因为输入是3维的。32次step，16个batch_size
        # reward = 32 * 16
        # .detach() 张量不需要计算梯度  
        action_prob:  step * batch * m  （所有的action的概率）--（buffer_step * env_batch * m）
        具体的select_action：step * batch--（buffer_step * env_batch）
        reward: step * batch --（buffer_step * env_batch）
        done： step * batch--（buffer_step * env_batch）

        禁用梯度计算给注释掉，因为我无法判断什么时候不用！TODO（我用了global_critic重采样只是计算adv和target_v，都用在job_actor网络里，更新时不更新global_critic的网络。，所以可以禁用梯度计算。相反，PPO的重要性采样，因为涉及到本身actor网络的更新，不能禁用 + global_critic网络的重采样，涉及到自身MSEloss的计算，需要更新自身网络，因此也不能禁用梯度）
        ppo中的目标函数的clip裁剪其实是一种正则化技术！
        """
        gae = 0
        adv = []  # 记录GAE方式产生的adv
        rewards = []  # 记录原始PPO的累加折扣reward，视为v_target（ESWA就是这样子做的）
        with torch.no_grad():
            
            # TODO 如果修改了value的输出维度=4，则shape = buffer_step * env_batch * 4
            """依据所采集的轨迹，对critic_value的值进行采样！"""
            multi_v = self.step_for_net_out_Critic_GAT(
                        net_model=self.global_critic,
                        task_fea=tasks_fea,  # step * （env_batch * task） * 12
                        graph_pool_avg=graph_pool_avg,  # step * env_batch * （env_batch * tasks）
                        adj=adj,  # step * env_batch * tasks * tasks
                        candidate=candidate, #
                        machine_fea1=machine_fea1, # step * env_batch * m* 6
                        machine_fea2=machine_fea2  # step * env_batch * m* 8
                        )    # 按照step重新输入，shape = buffer_step * env_batch * 1
            
            # TODO 所以我需要自己建立一个最后一个step是随机值的m_fea1_(计算adv的只会进行一次，我感觉可以！)（因为没有task可选，没有候选m节点特征了）
            machine_fea1_ = copy.deepcopy(machine_fea1)
            for i in range(machine_fea1.shape[0]): # step的个数
                if i == machine_fea1.shape[0]-1: # 表明是最后一个step
                    # machine_fea1_[i] = torch.rand((configs.env_batch, configs.n_machine, 6))  # bs*m*6, 01均匀分布的随机浮点数
                    machine_fea1_[i] = machine_fea1[i]  # 最后一个step，就用上一次一样的
                else:
                    machine_fea1_[i] = machine_fea1[i + 1]  # machine_fea1整体左移 = 去掉首位，补上一个最后位，总长度不变

            multi_v_ = self.step_for_net_out_Critic_GAT(
                        net_model=self.global_critic,
                        task_fea=tasks_fea_,
                        graph_pool_avg=graph_pool_avg,
                        adj=adj_,
                        candidate=candidate_,
                        machine_fea1=machine_fea1_, # TODO 一个很关键的问题，m_fea1没有最后一次的特征向量（因为最后一次的mask都屏蔽掉了，没有选择的action；这里是用来输出value，得有这个东西）
                        machine_fea2=machine_fea2_,
                        )  # 按照step重新输入，shape = buffer_step * env_batch * 1

            """-------------------------以上是全局value的采集！  +  j_v_和m_v_已经在外部采样并且记录在buffer里边！！-------------------------"""
            """
            local_adv_list： 大的list，包含4个reward的对应的adv，adv=(step * batch)
                4个reward的顺序 =  [mk_r, pt_r, tt_r, it_r]
            """
            local_adv_list = self.cal_local_job_machine_reward_GAE(
                        mk_r=mk_r,
                        pt_r=pt_r,
                        tt_r=tt_r,
                        it_r=it_r,
                        jv=job_v,
                        jv_=job_v_,
                        mv=machine_v,
                        mv_=machine_v_,
                        done_operation=done)
            job_adv_mk = local_adv_list[0]  # buffer_step * env_batch
            job_adv_it = local_adv_list[3]
            mac_adv_pt = local_adv_list[1]
            mac_adv_tt = local_adv_list[2]

            job_v_target_mk = job_adv_mk + job_v[:,:,0]  # 按照自身critic的输出值计算的adv=targetV-outV，然后产生自身的v_target = buffer_step * env_batch
            job_v_target_it = job_adv_it + job_v[:,:,1]
            machine_v_target_pt = mac_adv_pt + machine_v[:, :, 0]  
            machine_v_target_tt = mac_adv_tt + machine_v[:, :, 1]  # TODO [:,:,1]会直接3维变2维，不用.squeeze(-1) + [:,:,1:2]还是保留3维度，所以需要.squeeze(-1)

            # [mk_r, pt_r, tt_r, it_r] = 4个（buffer_step * env_batch）的元素
            
            """全局critic的gae的计算方式！"""
            adv_list = self.separate_cal_4_reward_GAE(
                mk_r=mk_r, 
                pt_r=pt_r, 
                tt_r=tt_r, 
                it_r=it_r,
                v=multi_v,
                v_=multi_v_,
                done_operation=done)  # TODO   （分别已norm！）4指标的分开计算adv_list，shape = step * env_bs
            
            # TODO 和下列统一：adv要是分开保存，v_target可以用一个list来存对应的4个指标的,方便不用改变量就能计算！
            adv_mk = adv_list[0]  # buffer_step * env_batch
            adv_pt = adv_list[1]
            adv_tt = adv_list[2]
            adv_it = adv_list[3]
            global_v_target_list = [adv_list[i] + multi_v[:, :, i:(i + 1)].squeeze(-1) for i in range(4)] # 优势函数A + 当前时刻的V = 预估值呀！ 32* 16 (step * batch)--（buffer_step * env_batch）
            Logger.log("Training/buffer_done/adv_targetV", f"Calculate the Local/Global Adv and TargetV based on GAE", print_true=1)

    
        loss_dict = {"job_actor_loss": [], "machine_actor_loss": [], "global_critic_loss": []}  # 记录k_epoch训练次数中的每一次的loss(求均值)
        grad_norm_dict = {"a_machine": [], "a_operation": [], "v": []}  # 记录k_epoch训练次数中的每一次的loss（求均值）
        # Optimize policy for K epochs:
        for i_K in range(args['K_epochs']):  # 就是ppo的网络更新多少次！    

            actor_loss_lst, actor_loss_lst_operation, critic_loss_lst = [], [], []  # 注意清0，否则累加！！！记录batch中minibatch的每一次的loss
            a_grad_norm_lst, a_grad_norm_lst_operation, v_grad_norm_lst = [], [], []  # 记录batch中minibatch的每一次记录梯度裁剪所返回的缩放因子

            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            # TODO： 如果要换成不是随机采样的minibatch，采用SequentialSampler！！！（实验完了就换回来，省的忘了）
            for index in BatchSampler(SubsetRandomSampler(range(tasks_fea.shape[0])),   # buffer里边的总的step的个数！
                                      mini_bs,  # mini_bs = args['n_job'] * args['n_machine']， step = mini_bs * args['buffer_size'] （本质就是while一次的step）
                                      False):  # 从buffer_step中随机抽取mini_bs个样本，随机,且遍历且不会重复！

                """-----------------重采样joint_actions，进行更新；critic网络因为单独网络，loss不想加进来----------------------"""

                """
                actor网络新生成的动作概率
                state = buffer_step * env_batch * xxxxxx
                index = [2, 5, 8, 1, 10, 15, 0, 7, 4, 13, 9, 6, 12, 3, 14, 11] 16个元素的索引
                s[index] = torch.Size([16, 16, 3, 4, 4])  16 * env_batch * xxx ---（mini_batch_size * env_batch * xxx ）  
                        所以可以用self.step_for_net_out来遍历s[index]一个个输出对应all action的概率
                """
                """因为牵扯到要是用中间状态全图特征，这里重新while True（for循环也可）一下！注意传入的是打乱之后state！！！"""

                # print("888:{}".format(get_GPU_usage()[1]))
                
                h_mch_pooled = None  # 初始化的时候m的全图特征=none，然后都用实时更新特征，不用保存
                o_prob_lst = []  # 记录输出的可选task节点的概率！
                m_prob_lst = []  # 记录输出的可选machine节点的概率！
                lst_job_v = []  # 记录重新采样的自身的value值
                lst_machine_v = []
                for i in range(tasks_fea[index].shape[0]):  # 求取step的个数: mini_batch的个数

                    """
                    重采样
                        用的buffer里边的数据，state+mask+candidate+task_index都是已知的(一一配对的)，buffer里边的轨迹数据重新采样
                        k_epoch第一次，新采样的网络参数一样，计算loss并更新
                        k_epoch第二次，网络参数已改变，重新采样，和buffer里边的旧轨迹值进行计算loss，并更新
                        迭代上述过程，完成重要性采样！
                        
                    需要循环重新跑mini_bs的step数量，用来采集新采样的数据。BUG 感觉可以直接全扔进去的
                    mini_bs（task）是从buffer_size（task*buffer_size）里边随机选取的，轨迹顺序已经打乱了！
                    """
                    _, _, _, prob, h_g_o_pooled, job_v = self.job_actor(
                                x_fea=tasks_fea[index][i],
                                graph_pool_avg=graph_pool_avg,  # 全局只有一个
                                padded_nei=None,
                                adj=adj[index][i],
                                candidate=candidate[index][i],
                                h_g_m_pooled=h_mch_pooled,# 这个m节点的全图池化均值,现在对应于选择的m的action
                                mask_operation=mask_operation[index][i],
                                use_greedy=False)
                    o_prob_lst.append(prob)  # 输出action的概率：mini_batch_size * env_batch * job
                    lst_job_v.append(job_v)  # 输出自身的value：mini_batch_size * env_batch * 2

                    """-----------------Machibe Actor网络重新采样----------------------"""
                    # print("666:{}".format(get_GPU_usage()[1]))
                    """
                    对应：选择o的时候用m的全图，选择m的时候用o的全图
                    # 相当于一个step一个step的进行输出，mac_mask: s[index] = torch.Size([minibatch, env_bs, 1, m])
                    """
                    mch_prob, h_mch_pooled, mac_v = self.machine_actor_gcn(
                                machine_fea_1=machine_fea1[index][i],  # 从ReplayBuffer里边查到的！
                                machine_fea_2=machine_fea2[index][i],
                                h_pooled_o=h_g_o_pooled,
                                machine_mask=mask_machine_batch_[index][i])  
                    m_prob_lst.append(mch_prob)  # mini_batch * env_batch * m
                    lst_machine_v.append(mac_v)  # 输出自身的value：mini_batch_size * env_batch * 2

                # 沿着第一个维度堆叠，list[tensor_1, tensor_2, ..., tensor_32]变成一个大tensor
                j_action_prob = torch.stack(o_prob_lst,dim=0)  # mini_batch_size * env_batch * job
                m_action_prob = torch.stack(m_prob_lst, dim=0)
                # Logger.log("Training/buffer_done/important_sample_a_prob", f"j_action_prob={j_action_prob.shape}, m_action_prob={m_action_prob.shape}", print_true=1)
                
                """TODO 1113-重采样---新增的job和machine的本地critic的重采样"""
                lst_job_v = torch.stack(lst_job_v, dim=0) # 0mk + 1it   value = Minibatch * env_bs * 2
                lst_machine_v = torch.stack(lst_machine_v, dim=0) # 0pt + 1tt
                # Logger.log("Training/buffer_done/important_sample_local_v", f"lst_job_v={lst_job_v.shape}, lst_machine_v={lst_machine_v.shape}", print_true=1)
                
                """----------------------先计算job_actor的loss，再计算machine_actor的loss-----------------------------------------"""
                # print("8989:{}".format(get_GPU_usage()[1]))
                
                """buffer里边的action在新采样离散分布里边的对数概率"""
                j_dist_now = Categorical(probs=j_action_prob)  # 离散分布 j_dist_now = mini_batch_size * env_batch * job = Categorical(probs: torch.Size([16, 16, 4])) = j_action_prob
                j_a_logprob_now = j_dist_now.log_prob(a_operation[index])  # shape = (mini_batch_size * env_batch) = torch.Size([16, 16]) = a_operation = j_a_logprob_now
                
                m_dist_now = Categorical(probs=m_action_prob)  # net输出的概率分布的离散分布： m_dist_now = mini_batch_size * env_batch * machine = Categorical(probs: torch.Size([16, 16, 4])) = m_action_prob
                m_a_logprob_now = m_dist_now.log_prob(a_machine[index])  # net输出的所选动作的对数概率：shape = (mini_batch_size * env_batch) = torch.Size([16, 16])   = a_machine = a_machine_logprob
                
                # Logger.log("Training/buffer_done/important_sample_a_logprob", f"j_a_logprob_now={j_a_logprob_now.shape}, lst_machine_v={lst_machine_v.shape}", print_true=1)
                
                """
                1、trick的原版：采用的是之前单一网络没更新时的a_logprob作为旧网络的重要性采样的分母（不再进行old网络的重新采样，ESWA也是这样子的思路，只保存采集到的不变）
                2、ppo基本的思路：之前写的，更新的时候旧网络也要重新采集数据（因为是随机选取呀，结果会有差别?不会，只看网络输出的概率，输入一样，输出的概率也一样，相当于没变，用buffer里边的正好），然后新旧之比作为重要性采样的
                注意：a_logprob_operation[index].detach()  buffer里边的
                    之前采集的和现在采集的梯度都不一样，这里old的就用detach去掉梯度，只用数值大小
                """
                job_ratios = torch.exp(j_a_logprob_now - a_logprob_operation[index].detach())  # 具体a的对数概率的比值：shape(mini_batch_size X env_batch) 
                machine_ratios = torch.exp(m_a_logprob_now - a_machine_logprob[index].detach())  # shape(mini_batch_size X env_batch) 
                # Logger.log("Training/buffer_done/important_sample_ratios", f"job_ratios={job_ratios.shape}, machine_ratios={machine_ratios.shape}", print_true=1)  # 重要性采样比值：
 
                """TODO 采用4指标的分开的adv来计算policy的loss"""
                """---------------------根据全局value计算获得的adv，计算job actor的policy的loss---------------------"""
                # adv_xx = （buffer_step * env_batch）改为使用  adv_xx[index] = (mini_batch_size * env_batch)
                surr1_job_mk = job_ratios * adv_mk[index]  # 原版重要性采样, shape = mini_batch_size * env_batch
                surr2_job_mk = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_mk[index]

                surr1_job_pt = job_ratios * adv_pt[index]  
                surr2_job_pt = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_pt[index]

                surr1_job_tt = job_ratios * adv_tt[index]  
                surr2_job_tt = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_tt[index]

                surr1_job_it = job_ratios * adv_it[index]  
                surr2_job_it = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_it[index]

                loss_mk = torch.min(surr1_job_mk, surr2_job_mk)
                loss_pt = torch.min(surr1_job_pt, surr2_job_pt)
                loss_tt = torch.min(surr1_job_tt, surr2_job_tt)
                loss_it = torch.min(surr1_job_it, surr2_job_it)
                
                """
                按照加权比例计算多个指标的总loss
                # global_loss_job_actor = configs.weight_mk * loss_mk + configs.weight_ec * (loss_pt + loss_it) + configs.weight_tt * loss_tt
                
                需要按照随机权重来搞！
                此时的RW = shape = train_bs * env_bs * 3, 按照Minibatch的index进行提取，然后分别提取对应mk、ec和tt的具体某一列，然后降维squeeze
                    # w1={random_weight[index][:, :, 0].squeeze(-1)}
                    # w2={random_weight[index][:, :, 1].squeeze(-1)}
                    # w3={random_weight[index][:, :, 2].squeeze(-1)},  buffer_step * env_bs, 选取其中的index个，= mini_bs * env_bs
                """
                # ！TODO [:,:,2]会直接3维变2维，不用.squeeze(-1) + [:,:,1:2]还是保留3维度，所以需要.squeeze(-1)
                global_loss_job_actor = random_weight[index][:,:,0] * loss_mk \
                            + random_weight[index][:,:,1] * (loss_pt + loss_it) \
                            + random_weight[index][:,:,2] * loss_tt   
                # Logger.log("Training/buffer_done/random_weight_sum_job_loss_globalADV", f"global_loss_job_actor={global_loss_job_actor}, shape={global_loss_job_actor.shape}", print_true=1)  # 根据全局value函数计算出来的adv向量，来计算job_actor的loss，其中采用随机权重进行加权求和

                """根据job_actor的本地value，计算job_actor关于mk和it指标的loss"""
                surr1_job_mk_local = job_ratios * job_adv_mk[index]  # 按照job本地value的ADV计算对应mk和it指标的surr
                surr2_job_mk_local = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * job_adv_mk[index]
                
                surr1_job_it_local = job_ratios * job_adv_it[index]  # 
                surr2_job_it_local = torch.clamp(job_ratios, 1 - self.epsilon, 1 + self.epsilon) * job_adv_it[index]
                
                loss_mk_local = torch.min(surr1_job_mk_local, surr2_job_mk_local)
                loss_it_local = torch.min(surr1_job_it_local, surr2_job_it_local)
                
                # TODO 1113-也按照随机权重的比例来计算本地critic的job_loss，！
                local_loss_job_actor = random_weight[index][:, :, 0] * loss_mk_local \
                                + random_weight[index][:, :, 1] * loss_it_local   
                # Logger.log("Training/buffer_done/random_weight_sum_job_loss_localADV", f"local_loss_job_actor={local_loss_job_actor}, shape={local_loss_job_actor.shape}", print_true=1)  # 根据本地value函数计算出来的adv向量，来计算job_actor的loss，其中采用随机权重进行加权求和
                
                """---------------------根据全局value计算获得的adv，计算machine actor的policy的loss---------------------"""
                # adv_xx = （buffer_step * env_batch）改为 adv_xx[index] = (mini_batch_size X env_batch)
                surr1_machine_mk = machine_ratios * adv_mk[index]  # 原版重要性采样
                surr2_machine_mk = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_mk[index] 
                
                surr1_machine_pt = machine_ratios * adv_pt[index]  
                surr2_machine_pt = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_pt[index]

                surr1_machine_tt = machine_ratios * adv_tt[index]  
                surr2_machine_tt = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_tt[index]

                surr1_machine_it = machine_ratios * adv_it[index]  
                surr2_machine_it = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_it[index]

                loss_mk_m = torch.min(surr1_machine_mk, surr2_machine_mk)  # (mini_batch_size X env_batch)
                loss_pt_m = torch.min(surr1_machine_pt, surr2_machine_pt)
                loss_tt_m = torch.min(surr1_machine_tt, surr2_machine_tt)
                loss_it_m = torch.min(surr1_machine_it, surr2_machine_it)

                """
                global_loss_machine = configs.weight_mk * loss_mk_m + configs.weight_ec * (loss_pt_m + loss_it_m) + configs.weight_tt * loss_tt_m
                
                需要按照随机权重来搞！
                此时的RW = shape = train_bs * env_bs * 3, 按照Minibatch的index进行提取，然后分别提取对应mk、ec和tt的具体某一列，然后降维squeeze
                    # w1={random_weight[index][:, :, 0].squeeze(-1)}
                    # w2={random_weight[index][:, :, 1].squeeze(-1)}
                    # w3={random_weight[index][:, :, 2].squeeze(-1)},  buffer_step * env_bs, 选取其中的index个，= mini_bs * env_bs
                """
                global_loss_machine = random_weight[index][:,:,0] * loss_mk_m \
                                + random_weight[index][:,:,1] * (loss_pt_m + loss_it_m) \
                                + random_weight[index][:,:,2] * loss_tt_m
                # Logger.log("Training/buffer_done/random_weight_sum_machine_loss_globalADV", f"global_loss_machine={global_loss_machine}, shape={global_loss_machine.shape}", print_true=1)  # 根据全局value函数计算出来的adv向量，来计算machine_actor的loss，其中采用随机权重进行加权求和

                """根据machine_actor的本地value，计算machine_actor关于pt和tt指标的loss"""
                surr1_pt_machine_local = machine_ratios * mac_adv_pt[index]  # 按照machine本地value的ADV计算对应pt和tt指标的surr
                surr2_pt_machine_local = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * mac_adv_pt[index]
                
                surr1_tt_machine_local = machine_ratios * mac_adv_tt[index]  
                surr2_tt_machine_local = torch.clamp(machine_ratios, 1 - self.epsilon, 1 + self.epsilon) * mac_adv_tt[index]
                
                loss_pt_m_local = torch.min(surr1_pt_machine_local, surr2_pt_machine_local)
                loss_tt_m_local = torch.min(surr1_tt_machine_local, surr2_tt_machine_local)
                
                # TODO 1113-也按照随机权重的比例来搞self的critic，或者就是自身就是mk+it的使用！
                local_loss_machine = random_weight[index][:, :, 1] * loss_pt_m_local  \
                                    + random_weight[index][:, :, 2] * loss_tt_m_local   
                # Logger.log("Training/buffer_done/random_weight_sum_machine_loss_localADV", f"local_loss_machine={local_loss_machine}, shape={local_loss_machine.shape}", print_true=1)  # 根据本地value函数计算出来的adv向量，来计算machine_actor的loss，其中采用随机权重进行加权求和                   

                """---------------------计算job和machine actor的policy的entropy---------------------"""
                """
                策略熵的loss = sum: prob * logprob   (最后去个mean，把env_batch个环境的取平均)
                .entropy会自动返回一个已经sum的prob*logprob！！！
                """
                job_dist_entropy = j_dist_now.entropy()  # shape(mini_batch_size * env_batch)
                machine_dist_entropy = m_dist_now.entropy()  # shape(mini_batch_size * env_batch)
                # Logger.log("Training/buffer_done/entropy", f"job_dist_entropy={job_dist_entropy.shape}, machine_dist_entropy={machine_dist_entropy.shape}", print_true=1)  # 根据离散分布计算策略熵

                """-------------------------------分别计算job和machine网络的本地critic的MSE的loss------------------------------"""
                w_mk = random_weight[index][:, :, 0]  # shape = mini_batch_step * env_bs  (因为加了[index],不然就是buffer_batch*env_bs)  
                w_ec = random_weight[index][:, :, 1]
                w_tt = random_weight[index][:, :, 2]

                """
                v_target = buffer_step * env_batch
                v_target[index] = （mini_batch_size * env_batch）
                job和machine的本地critic的重采样: lst_job_v = lst_machine_v = mini_batch_size * env_batch * 2
                """
                job_critic_loss_mk = self.global_critic_loss_func(w_mk * job_v_target_mk[index],
                                                              w_mk * lst_job_v[:, :, 0:1].squeeze(-1))  
                job_critic_loss_it = self.global_critic_loss_func(w_ec * job_v_target_it[index],
                                                              w_ec * lst_job_v[:, :, 1:2].squeeze(-1))  
                machine_critic_loss_pt = self.global_critic_loss_func(w_ec * machine_v_target_pt[index],
                                                              w_ec * lst_machine_v[:, :, 0:1].squeeze(-1))  
                machine_critic_loss_tt = self.global_critic_loss_func(w_tt * machine_v_target_tt[index],
                                                              w_tt * lst_machine_v[:, :, 1:2].squeeze(-1))  # MSE之后是标量

                job_critic_loss = job_critic_loss_mk + job_critic_loss_it  # TODO 注意对应好自定义顺序mk+pt+tt+it！！！
                machine_critic_loss = machine_critic_loss_pt + machine_critic_loss_tt 
                # Logger.log("Training/buffer_done/local_critic_MSEloss", f"job_critic_loss={job_critic_loss.shape}, machine_critic_loss={machine_critic_loss.shape}", print_true=1)  # 根据离散分布计算策略熵

                """------------------------------------计算job和machine网络的总LOSS，并更新网络--------------------------------------"""
                # TODO 0108-消融实验 = 只用全局critic和自身的重要性采样
                # job_actor_loss = -2 * global_loss_job_actor - self.ENTROPY_BETA * job_dist_entropy  # shape(mini_batch_size X env_batch)
                
                """job_actor_loss = -2 * 全局actor + -1 * 局部actor + 0.5*局部critic(MSE后标量，会自动广播拓展) - deta * entropy"""
                # 自身的loss之和，shape(mini_batch_size X env_batch)
                job_actor_loss = -2 * global_loss_job_actor + (-1) * local_loss_job_actor + 0.5 * job_critic_loss - self.ENTROPY_BETA * job_dist_entropy 
                
                # TODO 0108-消融实验 = 只用全局critic和自身的重要性采样
                # machine_actor_loss = -2 * loss_machine - self.ENTROPY_BETA * dist_entropy  # shape(mini_batch_size X env_batch)
                
                """machine_actor_loss = -2* 全局actor + -1 * 局部actor + 0.5*局部critic(MSE后标量，会自动广播拓展) - deta * entropy"""
                # 自身的loss之和，shape(mini_batch_size X env_batch)
                machine_actor_loss = -2 * global_loss_machine + (-1) * local_loss_machine + 0.5 * machine_critic_loss - self.ENTROPY_BETA * machine_dist_entropy  
                
                Logger.log("Training/buffer_done/actor_loss", f"job_actor_loss={job_actor_loss.shape}, machine_actor_loss={machine_actor_loss.shape}", print_true=1)  # 两个job和machine的网络的loss
                
                
                """
                # Update actor： aloss和entropy分别mean再相加，和相加之后再mean是一样的！（没区别）
                # 梯度范数：L2欧几里得范数，向量元素平方和的开方，表示梯度的大小；先计算每个参数的梯度，然后stack成一个大tensor，再求合并后大tensor的L2范数得到total_norm
                # 梯度裁剪return：总范数total_norm，越大，缩放因子越小，原始梯度被裁剪缩放的很多（按照CLIP阈值进行缩放的缩放因子：范数超阈值，原始梯度按比例缩放，得到裁剪后的梯度）: total_norm * 缩放因子 = max_norm最大范数CLIP_GRAD （因子越大）
                # 缩放因子来调整学习率！（max_norm越小，裁剪的梯度越大，得到的梯度就越小，防止梯度爆炸的效果越明显。）
                """
        
                self.job_actor_optimizer.zero_grad()
                """现在我网络一起重采样，然后有连续两个loss需要backward，下一个backward需要用到中间变量的计算图！！！不能默认free了！！！！！"""
                """job_actor_loss.mean().backward(retain_graph=True)  # 加上策略熵之后再求均值？"""
                if args['use_grad_clip']:  # Trick 7: Gradient clip
                    a_grad_norm_operation = torch.nn.utils.clip_grad_norm_(self.job_actor.parameters(),
                                                                           args['CLIP_GRAD'])  # 梯度裁剪，默认是0.5
                    a_grad_norm_lst_operation.append(a_grad_norm_operation)  # 保存minibatch的每一次的梯度缩放因子

                self.machine_actor_optimizer_gcn.zero_grad()
                """machine_actor_loss.mean().backward(retain_graph=True)  # 加上策略熵之后再求均值？"""
                if args['use_grad_clip']:  # Trick 7: Gradient clip
                    a_grad_norm = torch.nn.utils.clip_grad_norm_(self.machine_actor_gcn.parameters(),
                                                                 args['CLIP_GRAD'])  # 梯度裁剪，默认是0.5
                    a_grad_norm_lst.append(a_grad_norm)  # 保存minibatch的每一次的梯度缩放因子

                # TODO 既然叫做JointActor，那么loss加到一起再更新不是更符合？
                """试试loss相加再更新？之前分开更新，需要第一个loss保存计算图，现在加在一起，那就不用保存了，也能训练，笑死！"""
                loss = job_actor_loss.mean() + machine_actor_loss.mean()
                loss.backward()

                self.job_actor_optimizer.step()
                # print("999:{}".format(get_GPU_usage()[1]))
                self.machine_actor_optimizer_gcn.step()
                # print("777:{}".format(get_GPU_usage()[1]))
                
                Logger.log("Training/buffer_done/update_actor_net", f"loss = job_actor_loss.mean() + machine_actor_loss.mean() = {loss}", print_true=1) # 总loss进行梯度计算
                

                """-----------------更新global critic网络----------------------"""
                # TODO 按照buffer里边的随机选取的index个元素，进行全局critic的重新采样
                multi_v_s = self.step_for_net_out_Critic_GAT(
                            net_model=self.global_critic,
                            task_fea=tasks_fea[index],
                            graph_pool_avg=graph_pool_avg,
                            adj=adj[index],
                            candidate=candidate[index],
                            machine_fea1=machine_fea1[index],
                            machine_fea2=machine_fea2[index]
                            )  # 按照step重新输入，shape = buffer_step * env_batch * 4
                            
                """
                TODO 新增随机权重的critic的更新方式！！！！因为MSE之后是标量，所以要分别在MSE中的两个变量中乘以权重！！！（先计算W*V，加权和成一个）
                
                需要按照随机权重来搞！
                此时的RW = shape = train_bs * env_bs * 3, 按照Minibatch的index进行提取，然后分别提取对应mk、ec和tt的具体某一列，然后降维squeeze
                    # w1={random_weight[index][:, :, 0].squeeze(-1)}
                    # w2={random_weight[index][:, :, 1].squeeze(-1)}
                    # w3={random_weight[index][:, :, 2].squeeze(-1)},  buffer_step * env_bs, 选取其中的index个，= mini_bs * env_bs
                    
                v_target[index] = v_target（buffer_step * env_batch）--（mini_batch_size * env_batch）
                """
                w_mk = random_weight[index][:,:,0] # shape = minibatch * env_bs
                w_ec = random_weight[index][:,:,1]
                w_tt = random_weight[index][:,:,2]

                # global_v_target_list = [mk_r, pt_r, tt_r, it_r] = 4个（buffer_step * env_batch）的元素
                critic_loss_mk = self.global_critic_loss_func(w_mk * global_v_target_list[0][index],
                                                                w_mk * multi_v_s[:, :, 0:1].squeeze(-1))  # （mini_batch_size * env_batch）
                critic_loss_pt = self.global_critic_loss_func(w_ec * global_v_target_list[1][index],
                                                                w_ec * multi_v_s[:, :, 1:2].squeeze(-1))  
                critic_loss_tt = self.global_critic_loss_func(w_tt * global_v_target_list[2][index],
                                                                w_tt * multi_v_s[:, :, 2:3].squeeze(-1))  
                critic_loss_it = self.global_critic_loss_func(w_ec * global_v_target_list[3][index],
                                                                w_ec * multi_v_s[:, :, 3:4].squeeze(-1))  # TODO MSE之后就是标量了！！！！

                critic_loss = critic_loss_mk + critic_loss_pt + critic_loss_it + critic_loss_tt  # TODO 注意对应好自定义顺序mk+pt+tt+it！！！

                # Update critic
                self.global_critic_optimizer.zero_grad()
                critic_loss.backward()  # 多次计算需要保存计算图为true（选择action和计算adv无梯度计算，就不用保存计算图！）
                
                if args['use_grad_clip']:  # Trick 7: Gradient clip
                    v_grad_norm = torch.nn.utils.clip_grad_norm_(self.global_critic.parameters(),
                                                                 args['CLIP_GRAD'])  # 梯度裁剪，默认是0.5
                    v_grad_norm_lst.append(v_grad_norm)  # 保存minibatch的每一次的梯度缩放因子
                self.global_critic_optimizer.step()
                
                # print("1111:{}".format(get_GPU_usage()[1]))
                
                Logger.log("Training/buffer_done/update_global_critic_net", f"critic_loss={critic_loss}", print_true=1) # 全局critic_loss进行梯度计算
                

                """K_epoch=5遍更新，每次更新需要buffer_step/mini_bs_step=buffer_size轮才能选完buffer，每一轮都会计算一次loss，记录，然后更新网络！"""
                actor_loss_lst_operation.append(job_actor_loss.mean())  # 记录actor的loss，记得求解mean之后再记录！！！
                actor_loss_lst.append(machine_actor_loss.mean())  # 记录actor的loss，记得求解mean之后再记录！！！
                critic_loss_lst.append(critic_loss)  # 记录critic的loss，每次update的时候都会重新记录（TODO MSE后直接标量！）
                
                
                """查错误：打印出反向传播的梯度为NaN的模块名称和参数."""
                # for name, param in self.job_actor.named_parameters():
                #     if param.grad is not None and torch.isnan(param.grad).any():
                #         print("nan gradient found")
                #         print("name:", name)
                #         print("param:", param.grad)
                #         raise SystemExit
                # for name, param in self.machine_actor_gcn.named_parameters():
                #     if param.grad is not None and torch.isnan(param.grad).any():
                #         print("nan gradient found")
                #         print("name:", name)
                #         print("param:", param.grad)
                #         raise SystemExit
                # for name, param in self.global_critic.named_parameters():
                #     if param.grad is not None and torch.isnan(param.grad).any():
                #         print("nan gradient found")
                #         print("name:", name)
                #         print("param:", param.grad)
                #         raise SystemExit

                

            # 计算下整个buffer_step都训练完的平均loss,视为1次epoch,dim=0第一维度求均值（计算每一列的均值）
            loss_dict["machine_actor_loss"].append(torch.mean(torch.stack(actor_loss_lst), dim=0))  # 整个buffer_step都训练完，求均值，作为单个epoch的loss
            loss_dict["job_actor_loss"].append(torch.mean(torch.stack(actor_loss_lst_operation), dim=0))  # 先list中tensor堆叠成大tensor，再求mean，也是张量;
            loss_dict["global_critic_loss"].append(torch.mean(torch.stack(critic_loss_lst), dim=0))  
            
            Logger.log("Training/buffer_done/update_1_epoch_done", f"--------Update K_epoch: {i_K+1}/{args['K_epochs']}, ReplayBuffer: every {mini_bs} in {range(tasks_fea.shape[0])},  job_actor_loss={torch.mean(torch.stack(actor_loss_lst_operation), dim=0)}, machine_actor_loss={torch.mean(torch.stack(actor_loss_lst), dim=0)}, global_critic_loss={torch.mean(torch.stack(critic_loss_lst), dim=0)}--------", print_true=1) # log
            
            # 将张量移动到CPU设备，TODO 一次epoch，梯度裁剪相关系数，250425暂时不用
            a_grad_norm_lst_cpu = [tensor.cpu() for tensor in a_grad_norm_lst]
            a_grad_norm_lst_operation_cpu = [tensor.cpu() for tensor in a_grad_norm_lst_operation]
            v_grad_norm_lst_cpu = [tensor.cpu() for tensor in v_grad_norm_lst]
            grad_norm_dict["a_machine"].append(torch.mean(torch.stack(a_grad_norm_lst_cpu), dim=0))  # 先list中tensor堆叠成大tensor，再求mean，也是张量;
            grad_norm_dict["a_operation"].append(torch.mean(torch.stack(a_grad_norm_lst_operation_cpu), dim=0))  
            grad_norm_dict["v"].append(torch.mean(torch.stack(v_grad_norm_lst_cpu), dim=0))  
        
        """记录所有epoch训练之后的平均loss（每个epoch，有buffer_step个样本，分开minibatch个进行随机训练）"""
        train_loss_j_a = torch.mean(torch.stack(loss_dict['job_actor_loss']), dim=0).detach().cpu().numpy()   #所有epoch训练完，取均值作为当前train(update)的训练结果
        train_loss_m_a = torch.mean(torch.stack(loss_dict['machine_actor_loss']), dim=0).detach().cpu().numpy()
        train_loss_c_a = torch.mean(torch.stack(loss_dict['global_critic_loss']), dim=0).detach().cpu().numpy()
        Logger.log("Training/buffer_done/update_all_epoch_done", f"Mean: job_actor_loss={train_loss_j_a}, machine_actor_loss={train_loss_m_a}, global_critic_loss={train_loss_c_a}", print_true=1) # update函数调用一次，反馈的loss
        
        """250425-新增一个Logger，用来记录loss，用于wandb的输出 + loss随着episode发生变化，buffer满了就训练5次，采集5个轨迹就满了，训练每次随机挑mini_bs组成一个轨迹，buffer中5个轨迹用完，算作训练1次；5次epoch训练完，相当于这一轮update结束！记录的loss的5次训练均值，直接喂给wandb，用log的dict形式（DT中episode=iteration，Kepoch=num_steps训练次数） + 验证集是每过10个episode用100组数据来验证，相应的obj直接记录，然后喂给wandb！"""
        # episode里边，buffer_size倍task开始训练，训练kepoch次，每次采用mini_bs=task完成所有buffer_step（即更新网络参数buffer_size次）
        Result_Logger.log_not_str(f"Training/Update/job_actor_loss", train_loss_j_a)  
        Result_Logger.log_not_str(f"Training/Update/machine_actor_loss", train_loss_m_a)
        Result_Logger.log_not_str(f"Training/Update/global_critic_loss", train_loss_c_a)
        
        if args['use_lr_decay']:  # Trick 6:learning rate Decay
            # 线性衰减
            # 固定步数*系数衰减（update一次，衰减一次）
            self.job_actor_lr_decay.step()  # Actor的LR衰减，
            self.machine_actor_lr_decay_gcn.step()  # Actor的LR衰减，
            self.global_critic_lr_decay.step()  # Critic的LR衰减

            for p in self.job_actor_optimizer.param_groups:  # 可以访问优化器参数组的列表。通常情况下，我们只使用一个参数组
                lr_job = p["lr"]  # 获取update之后的lr
            for p in self.machine_actor_optimizer_gcn.param_groups:  # 可以访问优化器参数组的列表。通常情况下，我们只使用一个参数组
                lr_machine = p["lr"]  # 获取update之后的lr
            for p in self.global_critic_optimizer.param_groups:  # 可以访问优化器参数组的列表。通常情况下，我们只使用一个参数组
                lr_critic = p["lr"]
            Logger.log("Training/buffer_done/update_lr_decay", f"lr_job={lr_job}, lr_machine={lr_machine}, lr_critic={lr_critic}", print_true=1) # 获取update之后的lr
        
        loss_mean_lst = [train_loss_j_a, train_loss_m_a, train_loss_c_a]  # 所有kepoch训练完的mean
        loss_std_lst = [torch.std(torch.stack(loss_dict['job_actor_loss']), dim=0).detach().cpu().numpy(),
                        torch.std(torch.stack(loss_dict['machine_actor_loss']), dim=0).detach().cpu().numpy(),
                        torch.std(torch.stack(loss_dict['global_critic_loss']), dim=0).detach().cpu().numpy()]
        
        # return loss_dict, grad_norm_dict  # 返回k_epoch个元素的loss字典 + 总梯度范数字典
        return loss_mean_lst, loss_std_lst  # 返回k_epoch次训练的loss的mean和std，转为数组！ 
    
    # 全局变量清零函数
    def set_to_0(self, env):
        """
        A2C：如果连续运行，这些都需要请0！！！！！！！！！！！！！！！！！！！
        全局变量，记录某些随step变化的值
        """
        self.chosen_taskID_list = []  # 最终选择的task的id    TODO eval的时候使用
        self.chosen_taskID_list_batch = [[] for _ in range(self.batch_size)]  # ! TODO 所有batch中所选择的task的id的, 按照step顺序在对应bs的list中进行append 原chosen_action_list_batch
  
        self.pool_task_dict = {}  # 用作更新的任务池，可选择task还有哪些？   TODO eval的时候使用
        for i in range(self.n_job):
            self.pool_task_dict[i] = self.pool_task_list[i]
        
        self.pool_task_dict_batch = []  # 用作更新的任务池的batch版本，可选择task还有哪些？
        for _ in range(self.batch_size):
            task_dict = {}  # 初始化字典，用来添加到列表!一定要在这里重新初始化！！！！！！！！！！！！！！！！！！！！！！！！！相当于新建内存！否则改一个dict，其他dict全变了！
            for i in range(self.n_job):
                task_dict[i] = self.pool_task_list[i]
            self.pool_task_dict_batch.append(task_dict)  # # 记录batch_size个的各个job的可以选择的task的id = candidate的候选task的id


        self.remaining_m = {}  # 创建一个字典作为每个job的剩余任务数量，用作掩码，排除不能选择的job（即行数）  TODO eval的时候使用
        for i in range(self.n_job):
            self.remaining_m[i] = self.n_machine  # machine默认就是每个job的子任务数量
        
        self.remaining_m_batch = []
        for _ in range(self.batch_size):
            remain_m = {}  # 初始化字典，用来添加到列表!一定要在这里重新初始化！！！！！！！！！！！！！！！！！！！！！！！！！相当于新建内存！
            for j in range(self.n_job):
                remain_m[j] = self.n_machine  # machine默认就是每个job的子任务数量
            # remain_m = {0:4,1:4,2:4,3:4}
            self.remaining_m_batch.append(remain_m)  # # 记录batch_size个的各个job的剩余task个数

        lst = [0.0] * self.n_job  # ! TODO job_mask就是屏蔽当前可以选择的job（一列一列来选择，保证紧凑，所有指标都好！不然就是我旧方法，完全随机选择！）
        self.mask_new = torch.tensor(lst).cuda()  # job选择时的mask机制（第一列完全随机，后续按照candidate最小的先选）     TODO eval的时候使用
        
        self.mask_new_batch = []
        for _ in range(self.batch_size):  #! TODO job_mask就是屏蔽当前可以选择的job（一列一列来选择，保证紧凑，所有指标都好！不然就是我旧方法，完全随机选择！）
            lst1 = [0.0] * self.n_job
            self.mask_new_batch.append(lst1)  # 用于和batch版本的logit进行相加，注意转成tensor
        self.mask_new_batch = torch.tensor(self.mask_new_batch).cuda()  # 转成tensor用于和net中的logit相加， torch.tensor([0.0, 0.0],[0.0, 0.0],...)
        
        
        
        


            
    