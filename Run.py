import pickle
import csv
import math
import psutil
import os
import json
import torch
import argparse
import copy
import random
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL')  # 将a文件夹的路径添加到Python路径,0表示添加到最前边
from algorithm.ppo_algorithm import PPOAlgorithm
from instance.generate_allsize_mofjsp_dataset import Instance_Dataset, Logger, Result_Logger
from trainer.parallel_env import Parallel_env
from trainer.replaybuffer import ReplayBuffer
from model.gcn_mlp import g_pool_cal, aggr_obs
from algorithm.agent_func import select_machine_action
from trainer.validate import validate_cost_gcn_jointActor_GAT, read_MIP_result_from_csv
from trainer.fig_kpi import plot_show
from parameters import args
import wandb
import heapq  # 最小堆，堆中从小到大进行排序，插入会自动保持堆的顺序





def experiment(
        variant,   # 传入的是arg定义好的参数，dict形式
): 
    
    """加载生成的数据"""
    pth = '/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/instance'
    dataset_path = pth + f"/Instance_J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}.pkl"
    train_instance = Instance_Dataset(  # TODO def的所有形参都默认值，这里传入可以只选择要改的值！
            generate_true=0,    # TODO generate_true=0直接读取文件的地址，不需要其他参数
            dataset_pth=dataset_path)  
    eval_dataset_path = pth + f"/eval_Instance_J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}.pkl"
    eval_instance = Instance_Dataset(  # TODO def的所有形参都默认值，这里传入可以只选择要改的值！
            generate_true=0,    # TODO generate_true=0直接读取文件的地址，不需要其他参数
            dataset_pth=eval_dataset_path)
    
    """
    将样本数据按照env_batch进行拆分，并自动转成tensor张量 !!!!!!!!
    1、drop_last: bool = False, 默认不会舍弃最后不够整除的数据！
    2、num_workers：用于数据加载的子进程数，默认为 0（即单线程）。 高了可能可以加快数据读取
    3、shuffle = None, true是会每一个epoch都重新洗牌, 总的样本数据不会变化，就是数据顺序打乱了，选的每次bs都不一样（有用的trick，训练有奇效？）
    
    DataLoader按照你写的__getitem__函数和batch_size进行数据返回！你写的都是返回一个样本的数据，bs=10，那么一次就是10个样本！
    每env_batch个样本作为一组数据，for batch_idx, batch in enumerate(data_loader):进行提取
        自定义：返回是一个ability_dict，其中有4个key（t,p,tranT,edge）,每个key中元素是一个大的tensor：shape = env_batch * task*m（m*m）（edge_num * m/edge_num(均分)）
    """
    train_loader = DataLoader(train_instance, 
                               batch_size=variant['env_batch'], 
                               shuffle=True, 
                               num_workers=4)  # 传入numpy会自动转成tensor张量！
    
    specific_name = 'No_LR_decay+Add_eval()'  # ！TODO 250508-PPO代码里边的316和418的return改成了原始的mask！
    # specific_name = '_'
    group_name = f"Instance_J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}" + f"-Weight_{variant['weight_mk']}{variant['weight_ec']}{variant['weight_tt']}" + f"-BS_{variant['env_batch']}-Seed_{variant['train_seed']}"   # 便于区分实验场景
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'   # 生成一个介于100,000到999,999之间随机整数，作为独立易于识别的实验标识符
    if variant['log_to_wandb']:  # wandb的初始化，就可以在网站看到统计结果
        wandb.init(
            name=exp_prefix + specific_name,
            # name=exp_prefix,
            group=group_name,
            project='E2E-MAPPO_for_MT-FJSP',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug
        
    
    ppo = PPOAlgorithm(args=variant, load_pretrained=False)
    
    
    """
    是否加载模型
     "load_pth": {
        "Exist_jme": [[6,6,2],[10,6,2],[20,6,3],[10,10,2],[15,10,2],[20,10,5]],
        "Exist_epi": [2840,2840,2840,2840,2840,2840]
      },
    """
    model_pth = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/trained_model/"
    job_name = "PPO_job_actor_"
    machine_name = "PPO_machine_actor_"
    critic_name = "PPO_global_critic_"
    
    size_index = 0  # 确定当前是哪一个场景
    Exist_j = variant['load_pth']['Exist_jme'][size_index][0]  # 之前的小场景训练好的
    Exist_m = variant['load_pth']['Exist_jme'][size_index][1] 
    Exist_e = variant['load_pth']['Exist_jme'][size_index][2] 
    Exist_epi = variant['load_pth']['Exist_epi'][size_index]  # TODO 250429-完全跑完有top123，否则有f'_EP{i_episode + 1}_.pth'的前三小
    """# 0=best/top1,1=final,2=error选episode,3=top2/3"""
    if variant['load_model_type'] == 0:
        job_f = job_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + "_top1.pth"    # 选择保存的已存在model的文件名称
        machine_f = machine_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + "_top1.pth" 
        critic_f = critic_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + "_top1.pth"
    elif variant['load_model_type'] == 1:
        job_f = job_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_final.pth'
        machine_f = machine_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_final.pth'
        critic_f = critic_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_final.pth'
    elif variant['load_model_type'] == 2:
        job_f = job_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + f'_EP{Exist_epi}_.pth'
        machine_f = machine_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + f'_EP{Exist_epi}_.pth'
        critic_f = critic_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + f'_EP{Exist_epi}_.pth'
    elif variant['load_model_type'] == 3:
        job_f = job_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_top2.pth'
        machine_f = machine_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_top2.pth'
        critic_f = critic_name + f"J{Exist_j}M{Exist_m}E{Exist_e}" + '_top2.pth'

    # 使用 os.path.join() 函数拼接路径
    path = variant['trained_model_pth']  # pth模型路径
    job_model = os.path.join(model_pth, job_f)
    mch_model = os.path.join(model_pth, machine_f)
    critic_model = os.path.join(model_pth, critic_f)
    if os.path.exists(job_model) and variant['use_load_model'] == True:   # s TODO 加载旧模型，可以采用课程学习的思路！！！
        print('-------------------------------------------load the TRAINNED model--------------------------------------------')
        ppo.job_actor.load_state_dict(torch.load(job_model))
        ppo.machine_actor_gcn.load_state_dict(torch.load(mch_model))  # 按照路径地址加载预训练好的模型
        ppo.global_critic.load_state_dict(torch.load(critic_model))  # 按照路径地址加载预训练好的模型
        
        # ppo_test.actor_model.load_state_dict(torch.load(fix_ability_file))
        Logger.log("Init/load_model", f"ppo.job_actor={ppo.job_actor.state_dict()}, ppo.machine_actor_gcn={ppo.machine_actor_gcn.state_dict()}, ppo.global_critic={ppo.global_critic.state_dict()}", print_true=1) # 打印出加载后的模型中的所有参数
    
    
    """
    并行环境的初始化：
    1、自动采样Batch_size个ability instance
    2、若放在每个episode里边，每次init变量都会初始化，下文各个变量需要重新赋值才行！！！
    """
    paral_env = Parallel_env(args=variant)  # 采样batch_size次，生成batch_size个env
    
    """
    经验池的缓存：
    会有step是因为我每一step都在记录
    state_m = [m+t+e1] + fx  step * batch * channel * j * m 
    action_prob:  step * batch * m
    reward: step * batch 
    """
    replay_buffer = ReplayBuffer(args=variant)  # 初始化缓存：初始化要记录的变量; 传入的configs就是parse的参数
    
    """
    初始化graph_pool_avg：
    1、graph_pool_avg 全图节点嵌入求均值的矩阵：batch，batch*task  （只用初始化一次，按jm场景即可，不用每次都循环）
    """
    graph_pool_avg = g_pool_cal(graph_pool_type=variant['neighbor_pooling_type'],  # average的type：1/n_nodes
                                batch_size=variant['env_batch'],
                                # batch_size`参数使用了`torch.Size`来指定批次的形状。根据代码中的参数设置，它是一个大小为`[batch_size, n_j * n_m, n_j * n_m]`的张量大小。这意味着每个批次中有`batch_size`个图，每个图具有`n_j * n_m`个节点。
                                n_nodes=variant['n_job'] * variant['n_machine'],  # task的个数
                                device=variant['device'])
    
    """
    MIP的对应值存在cost_dict的对应key的list中，按照样本123.。。100的顺序（用于eval求相对MIP的gap，便于展示收敛度）
    cost_dict = {
        "runtime": [],
        "best_objective": [],
        "Makespan": [],
        "MachineEC": [],
        "MachineIdleT": [],
        "TransEC": [],
    }
    """
    MIP_result_file = "/remote-home/iot_wangrongkai/RUN/MO-FJSP-DRL/MIP_result/MO_FJSP_MIP_result_(J%s_M%s_seed%s_sample%s_w%s%s%s).csv" % \
                      (variant['n_job'], variant['n_machine'], variant['eval_seed'], 100, 
                       int(variant['weight_mk'] * 10), int(variant['weight_ec'] * 10), int(variant['weight_tt'] * 10))
    MIP_cost_dict = read_MIP_result_from_csv(MIP_result_file)
    
    """训练用参数"""
    current_bs_idx = 0 # 每次采用不同bs的数据的指示
    all_steps = 0 # 记录所有episode中运行的step的个数
    
    loss_mean_lst = [] # 每一update的所有kepoch的loss均值 = list = 转为np = update * 3 
    loss_std_lst = [] # 每一update的所有kepoch的loss标准差 = list = 转为np = update * 3 
    lr_lst = [variant['LR']]  # 记录每一次update之后被衰减的LR（暂时是同一个）,初始值先放进去
    
    # ! TODO 以下变量查看变化没啥用啊，但至少有5轮episode用的相同的ability_instance并行环境，但是在重新run的时候，instance都没有固定（会shuffle！）
    infor_means_lst = []  # 记录每一step的oenv_step_info = [r, o_done, rmk, ridle, renergy_m, renergy_transM] 中的env_bs个对位元素的均值(step返回的不是张量)
    Final_4cost_lst = []  # done之后存下轨迹的最终的4个reward，转为list = episode*4  TODO != infor_means_lst中各个元素的累加值，因为r0初值不为0
    Obj_lst = []  #这里是更新的网络采集instance数据集产生的最终reward的加权和， 
    
    eval_cost_lst = []  # 记录验证集给出的-cost，同于画图 , 每次eavl的时候100个数据的均值，记录一下
    cost_best = float('inf')  # 寻找min的cost（正数数），即累加Reward，初值是inf，任意一个都比他小
    top3_obj_heap = []  # 使用最大堆来存储最小的三个obj_eval_mean,（存储记录pth的文件名）
    
    traj_lst =[]  # 储存轨迹 = episode_num，同一批batch走buffer_size遍，ppo不断更新进化中 (TODO list中多个tuple，内含元素shape不同)
    
    for i_episode in range(variant['episode_num']):
        """
        泛化性：
        1、尝试不断更改ability instance的数值：每20轮 + 20的batch_size
        2、最好做到数量级统一
        3、每一个task：加工时间t + 加工能耗（or 加工功率）e1。 m*m：运输时间e2。 分边情况edge

        首先是episode=0，随机样本！然后19的时候已经玩了20次；那么下一次的episode=20要开始重新生成了！（//除法取整数）
        """
        if i_episode % variant['resample_freq'] == 0:  # 20次正好是下个循环的首位！前20是0-19
            
            """生成当前ability_instance的信息，提取每一个bs中的instance，存在一个大list中"""
            print('=' * 250)
            Logger.log("Training/resample_instance", f"--------------------Instance Sample：{(i_episode // variant['resample_freq'] + 1) * variant['env_batch']}/{train_instance.__len__()}---------------", print_true=1)

            for bs_idx, bs_data in enumerate(train_loader):
                if bs_idx == current_bs_idx:
                    instance_bs_dict = bs_data  # 当前的数据：env_batch个，tensor张量，存成字典形式的（dict形式的！！！！！）
            Logger.log("Training/current_instance", f"bs_idx={bs_idx}, instance_bs_dict['t'].shape={instance_bs_dict['t'].shape}", print_true=1)
            current_bs_idx += 1  # 下一轮重新选取env_bs个新的数据，所以指标+1，还是dict但是里边的元素都多了bs个维度
            
            paral_env.get_batch(instance_bs_dict)  # 采样Batch_size个，然后赋值self.ability_instance（不再清0和变化！） = batch_size个instance，每个instance有4元素 

            # TODO 初始化reward动态放缩的class(类中初始化RunningMeanStd类，其中self.n在整个episode跑完中都不会重新reset，反应更多样本数据)
            """
            相同场景的env是不会初始化self.n的，除非你换了env，那reward铁定不一样了啊！
            所以：我在更换样本数据这里进行动态计算mean和std的self.n的初始化 + 同时考虑env_bs的影响，同时初始化多个实例，针对不同来运行！！！！
            """
            paral_env.init_RewardScaling_sameBATCH(shape=4)  # TODO 指标有4! + 每次的episode只会重新reset这个RewardScaling类中的self.R（记录当前episode的累加误差）！！
            Logger.log("Training/resample_status", f"Now is paral_env.get_batch + paral_env.init_RewardScaling_sameBATCH", print_true=1)
    
        """---------------------------------------初始化task节点的状态 + 被调度machine节点的状态--------------------------------------------"""
        """
        初始化S0：
        1、adj =（batch，tasks，tasks）
        2、tasks_fea = 12元素《预估ST + 预估FT + 预估PT + if被调度I + 入边的个数in_dedge_n + 分配给m_id + m的t能力 +m的p能力 + 归属作业j_id + 固定随机权重*3》  =（env_batch*tasks， 12）
        3、machine_fea = 8元素《被调度FT_last_task + 被调度sumPT/task + 被调度sumTransT + 被调度sumIdleT + 同一m的被选次数的累加sumIm + 固定随机权重*3》= env_batch * m * 8 （被调度后）
        4、在ENV的reset里边随机初始化RandomWeight：不同bs不一样 + 且整个episode可以不变，除非又重新reset了一下！
        """
        adj_batch, machine_scheduled_fea_batch, tasks_fea_batch = paral_env.init_DGFJSPEnv_state0()  # 初始化DGFJSPEnv的并行环境！TODO self.ability_instance（不再清0和变化！）还是同一批数据，跑5轮episode之后进行换数据
        # Logger.log("Training/init_states_0", f"Shape: adj_batch={adj_batch.shape}, tasks_fea_batch={tasks_fea_batch.shape}, machine_scheduled_fea_batch={machine_scheduled_fea_batch.shape}", print_true=1)
        
        
        """------------------------初始化候选节点index矩阵 + task节点的mask矩阵（0可用） + machine节点的mask矩阵(0可用)------------------------------"""
        """
        初始化候选列表 + task的mask：batch = env_batch
        1、candidate：batch * job (注意：需要记录的是可选的task的索引值，为了方便gather提取特征！)
        2、mask：batch * job (注意：原有的是float01的张量，这里转成false和true的张量，mask_value=1来赋值！！)
        """
        value_list = []
        for dict in ppo.pool_task_dict_batch:  # 遍历list中的所有dict，储存的是每一batch中可选的task的id（每个job的任务池，可做task的id，1开始）
            value_list.append(list(dict.values()))  # 将列表中的字典的值转换为列表
        candidate_batch = np.array(value_list) - 1  # list转np.array，TODO 然后taskid-1转成对应的索引index {1：task1=idx0，2：task7=idx6，3：...}*bs
        # Logger.log("Training/init_candidate", f"candidate_batch={candidate_batch}, shape={candidate_batch.shape}", print_true=1) # = batch * job

        mask_operation_batch = ppo.mask_new_batch.bool()  # 01float张量转成true和false TODO 初始全0 = torch.tensor([0.0, 0.0, 0.0, 0.0]) * bs
        # Logger.log("Training/init_mask_operation", f"mask_operation_batch={mask_operation_batch}, shape={mask_operation_batch.shape}", print_true=1)  # = batch * job
        
        '''
        生成用于machine的mask，
        按照instance_t的方式直接找minus就是不能干的！
        没写在init_DGFJSPEnv_state0中，便于进行控制变量法和消融实验
        1、当次bs的数据来源：paral_env.ability_instance, 其中元素：env_bs * [t+p+transT+edge]
            注意：instance_bs_dict字典，其中【“t”】是bs*task*m的张量！！！
        2、machine_mask的需求： shape= bs * tasks * m
        3、我定义t和p都是对应的负数，所以用一个判断就ok！t shape = env_bs*task*m
        4、数据是np.array，已被转成numpy

        注意：我的parallel是并行env的处理，按照env_batch进行逻辑运算！！！！！
        转成矩阵，然后再转成tensor，真多此一举
        '''
        # 将矩阵中的元素大于等于0的设置为True，小于0的设置为False
        mask_machine_batch = instance_bs_dict["t"].numpy() >= 0  # t能力矩阵转为bool，bs * tasks * m
        mask_machine_batch = torch.tensor(mask_machine_batch).to(variant['device'])
        # TODO 将布尔张量取反
        mask_machine_batch0 = ~mask_machine_batch  # mask作用是，true代表不能选
        # Logger.log("Training/init_mask_machine", f"mask_machine_batch0={mask_machine_batch0}, shape={mask_machine_batch0.shape}", print_true=1)  # = bs * tasks * m
        
        """
        win_done：解决最后只剩一个job的0001学习问题：只计算，但是不训练到net中
        """
        # win_dones = np.full((n_total_task, Batch_size), False)  # 选m没有win_done，按照step步数和batch的大小初始化
        # win_dones = torch.FloatTensor(win_dones).to(device)  # 转到gpu运行,转成float之后变成0.和1.

        """---------------------------------------初始化m节点的全图特征 + 动态reward放缩系数的初始化-----------------------------------------------"""
        
        """
        在while之前初始化，只赋值一次！为了保证进入到选择operation的网络时是None的，用可学习的参数；后续都是实际的m的全图特征
        注意：ESWA互相传入的两个全图特征向量，在update的时候并没有进行memory；更新时还是按照新生成的特征向量进行互相学习！！！！！
        """
        h_mch_pooled = None  # m节点的全图特征，防止初始化第一步缺少参数

        #  TODO reward动态放缩的reset，每个episode都重新搞下
        for i in range(len(paral_env.paral_Rscaling_instance)): # 有几个env_bs，验证并行环境中是否有错
            paral_env.paral_Rscaling_instance[i].reset()  # TODO 每个episode都要初始化reward的缩放

        # 开始JSP问题：选择方向
        step_flag_for_v_ = 0 # ！ TODO 1113-新增为了记录job和machine的v_，计数
        
        """--------------------------------------循环采样轨迹-----------------------------------------------"""
        while True:
            with torch.no_grad():
                
                """
                np.array的数据，然后在job_actor转成张量 + GPU
            
                调用ppo.job_actor中的forward函数
                    x_fea,  # task节点的特征向量：batch*task，4
                    graph_pool_avg,  # 全图节点嵌入求均值的矩阵：batch，batch*task
                    padded_nei, # max的pooling才有用
                    adj,  # 输入：env_batch * tasks * tasks。 会在学习前进行转换： 邻居矩阵（带权重+j到i+自身置1）batch*task，batch*task
                    candidate,  # 可选task的id，batch，job
                    h_g_m_pooled,# 传入每一step的m节点的全图节点嵌入均值：batch，hidden
                    mask_operation,  # 传入当前可选节点的mask位，某一job选完了，那就置为true：batch，job
                    use_greedy=False     

                return task_index, action_index, log_a, prob, h_g_o_pooled, job_v  # 全图的特征batch*hidden
                    # task_index = 选择tasks的具体index（从0开始!!）：一维，4元素，（batch）
                    # 动作job的索引张量action_index：一维，（batch），离散采样得到的动作index（job个概率的索引）, 还没有转化成task_id呢
                    # 动作对应的对数概率张量log_a：一维，（batch），采样动作的对应的离散概率  
                    # 每一个task的prob概率张量prob： 二维： shape = env_batch * job  每一个可选task的prob概率！
                    # task全图张量h_g_o_pooled = bs * hidden 
                    # job网络的本地value值张量job_v = batch*2

                采集数据，选择action，只需要数值进行loss计算，没必要计算梯度
                """
                task_index, action_index, log_a, _, h_g_o_pooled, job_v = ppo.job_actor(
                            x_fea=tasks_fea_batch,   # TODO task节点的最新的12特征
                            graph_pool_avg=graph_pool_avg,  # 全局只有一个，值=1/task
                            padded_nei=None,
                            adj=adj_batch,
                            candidate=candidate_batch,
                            h_g_m_pooled=h_mch_pooled, # 这个m节点的全图池化（均值） 
                            mask_operation=mask_operation_batch,
                            use_greedy=False
                            )
                # Logger.log("Training/while/ppo_job_actor", f"task_index={task_index.shape}, action_index={action_index.shape}, log_a={log_a.shape}, h_g_o_pooled={h_g_o_pooled.shape}, job_v={job_v.shape}", print_true=1)

                """
                选择执行某一task，对应的machine的能力已知，作为machine的mask_machine_batch
                1、每一bs样本在训练的时候，初始化env的state0的时候也按照ability_instance生成对应的bs_task_m的T/F矩阵，转为张量和上GPU
                2、mask_machine_batch0 = bs_task_m | task_index(从0开始)=（batch，） 16元素一维张量，两次扩维再复制至batch_1_m 
                3、从所有bs样本0（每个epi更新）中选取对应bs中action选中的索引，找到对应的machine_mask！！！ (mask_machine_batch_ = shape = bs_1_m)
                """
                # 每一批的样本都不会对mask有影响，只有job选择不同的task，gather之后的mask不一样，传入到machine网络中！
                mask_machine_batch_ = torch.gather(mask_machine_batch0,
                                                   1,  # 维度2
                                                   task_index.unsqueeze(-1).unsqueeze(-1).expand(mask_machine_batch0.size(0), -1,mask_machine_batch0.size(2)))  
                # Logger.log("Training/while/machine_mask", f"mask_machine_batch_={mask_machine_batch_}, shape={mask_machine_batch_.shape}", print_true=1)  # 4 * 1 * 4 = batch * 1 * m

                """
                TODO 新增当前task的对应可选m的节点特征 
                1、候选m的节点特征（bs，m，6）= 做当前task的<能力t, 能力pt, 所需transt, 可选Im, 能力p, 属于edge>(有则真，无则估计mean)
                2、(it可以深拷贝当前的env，然后遍历所有可选的m，反馈一个新增it，然后添加到对应位置，可以但没必要，不是直观看的，且每个step都遍历，跑死了！)
                3、m_fea1 = candidate + m_fea2 = shceduled
                """
                """ESWA和逻辑上，应该拿这个m_fea1来输入网络，输出action的"""
                machine_candidate_fea_batch = paral_env.cal_cur_task_machine_feature(  # return = bs * m * 6
                            task_index=task_index,  # batch,
                            m_mask=mask_machine_batch_,   # batch * 1 * m
                            all_task_fea=tasks_fea_batch)  # （env_batch*tasks， 12）
                # Logger.log("Training/while/machine_candidate_fea", f"machine_candidate_fea_batch={machine_candidate_fea_batch}, shape={machine_candidate_fea_batch.shape}", print_true=1)  #
                
                """
                TODO m_fea1（job_actor后产生） 和 m_fea2（env后产生） 都要输入到machine网络中，进行GAT自聚合
                
                return mch_prob, h_pooled, machine_v  
                    的选择m的概率 = batch*m 
                    所有m节点的全局嵌入（求均值）= batch * hidden_dim 
                    machine的本地value = bs*2
                
                对应：选择o的时候用m的全图，选择m的时候用o的全图
                """
                mch_prob, h_mch_pooled, machine_v = ppo.machine_actor_gcn(
                            machine_fea_1=machine_candidate_fea_batch,  # bs*m*6  对应task的m特征
                            machine_fea_2=machine_scheduled_fea_batch,  # bs*m*8  上一次选完m后的ENV的更新m特征
                            h_pooled_o=h_g_o_pooled,   # task全图张量h_g_o_pooled = bs * hidden 
                            machine_mask=mask_machine_batch_)  # batch * 1 * m machine的mask
                # Logger.log("Training/while/ppo_machine_actor_out", f"mch_prob={mch_prob.shape}, h_mch_pooled={h_mch_pooled.shape}, machine_v={machine_v.shape}", print_true=1)

                # print("3:{}".format(get_GPU_usage()[1]))
                
                """
                1、按照prob进行machine动作的选取，返回tensor（batch个元素
                2、返回所选动作的索引 + 对应的log_prob： tensor（batch个元素）
                """
                m_action, m_action_logprob = select_machine_action(mch_prob)  # 传入概率分布mch_prob = batch*m
                # Logger.log("Training/while/ppo_machine_actor", f"m_action={m_action.shape}, m_action_logprob={m_action_logprob.shape}", print_true=1)

            """------------------------至此，完成当前state到job和machine的动作输出 + 相应计算loss的参数---------------------------------"""
            """
            判断是否只剩下一个job，防止学习到了0001
                记录在win_dones中, 都是np.array的形式 
                unsqueeze(0) 在第一个维度增加，16---1*16
                win_done开始是False和True的张量，转成01的浮点张量
            """
            # # 判断每一行中只有一个0
            # win_done = torch.sum(ppo.mask_new_batch == 0, dim=1) == 1  # 判断batch个env是否每一个都到了win_done
            # win_done = win_done.unsqueeze(0)  # 2维变1维度，16---1*16
            # # win_dones = np.concatenate((win_dones, copy.deepcopy(win_done)), axis=0) # 将矩阵b合并到矩阵a中,b沿着垂直方向维度0（行）
            # win_dones = torch.cat((win_dones, copy.deepcopy(win_done)), dim=0)  # 将张量b合并到张量a中,维度0方向：16*16+1*16=17*16（step，batch）
            # # print("----test: win_dones = ", win_dones, win_dones.shape)
            
            # tensor张量用size，array数组用shape，list列表用len（看大小）

            """
            DGEnv改成单步：所以传入并行环境中的是[task_index, m_idx]: 从0开始，并行环境个当前step的action。注意：
            1、因为是并行环境env_batch：所以生成action都是一个list，需要合并成每个元素都是一组[o,m]对
            2、chosen_action_list_batch = [[env1:123...],[env2:1234...],...] (暂时不用，修改对应的DGFJSPEnv_paral_step) = 所选择的task_id，从1开始的
                env_batch个列表，每个列表都是当前env的已选择的task的id
            3、machine：m_action = tensor([1, 3, 2, 3, 3, 0, 2, 3, 3, 1, 2, 0, 0, 0, 0, 0])，torch.Size([16]) 对应batch=16，当前的
            4、operation：task_index = tasks的具体index（从0开始!!）：一维，4元素，（batch）
            5、adj_batch_ = env_batch * tasks * tasks (np.array)
            6、tasks_fea_batch_ = (env_batch*tasks) * 12 (np.array)
            7、machine_scheduled_fea_batch_ =  bs*m*8 (np.array)
            7、oenv_step_info = [r, o_done, rmk, ridle, renergy_m, renergy_transM] * batch_size 
                TODO oenv_step_info后4个指标，都是当前step = prev-curr，然后是没有放缩的版本（完全是按照samples数据！）
                
            TODO 并行处理step的函数，对每一个step返回的即时的4个r指标，进行动态放缩 + 相同样本运行越多，放缩效果越好
            """
            # 都是env_batch个！返回的是一个list，zip进行元素配对，每个元素是元组（a，b），分别来自task_index和m_action
            joint_actions = [x for x in zip(task_index.tolist(), m_action.tolist())]  
            adj_batch_, oenv_step_info, machine_scheduled_fea_batch_, tasks_fea_batch_ = paral_env.DGFJSPEnv_paral_step(joint_actions)
            # Logger.log("Training/while/ppo_parallel_env_step", f"also output list oenv_step_info={np.array(oenv_step_info).shape}", print_true=1)
            
            """
            TODO 1223-更新ESA方法的mask = 按照一列一列选完才能进行，然后每次当前列的min才能选，min相等那就可以按照概率随机选！
                    先选action再更新剩余m，更新ppo.mask_new_batch
                    
            选择了task，对应的转为task_id，判断剩余task个数，判断mask的思路没有变啊！
            paral_env是并行环境类
            action_index是动作job的索引张量，一维，（batch），离散采样得到的动作index（job个概率的索引）, 还没有转化成task_id呢
            
            Return:
            1、candidate：batch * job (注意：需要记录的是可选的task的索引值，为了方便gather提取特征！)
            2、mask：batch * job (注意：原有的是float01的张量，这里转成false和true的张量，mask_value=1来赋值！！)
            """
            candidate_batch_, mask_operation_batch_ = ppo.esa_update_chosenTaskID_CandidateTaskIDx_JobMask(
                        paralenv=paral_env, 
                        action_batch=action_index,
                        mask_value=variant['mask_value'])  # 更新剩余可选任务（每行=每个job）

            """
            整形+记录reward:  [batch个元素]，每个step都在记录，done之后算是一个轨迹trajectory
                1、o_r = env_batch个元素：[x,x,x,x,....]
                TODO 2、分别记录step返回的4个指标的reward=prev-curr，每个step每个指标：[env1,env2,env3,env4,...] 这样子存进来！(动态缩放之后的结果)
            判断env_batch的done
            """
            o_r = [copy.deepcopy(info[0]) for info in oenv_step_info]  # 提取每一step的即时r = 首位0对应的是权重加权和之后的即时r
            mk = [copy.deepcopy(info[2]) for info in oenv_step_info]  # 位2对应的是finish time
            it = [copy.deepcopy(info[3]) for info in oenv_step_info]  # 位3对应的是idle time
            pt = [copy.deepcopy(info[4]) for info in oenv_step_info]  # 位4对应的是p*t
            tt = [copy.deepcopy(info[5]) for info in oenv_step_info]  # 位5对应的是transT
            check_done_operation = [info[1] for info in oenv_step_info]  # 提取每一个batch的done的元素组成新的list
            # Logger.log("Training/while/ppo_step_reward_done", f"check_done_operation={check_done_operation}, o_r = {o_r}", print_true=1)
            
            """
            记录每一step的下一时刻的v_, 用于loss计算
                env的step之后，保证有value产生，然后step>=2再进行计数，保证记录的就是v_
                # TODO 1113-新增记录局部的value，要连续记录，因为有些变量是交替产生的！
            """
            step_flag_for_v_ += 1
            if step_flag_for_v_ > 1:
                replay_buffer.store_v_next(j_v_=job_v,  # ！ TODO 从计数2开始记录，那么记录的就是v_ + 还缺少最后一次的v_记录
                                           m_v_=machine_v)
            if all(check_done_operation):  # 检查所有batch的m_done是否为True, TODO 需要在此基础上再跑一下！最后一步走完，env输出的state，下一时刻的value是多少
                with torch.no_grad():
                    _, _, _, _, h_g_o_pooled_, job_v_ = ppo.job_actor(
                                x_fea=tasks_fea_batch_,  # TODO task节点的最新的9特征
                                graph_pool_avg=graph_pool_avg,# 全局只有一个
                                padded_nei=None,
                                adj=adj_batch_,
                                candidate=candidate_batch_, # TODO done之后，candidate = 最后一列的task_index，还有值
                                h_g_m_pooled=h_mch_pooled,  # TODO 1113-接着上一个m网络产生全图特征，顺便求一下v_
                                mask_operation=mask_operation_batch,  # ！TODO done之后，这个mask全是TRUE，选不了，那么就是softmax全是0，不可能；换成用上一次的
                                use_greedy=False
                                )
                    machine_candidate_fea_batch_ = machine_candidate_fea_batch  # 就用最后一步的m_fea1，因为下一时刻没有task可以选择，不会有m_fea1
                    _, _, machine_v_ = ppo.machine_actor_gcn(
                                machine_fea_1=machine_candidate_fea_batch_, # bs*m*6  对应上一次task的m特征
                                machine_fea_2=machine_scheduled_fea_batch_, # bs*m*8  上一次选完m后的ENV的更新m特征
                                h_pooled_o=h_g_o_pooled_,   # 这里使用的是上一个job产生的全图的节点嵌入特征值
                                machine_mask=mask_machine_batch_)  # ！TODO 最后一步的时候已经没有task了，所以没法找到对应大于0的t，所以用最后一次的mask
                    replay_buffer.store_v_next(j_v_=job_v_,  # 最后一步其实已经记录过了，这是重新记录最后一次的下一次state的对应的v_
                                               m_v_=machine_v_)
                    step_flag_for_v_ = 0 # 此时可以清0了，因为马上也要done了！！！！

            """TODO 将ENV中的random_weight找出来然后转成 = shape = [env_bs, 3]"""
            rw = [copy.deepcopy(paral_env.paral_env_DG[i_bs].reward_random_weight) for i_bs in range(len(oenv_step_info))]  # env_bs个,遍历不同ENV中的随机权重值，是一个3元素的一维矩阵

            """
            ReplayBuffer的memory记录：
            1、只记录当前值，不是下一次的值: 当前step的s, a, a_logprob, r, s_, dw, done都存一下
            2、单次的存储：batch个env产生的数据

            每一step都会记录当前的state和此时下一时刻的state_, 对应是否done，反馈r
            count清0之后，每次都是新的记录重新覆盖，然后满足buffer_size之后，才会开始训练（相当于每次都是抛弃了buffer,所以不用清0）
            """
            replay_buffer.store_operation(
                        adj=adj_batch,  # self.count_operation += 1  # 记录缓存次数的变量在此！！！
                        fea=tasks_fea_batch,  # BUG：250425-为什么v8我用的是env.step之后的新fea = tasks_fea_batch_？？？？？？？？？？？？？？？
                        candidate=candidate_batch,
                        mask=mask_operation_batch,
                        a_o=action_index,
                        a_o_logprob=log_a,
                        r=o_r,
                        adj_=adj_batch_,
                        fea_=tasks_fea_batch_,
                        candidate_=candidate_batch_,
                        mask_=mask_operation_batch_,
                        mch_fea1=machine_candidate_fea_batch, # 当前task对应的m的可选节点特征
                        mch_fea2=machine_scheduled_fea_batch, # env产生的m的被选之后的state，从初值开始记录
                        mch_fea2_=machine_scheduled_fea_batch_,
                        a_m=m_action,
                        a_m_logprob=m_action_logprob,
                        dw=None,
                        done=check_done_operation,
                        mask_machine_=mask_machine_batch_,
                        mk=mk,
                        pt=pt,
                        tt=tt,
                        it=it,
                        rw=rw,
                        j_v=job_v,
                        m_v=machine_v)
            

            """
            判断是否开始训练
            """
            all_steps += 1  # 不间断记录step的次数，用于动态递减LR (虽然只记录了选择operation的step，但是选择machine和operation的次数是一样的)
            # Logger.log("Training/while/save_in_buffer", f"all_steps = {all_steps}, replay_buffer.store_operation", print_true=1)

            """
            When the number of transitions in buffer reaches batch_size, then update
            1、只有>=train_batch: 才会开始训练：每一轮episode = job * machine个计数
            2、最后如果不够train_batch，那就不要训练，因为buffer没有清空，而是覆盖（前边新网络采集的，后边还是旧的网络，不能训练！！）

            现在是选operation的次数够了（理论上machine的次数也一样）是要够train_batch，才会开始训练
            """
            if replay_buffer.count_operation == variant['n_job'] * variant['n_machine'] * variant['buffer_size']:  # buffer多大，所以维度就是多大  # state和action都存够了，然后就开始更新网络，不一定是done之后再更新
                
                tasks_n = variant['n_job'] * variant['n_machine']
                mini_batch_size = tasks_n  # 从经验池replayBuffer中取出数据进行训练的minibatch大小:tasks
                buffer_size = tasks_n * variant['buffer_size']  # task的倍数的大小
                
                Logger.log("Training/while/buffer_done/update", f"++++++++++++++++++++++++++++ ppo update {all_steps//buffer_size}/{variant['episode_num']*tasks_n//buffer_size} buffer_size={buffer_size}||mini_bs={mini_batch_size}||Date_freq_epi={variant['resample_freq']} ++++++++++++++++++++++++++++", print_true=1)

                # print("000:{}".format(get_GPU_usage()[1])) # update里边也没有显存的增加！
                
                """250429-储存轨迹，bs=16走buffer_size=5遍，然后ppo还不断更新"""
                traj_tuple = replay_buffer.numpy_to_tensor_operation()  # 存满了，直接取出来保存！ buffer_step * bs * xxx
                traj_tuple = tuple(tensor.cpu().numpy() for tensor in traj_tuple)# 将tensors转换为NumPy数组并存储为元组
                traj_lst.append(traj_tuple)  # 储存轨迹 = episode_num，同一批batch走buffer_size遍，ppo不断更新进化中 (TODO list中多个tuple，内含元素shape不同)
                
                """
                global_update版本都是用的固定的adv，（这是错的：网络在更新，v会一直变，怎么训练趋近？而不是采集一次就是一次不同的adv，紧跟实际；）！！！！
                包括那个old_a_logprob，我这边和ESWA一样都是固定了，不用old网络重新采集了  
                
                return
                loss_dict = {"job_actor_loss": [], "machine_actor_loss": [], "global_critic_loss": []}  # 记录k_epoch训练次数中的每一次的loss(求均值)
                """
                # TODO 1113-新增的全局+局部的critic方式，使每个agent的自己调整自己！ 
                #  # 返回k_epoch次训练的loss的mean和std，转为数组！ 
                loss_mean, loss_std = ppo.global_update_JointActions_GAT_selfCritic(  
                                replay_buffer=replay_buffer, 
                                all_steps=all_steps, # update的时候，会有lr的动态递减（线性衰减，和total_steps有关，更新多了，不用学那么快了）
                                graph_pool_avg=graph_pool_avg,  # 全局只有一个，用来求task节点的全图平均嵌入的矩阵
                                args=variant,
                                mini_bs=mini_batch_size) 
                                                                                                        
                # print("111:{}".format(get_GPU_usage()[1]))
                
                replay_buffer.count_operation = 0  # 清0，用来重新缓存
                replay_buffer.count_operation_ = 0  # 清0，用来重新缓存  TODO 1113-注意此时也要清0专门记录v_的指标

                """++++++++++++++++++++++++++++++++++
                              记录数据：loss：all Kepoch's mean and std
                +++++++++++++++++++++++++++++++++++++"""
                loss_mean_lst.append(loss_mean) # 每一update的所有kepoch的loss均值 = list = 转为np = update * 3 
                loss_std_lst.append(loss_std) # 每一update的所有kepoch的loss均值 = list = 转为np = update * 3 
                
                # 返回的是字典，包含k_epoch个元素，每一次训练的loss，都是tensor元素列表,列表合并，不是元素相加
                # a_grad_norm += gradnorm_dict_each_kepoch["a_machine"]  # 传回梯度裁剪缩放因子都是tensor元素列表，每次都是kepoch个！（训练了k_epoch次）
                # a_grad_norm_operation += gradnorm_dict_each_kepoch["a_operation"]  
                # v_grad_norm += gradnorm_dict_each_kepoch["v"]

                # update一次，就会按照总共走的total_step进行线性衰减lr（暂时a和v用的都是同一个lr）
                for p in ppo.job_actor_optimizer.param_groups:  # 可以访问优化器参数组的列表。通常情况下，我们只使用一个参数组
                    lr_lst.append(p["lr"])  # 获取update之后的lr

                # 更新完，更新一下GPU的使用情况（如果有其他的程序进来，显存也会增大）
                # now_GPU_info["id"], now_GPU_info["gpu_me"], _, _ = get_GPU_usage()
                # now_gpu_memory = now_GPU_info["gpu_me"] - begin_GPU_info["gpu_me"]

                Logger.log("Training/while/buffer_done/update_done", f"Update Done on episode {i_episode+1}/{variant['episode_num']}", print_true=1)  # update函数中已log每次update返回的k_epoch个loss的平均值
                print("++" * 250)

            """
            迭代更新state
            记录相关指标:
            1、oenv_step_info = env_batch_size * [r, o_done, rmk, ridle, renergy_m, renergy_transM]  (返回)
            2、对env_batch个即时r求均值，得到当前step的即时r，记录用以done之后计算cost；记得清0
            3、mask_machine_batch_  选完task才更新，不用重新赋值
            4、machine_candidate_fea_batch 选完task才更新，不用重新赋值
            """
            adj_batch = adj_batch_
            tasks_fea_batch = tasks_fea_batch_
            candidate_batch = candidate_batch_
            mask_operation_batch = mask_operation_batch_
            machine_scheduled_fea_batch = machine_scheduled_fea_batch_  # 迭代更新下一次的machine_scheduled_fea_batch_

            """++++++++++++++++++++++++++++++++++++++++++
                            记录数据：4实时reward(train)：env_bs求mean
            +++++++++++++++++++++++++++++++++++++++++++++"""
            # 记录的reward真实值进行对位mean并放缩, 每个episode的每个step都会有env_bs个[r, o_done, rmk, ridle, renergy_m, renergy_transM] 
            # 记录的是每一step的实时reward：prev-current, 计算每个位置所有元素的均值
            infor_means_step = np.array(oenv_step_info).mean(axis=0)  # bs * 6, 维度1进行求均值 = (6,)  |T/F视为1/0计算mean
            infor_means_lst.append(infor_means_step)  # 每一step都是全新的oenv_step_info
            
            """
            判断是否done：现在只看DG图的done就可以了
            1、reset
            2、画图的指标
            """
            if all(check_done_operation):  # 检查所有batch的m_done是否为True

                """
                至今为止：一定要在done之后reset之前读取：全选完的当前this step的cost，没有权重，不用累加，不用相减
                shape（正数） = env_bs个真实cost，需要除以env_bs求个平均，然后画出
                
                ! TODO 
                xxx_previous_step不会清0，reset的时候会变成理想预估初值，所以累加就不是=真实值，因为初始xxx_previous_step！=0 ！！！！
                这里done之后选这个变量，this_step就是计算的mk真值，pt累加，transT累加和idleT累加）
                
                env_bs个并行环境求mean：
                上次的makespan +  上次的加工能耗之和（已分配设备的）+ 至今为止的运输时间t （暂时没有乘以运输设备的e）
                """
                """++++++++++++++++++++++++++++++++++++++++++
                            记录数据：4最终reward(=objective)：env_bs求mean
                            Final_cost = np.array([mk_cost, pt_cost, transT_cost, idleT_cost])
                +++++++++++++++++++++++++++++++++++++++++++++"""
                costs = [sum(getattr(paral_env.paral_env_DG[i_bs], attr) for i_bs in range(variant['env_batch']))
                        for attr in ['makespan_previous_step', 'total_e1_previous_step', 'trans_t_previous_step', 'idle_t_previous_step']]
                costs[1] /= variant['n_job'] * variant['n_machine']  #  ! TODO 别忘了这里是平均能耗
                Final_4cost = np.array(costs) / variant['env_batch'] # ! TODO 当前episode的 = 真实cost对env_bs求平均
                Final_4cost_lst.append(Final_4cost)  # done之后存下轨迹的最终的4个reward，转为list = episode*4
                # !TODO 直接打印出4个平均env_bs之后指标的加权和！（不是偏差，直接平均真实值加权） = Cost/Objective in Paper!
                Objective = variant['weight_mk'] * Final_4cost[0] + variant['weight_ec'] * (Final_4cost[1] + Final_4cost[3]) \
                                         + variant['weight_tt'] * Final_4cost[2]
                Obj_lst.append(Objective)  # ! 这里是更新的网络采集instance数据集产生的最终reward的加权和，TODO 查看变化没啥用啊，至少有5轮episode用的相同的ability_instance并行环境，但是在重新run的时候，instance都没有固定（会shuffle！）
                
                print('**'*250)
                Logger.log("Training/while/done", f"Trajectory Done (step = tasks) episode {i_episode+1}/{variant['episode_num']} ", print_true=1)  # 
                
                """最后一个action，全是-15，判断不出来，但是也是属于win_done的范围"""
                # win_dones[-1, :] = 1.  # 最后一行都是1.
                # print("----test: win_dones = ", win_dones, win_dones.shape)

                """++++++++++++++++++++++++++++++++++++++++++
                        画出图像：render渲染
                +++++++++++++++++++++++++++++++++++++++++++++"""
                if i_episode == variant['episode_num'] - 1:  # 最后一次的训练
                    paral_env.paral_env_DG[-1].render()  # 只画出batch_size中的最后一个env的调度图 TODO 1202-先不画出gantt了！！！
                
                """
                变量清0，为下一episode做准备
                """
                ppo.set_to_0(None)  # ppo清0
                for i in range(paral_env.batch_size):
                    paral_env.paral_env_DG[i].reset()  # 各个并行的DGjspEnv都要reset一下
                paral_env.reset_data()  # 并行环境的reset和对应的m和o的环境reset
                
                print("**"*250)

                break  # 直接跳出 while循环！
        
        
        """
        开始eval：
        1、不在while循环中，单论episode
        """
        if ((i_episode + 1) % variant['eval_freq'] == 0) or (i_episode == variant['episode_num'] - 1):  # 每过validate_iter个episode，就进行validate

            print("------------------------------------------------Evaluation-------------------------------------------------------")
            eval_obj_lst = []  # 用于记录每一个测试数据的加权和obj
            eval_4_cost_lst = []  # 用于记录每一个测试数据的4指标的真实值，大lst，每个小lst包含 [mk, pt, transT, idleT]  
            
            gantt_flag = False   # 是否render渲染画图
            for i_eval in range(variant['eval_sample']):  # 固定100个样本数据（随即生成），计算r
                if i_eval == variant['eval_sample'] - 1:
                    gantt_flag = True   # 最后一个验证集，画图
                if i_eval >= 99:
                    gantt_flag = True
                    print('*'*250)
                    print("Validation {}/{}".format((i_eval + 1), variant['eval_sample']))
                    print('*'*250)
                
                """
                1、固定使用这100个数据！！！从0-99的index的变换
                2、采用untilNow = [mk, pt, transT, idleT]的最后done的step的指标全图的真实值！！！！
                3、采用idea_cost_refer = [mk, pt, transT, idleT]预估的当前样本下的理想值！！！！(不行，运输和空闲时间的idea都是0，不能除0；如果估计一个最差值，空闲和运输又会麻烦，本质上只需要一个同样本下的不同指标的固定值即可！按照样本求取的！) So，MIP求解器的值最简单好得！
                
                #记录当前轨迹的各指标累加cost，累加的，TODO ！= 真实值，初始的r0不是0
                eval_cost_dict_cumsum = {
                    "opr_Gt": 0,
                    "opr_mk": 0,
                    "opr_idleT": 0,
                    "opr_pt": 0,
                    "opr_transT": 0
                }
                真实值：eval_4_cost = [mkt, pt, transT, idleT]  # 轨迹走完的最终指标 = cost   +  obj
                """
                eval_cost_dict_cumsum, eval_4_cost, obj = validate_cost_gcn_jointActor_GAT(
                            ppo=ppo,
                            gantt_flag=gantt_flag,
                            data=eval_instance,  # 传入的是dataset，定义的可以直接.x读取数据
                            data_index=i_eval,
                            data_type=variant['eval_data_type'],  
                            greedy=True,
                            args=variant)  # 

                
                idea_cost_refer = [MIP_cost_dict["Makespan"][i_eval], MIP_cost_dict["MachineEC"][i_eval],
                                   MIP_cost_dict["TransEC"][i_eval], MIP_cost_dict["MachineIdleT"][i_eval]]  # TODO 别忘了这里是平均能耗
                new_related_cost = (np.array(eval_4_cost) - np.array(idea_cost_refer)) / np.array(idea_cost_refer)  # 差值为正数，越小越好！！！
                eval_obj_gap = variant['weight_mk'] * new_related_cost[0] + variant['weight_ec'] * (new_related_cost[1] + new_related_cost[3]) \
                                + variant['weight_tt'] * new_related_cost[2]
                
                eval_obj_lst.append(obj)  # 记录eval数据集中每一个instance的最终obj
                eval_4_cost_lst.append(eval_4_cost)  # 用于记录每一个测试数据的4指标的真实值，大lst，每个小lst包含 [mk, pt, transT, idleT] 
                
                if i_eval >= 99:  # 最后一个测试集
                    Logger.log("Evaluation/last_instance", f"Objective={obj}, eval_obj_gap={eval_obj_gap}, mk={eval_4_cost[0]}, pt={eval_4_cost[1]}, transT={eval_4_cost[2]}, idleT={eval_4_cost[3]}", print_true=1)  # eval最后一组数据的结果
                    
            # eval里边，需要所有的instance都跑完，取一个mean和std，进行保存，然后喂给wandb，随着eval_freq进行显示
            """存一次结果，就需要输入到wandb一次，然后重新赋值和清0，get_log会自动清lst的，放心"""
            obj_eval_mean = np.mean(np.array(eval_obj_lst), axis=0)  # lst转1维矩阵，100个元素，求一个平均值
            obj_eval_std = np.std(np.array(eval_obj_lst), axis=0)  # 求std
            eval_4cost_mean = np.mean(np.array(eval_4_cost_lst), axis=0)  # lst转2维矩阵，100*4，求一个平均值
            eval_4cost_std = np.std(np.array(eval_4_cost_lst), axis=0)  # lst转2维矩阵，100*4，求一个平均值
            
            Result_Logger.log_not_str(f"Evaluation/100instances/obj_eval_mean", obj_eval_mean)  
            Result_Logger.log_not_str(f"Evaluation/100instances/obj_eval_std", obj_eval_std)
            Result_Logger.log_not_str(f"Evaluation/100instances/mk_eval_mean", eval_4cost_mean[0]) # mkt
            Result_Logger.log_not_str(f"Evaluation/100instances/mk_eval_std", eval_4cost_std[0])
            Result_Logger.log_not_str(f"Evaluation/100instances/pt_eval_mean", eval_4cost_mean[1])  # pt
            Result_Logger.log_not_str(f"Evaluation/100instances/pt_eval_std", eval_4cost_std[1])
            Result_Logger.log_not_str(f"Evaluation/100instances/transT_eval_mean", eval_4cost_mean[2])  # transT
            Result_Logger.log_not_str(f"Evaluation/100instances/transT_eval_std", eval_4cost_std[2])
            Result_Logger.log_not_str(f"Evaluation/100instances/idleT_eval_mean", eval_4cost_mean[3]) # idleT
            Result_Logger.log_not_str(f"Evaluation/100instances/idleT_eval_std", eval_4cost_std[3])
            
            eval_cost_lst.append(obj_eval_mean)  # 记录验证集给出的-cost，同于画图 , 每次eavl的时候100个数据的均值，记录一下
            
            Logger.log("Evaluation/100instances/each_episode_result", f"----------episode={i_episode}, obj_100ins_mean={obj_eval_mean}, mk_100ins_mean={eval_4cost_mean[0]}, pt_100ins_mean={eval_4cost_mean[1]}, transT_100ins_mean={eval_4cost_mean[2]}, idleT_100ins_mean={eval_4cost_mean[3]}------------------------", print_true=1)  # eval最后一组数据的结果
            
            """++++++++++++++++++++++++++++++++++++++++++
                        保存数据到csv：100evalInstance的均值
            +++++++++++++++++++++++++++++++++++++++++++++"""
            save_pth = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/results/"
            result_file = save_pth + f"Obj_100_EvalInstance_J{variant['n_job']}_M{variant['n_machine']}_E{variant['n_edge']}_BS{variant['env_batch']}_Weight{int(variant['weight_mk'] * 10)}{int(variant['weight_ec'] * 10)}{int(variant['weight_tt'] * 10)}.csv" 
            # 将数据存储到CSV文件中; 每次写入都是覆盖的方式，最终展示的是跑完100次的结果
            with open(result_file, 'w', newline='') as csvfile1:  # `newline`参数设置为空字符串`''`，以确保写入CSV文件时不进行额外的换行操作
                writer = csv.writer(csvfile1)
                writer.writerows([eval_cost_lst])  # TODO 防止模型中途断裂，每一episode的eval的100个结果的均值都记录下来(需要再加一个[]，相当于2维列表，才能保存！)

            """++++++++++++++++++++++++++++++++++++++++++
                        保存模型参数：top3+final
            最小堆：  根节点2最小，且父节点（上）都小于等于子节点（下）
                  2
                 / \s
                3   5
               / \ / \s
              8  10 9 12         [2, 3, 5, 8, 10, 9, 12]
            +++++++++++++++++++++++++++++++++++++++++++++"""
            # 最前有定义
            # model_pth = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/trained_model/"
            # job_name = "PPO_job_actor_"
            # machine_name = "PPO_machine_actor_"
            # critic_name = "PPO_global_critic_"
            id_name = f"J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}"
            top_prex = ["_top3.pth", "_top2.pth", "_top1.pth"] 
            
            # !TODO 保存最后一次(eval的时候，保证有obj可看)的网络参数, 名字一样直接覆盖完事
            model1f = model_pth+job_name+id_name + "_final.pth"
            model2f = model_pth+machine_name+id_name + "_final.pth"
            model3f = model_pth+critic_name+id_name + "_final.pth"
            torch.save(ppo.job_actor.state_dict(), model1f)  # 
            torch.save(ppo.machine_actor_gcn.state_dict(), model2f)  # 记录policy，test的时候不需要critic
            torch.save(ppo.global_critic.state_dict(), model3f)  # 
            
            model1 = model_pth + job_name + id_name + f'_EP{i_episode + 1}_.pth'
            model2 = model_pth + machine_name + id_name + f'_EP{i_episode + 1}_.pth'
            model3 = model_pth + critic_name + id_name + f'_EP{i_episode + 1}_.pth'
            # 添加数字到最大堆
            heapq.heappush(top3_obj_heap, (-obj_eval_mean, model1, model2, model3))   # 添加负数，最小堆转为最大堆
            torch.save(ppo.job_actor.state_dict(), model1)  # 
            torch.save(ppo.machine_actor_gcn.state_dict(), model2)  # 记录policy，test的时候不需要critic
            torch.save(ppo.global_critic.state_dict(), model3)  # 
            
            # 如果堆的大小超过3，弹出最小的元素，即最大的数字
            if len(top3_obj_heap) > 3:
                _, m1_old, m2_old, m3_old = heapq.heappop(top3_obj_heap)  # 始终保持top3里边都是前三小的obj_mean, 返回弹出的元组。（最大堆都是按照元组中的第一个元素进行判断）
                os.remove(m1_old)   # 用于删除指定路径的文件
                os.remove(m2_old)   # 用于删除指定路径的文件
                os.remove(m3_old)   # 用于删除指定路径的文件
            
            if i_episode == variant['episode_num'] - 1:  # 最后一轮episode，也会eval，也会有obj和保存模型参数
                
                # 按照元组的第一个元素（数字）进行排序
                sorted_list = sorted(top3_obj_heap, key=lambda x: x[0])  # 按照元组的第一个元素排序，默认从小到大，lamda：参数和返回形式
                for i_tup in range(len(sorted_list)):  # 第一个就是最大的obj_mean（注意存的值是负数！）
                    # directory1, filename1 = os.path.split(tuples[1]) # 将文件路径分割成两部分：目录路径和文件名称。它返回一个元组，其中第一个元素是目录路径，第二个元素是文件名称。输出: Directory: /home/user/documents   # 输出: Filename: report.txt
                    # new_file_path = os.path.join(directory, new_filename)  #用于连接目录路径和文件名称，生成一个新的完整的文件路径。它确保路径在不同操作系统上正确地连接，因为不同操作系统路径的表示方式可能不同. New file path: /home/user/documents/report.txt
                    
                    model1 = model_pth + job_name + id_name + top_prex[i_tup]
                    model2 = model_pth + machine_name + id_name + top_prex[i_tup]
                    model3 = model_pth + critic_name + id_name + top_prex[i_tup]
                    # 使用os.rename()来重命名文件, 保证原文件存在
                    os.rename(sorted_list[i_tup][1], model1)  # list里边不同的元组，元组包含最小cost和3个网络的参数
                    os.rename(sorted_list[i_tup][2], model2)
                    os.rename(sorted_list[i_tup][3], model3)
                
                Logger.log("Evaluation/save model.pth", f"Final and TOP3 Best model parameters saved. Best_model={sorted_list[-1]}", print_true=1)  #  
            
        """
        验证完数据之后，保存此时的fig
        1、保存当前variant['eval_freq']这么多数据的训练效果 or 保存累积到现在的训练效果
        2、注意loss的个数 = eval_freq*K_epochs
        3、eval是直接把当前100次实验的效果都画出来
        """
        # loss_num = variant['eval_freq'] * variant['K_epochs']
        # plot_show(da_loss=da_loss[-loss_num:],
        #         da_loss_operation=da_loss_operation[-loss_num:],
        #         dv_loss=dv_loss[-loss_num:],
        #         rewards_sum=[],
        #         rewards_sum_operation=rewards_sum_operation[-variant['eval_freq']:],
        #         vali_cost_lst=vali_R_lst,  # 当前模型参数下的100次的eval的结果
        #         totalCost_t=totalCost_t[-variant['eval_freq']:],
        #         net1_totalCost_e=totalCost_energy_m[-variant['eval_freq']:],
        #         net1_totalCost_trans=totalCost_energy_transM[-variant['eval_freq']:],
        #         totalCost_idle=totalCost_idle[-variant['eval_freq']:],
        #         net1_totalCost_ratio=[],  # 利用率也没有了，没用m_env，不会有这些指标了！
        #         n_job=variant['n_job'],
        #         n_machine=variant['n_machine'],
        #         n_edge=variant['n_edge'],
        #         episode=i_episode,
        #         show_on=False)
        
            
        """一个episode存一次结果，就需要输入到wandb一次，然后重新赋值和清0，get_log会自动清lst的，放心"""
        pth = '/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/results/'
        result_f = pth + f"Loss_Cost_J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}.txt"
        logg = Result_Logger.get_logs(pth=result_f)  # TODO 返回所有保存的log，log是dict形式，储存的txt文件是"追加"模式，会不断很长 (队列会清0！)
        if variant['log_to_wandb']:
            wandb.log(logg)   # logg直接上wandb！循环输出log，log是一个dict存了多个参数，随着episode的次数来画图！
        # 注意：result_f因为调用的是get_logs，每个epi都会调用，会默认打印”===“，相当于episode的次数
            
    """从坏到好的轨迹，bs=16走buffer_size=5遍，保存结果到pickle文件"""
    pth1 = '/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/trajectory/'
    traj_f = pth1 + f"Trajectory_{variant['episode_num']}_J{variant['n_job']}M{variant['n_machine']}E{variant['n_edge']}_Seed{variant['train_seed']}_BS{variant['env_batch']}_Weight{int(variant['weight_mk'] * 10)}{int(variant['weight_ec'] * 10)}{int(variant['weight_tt'] * 10)}.pkl"
    with open(traj_f, 'wb') as f:
        pickle.dump(traj_lst, f)  # 储存轨迹 = episode_num，同一批batch走buffer_size遍，ppo不断更新进化中
    Logger.log("Training/All_episode/save_buffer_trajectory", f"save trajectory in {traj_f}.pkl'", print_true=1)

            

    
    
    
    
    
    
    
    

if __name__ == '__main__':
    
    experiment(variant=vars(args))  # vars内置函数，转为dict，表示参数和其值
    
    pth2 = '/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/'
    training_log_file = pth2 + f"/training_log_J{args.n_job}M{args.n_machine}E{args.n_edge}_BS{args.env_batch}_InsSeed{args.train_seed}_Weight{int(args.weight_mk*10)}{int(args.weight_mk*10)}{int(args.weight_mk*10)}.txt"
    Logger.get_logs(training_log_file)
    
    print('=' * 250)
    
