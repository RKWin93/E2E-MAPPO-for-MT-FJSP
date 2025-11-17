import csv
import os
import time
import numpy as np
import torch
import copy
import json
import argparse
import random

from algorithm.ppo_algorithm import PPOAlgorithm
from instance.generate_allsize_mofjsp_dataset import Instance_Dataset, Logger, Result_Logger
from trainer.validate import read_MIP_result_from_csv, validate_cost_gcn_jointActor_GAT, esa_validate_cost_gcn_jointActor_GAT
from tester.pdrs import run_Rules_jointActions_withMinus_1217
from trainer.fig_kpi import result_box_plot, plot_test_3d_cross_fig, result_box_plot_eachEpisode_1217
import wandb
from parameters import args



def PPO_test_jointActor_PDRs_MIP_withMinus_DeltaIndicator_ESA_1225(args,
                                                                   samples=100, 
                                                                   seed=3, 
                                                                   model_typy="top1",
                                                                   model_pth=None, 
                                                                   result_pth=None,
                                                                   mip_pth=None,
                                                                   MLWKR_type="max", 
                                                                   greedy=True):

    
    specific_name = '_'
    group_name = f"Testing_J{args['n_job']}M{args['n_machine']}E{args['n_edge']}" + f"-Weight_{args['weight_mk']}{args['weight_ec']}{args['weight_tt']}" + f"-Seed_{args['test_seed']}"   # 便于区分实验场景
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'   # 生成一个介于100,000到999,999之间随机整数，作为独立易于识别的实验标识符
    wandb.init(
        name=exp_prefix + specific_name,
        # name=exp_prefix,
        group=group_name,
        project='Test_for_MT-FJSP',
        # config=args
    )
    # wandb.watch(model)  # wandb has some bug
    
    
    save_pth = result_pth
    mip_path = mip_pth

    n_job = args['n_job']
    n_machine = args['n_machine']
    n_edge = args['n_edge']

    weight_mk=args['weight_mk'] # 还是按照小数来的！
    weight_ec=args['weight_ec']
    weight_tt=args['weight_tt']

    # 加载模型的名称，只有top123和final，取消了epi参数
    job_name = "PPO_job_actor_"
    machine_name = "PPO_machine_actor_"
    path_new = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/trained_model/can_use/No_lr_decay/"
    file_mch = path_new + machine_name + f"J{args['mappo_scene'][0]}M{args['mappo_scene'][1]}E{args['mappo_scene'][2]}" + f"_{model_typy}.pth"  
    file_operation = path_new + job_name + f"J{args['mappo_scene'][0]}M{args['mappo_scene'][1]}E{args['mappo_scene'][2]}" + f"_{model_typy}.pth"

    """
    源代码的已训练好的模型： MAPPO + ESA
    """
    mappo_pth = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/tester/IoTJ_MAPPO/"
    mappo_mch = mappo_pth + "PPO_machine_actor_J%sM%sE%s_%s.pth" % (
        args['mappo_scene'][0], 
        args['mappo_scene'][1], 
        args['mappo_scene'][2],
        args['mappo_id'])  # 也记录训练的最后一次的model TODO 1124-加载小场景，训练大场景
    mappo_operation = mappo_pth + "PPO_operation_actor_J%sM%sE%s_%s.pth" % (
        args['mappo_scene'][0], 
        args['mappo_scene'][1], 
        args['mappo_scene'][2],
        args['mappo_id'])  # 也记录训练的最后一次的model

    esa_pth = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/tester/ESWA_MPPO/"
    esa_mch = esa_pth + "esa_PPO_machine_actor_J%sM%sE%s_%s.pth" % (
        args['esa_scene'][0], 
        args['esa_scene'][1], 
        args['esa_scene'][2],
        args['esa_id'])  # 也记录训练的最后一次的model TODO 1225-esa:加载小场景，训练大场景
    esa_operation = esa_pth + "esa_PPO_operation_actor_J%sM%sE%s_%s.pth" % (
        args['esa_scene'][0], 
        args['esa_scene'][1],
        args['esa_scene'][2], 
        args['esa_id'])  # 也记录训练的最后一次的model
    

    """
    循环运行rules，记录关注的指标，并画出结果
    1、画箱型图
    2、一图数据存在一个大list中：每种组合都是其中的一个小list
    3、依次循环遍历所有Rules的组合即可
    """

    total_rules = 12 # PDRs总共12个
    method_num = 6  # PDRs + RA + MIP + ESA-G + G + S + TODO 250510-new12800样本训练
    
    """
    4指标分别和MIP对应的4指标求gap，然后进行加权和
    """
    Obj_4rGap_list = [[] for _ in range(total_rules + method_num)]  # 初始化为空列表，

    """
    4指标分别和MIP对应的4指标求gap
    """
    pt_gap_list = [[] for _ in range(total_rules + method_num)]  # 初始化为空列表,
    tt_gap_list = [[] for _ in range(total_rules + method_num)]  # 
    mk_gap_list = [[] for _ in range(total_rules + method_num)]  # 
    it_gap_list = [[] for _ in range(total_rules + method_num)]  # 

    """
    4个指标的真实值 ： done之后的env输出的各个指标的真实值
    """
    real_pt_list  = [[] for _ in range(total_rules + method_num)]  #   加工能耗，功率=1
    real_tt_list  = [[] for _ in range(total_rules + method_num)]  #   运输能耗，功率=1
    real_mk_list = [[] for _ in range(total_rules + method_num)]  #   带运输时间的makespan
    real_it_list  = [[] for _ in range(total_rules + method_num)]  #  待机能耗，功率=1

    # TODO 新增
    time_list = [[] for _ in range(total_rules + method_num)]  # 算法推理时间
    std_list = [[] for _ in range(total_rules + method_num)]  #  

    """
    done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0
    """
    Obj_allmethod_lst = [[] for _ in range(total_rules + method_num)]  # 初始化为空列表

    # ==================================================================================================
    
    
    """
    生成用于测试的dataset：
    1、直接就是 = eval_samples, 测试和验证集的数据集大小一样
    2、seed ！= eval_seed，采用的随机数seed不用改相同 
    """
    '''换成负数的样本数据！！！！！！！！！！！！'''
    test_data_pth = '/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/instance/'
    test_dataset_path = test_data_pth + f"test_Instance_J{n_job}M{n_machine}E{n_edge}.pkl"
    test_instance = Instance_Dataset(  # TODO def的所有形参都默认值，这里传入可以只选择要改的值！
            generate_true=0,    # TODO generate_true=0直接读取文件的地址，不需要其他参数
            dataset_pth=test_dataset_path)
    Logger.log("Testing/instance", f"========== test_instance.t[-1]={test_instance.t[-1].shape} ==========", print_true=1)

    """
    实例化PPO，里边有网络的初始化
    """
    ppo_12800ins = PPOAlgorithm(args=args, load_pretrained=False)
    ppo_test_greedy = PPOAlgorithm(args=args, load_pretrained=False)
    ppo_test_sampling = PPOAlgorithm(args=args, load_pretrained=False) # TODO 1217-新增进行随机采样的模型！
    esa_ppo_test = PPOAlgorithm(args=args, load_pretrained=False) # TODO 1225-新增ESA的greedy模型！

    """
    加载训练好的模型
    """
    if os.path.exists(mappo_mch):
        print('------------load the model----------------')
        
        ppo_12800ins.machine_actor_gcn.load_state_dict(torch.load(file_mch))  # 按照路径地址加载预训练好的模型 TODO 250509-所有12800样本训练
        ppo_12800ins.job_actor.load_state_dict(torch.load(file_operation))
        print('ppo_12800ins.machine_actor_gcn.state_dict() = ', ppo_12800ins.machine_actor_gcn.state_dict()) 
        print('ppo_12800ins.job_actor.state_dict() = ', ppo_12800ins.job_actor.state_dict())  # 打印出加载后的模型中的所有参数
        
        ppo_test_greedy.machine_actor_gcn.load_state_dict(torch.load(mappo_mch))  # 按照路径地址加载预训练好的模型 TODO 1225-greedy方式
        ppo_test_greedy.job_actor.load_state_dict(torch.load(mappo_operation))
        print('ppo_test_greedy.machine_actor_gcn.state_dict() = ', ppo_test_greedy.machine_actor_gcn.state_dict())  
        print('ppo_test_greedy.job_actor.state_dict() = ', ppo_test_greedy.job_actor.state_dict())  

        ppo_test_sampling.machine_actor_gcn.load_state_dict(torch.load(mappo_mch))  # 按照路径地址加载预训练好的模型 TODO 1225-随机采样方式
        ppo_test_sampling.job_actor.load_state_dict(torch.load(mappo_operation))
        print('ppo_test_sampling.machine_actor_gcn.state_dict() = ', ppo_test_sampling.machine_actor_gcn.state_dict())  
        print('ppo_test_sampling.job_actor.state_dict() = ', ppo_test_sampling.job_actor.state_dict())  

        esa_ppo_test.esa_machine_actor_gcn.load_state_dict(torch.load(esa_mch))  # 按照路径地址加载预训练好的模型 TODO 1225-ESA的greedy
        esa_ppo_test.esa_job_actor.load_state_dict(torch.load(esa_operation))
        print('esa_ppo_test.esa_machine_actor_gcn.state_dict() = ', esa_ppo_test.esa_machine_actor_gcn.state_dict())  
        print('esa_ppo_test.esa_job_actor.state_dict() = ', esa_ppo_test.esa_job_actor.state_dict())  
    else:
        print( f"--------------------------------------------------------Attention!!!!! Model is not loaded!!!----------------------------------------------------------")

    """
    注意：MIP_MO_FJSP_Gurobi的结果：
    1、已经是同样数据跑完configs.eval_sample次数了！（所以直接=就好！）
    2、需要按照这里的形参，只选取其中的samples的数据来对比，防止数据维度不一样，没法画图！
    3、samples <= configs.eval_sample!!!
    4、Gurobi的结果放在倒数第二个！！

    注意：真实值用负数来衡量 + 真实无权重总和用正数 + cost因为统计的是reward累加，所以会有一个放缩100倍的因子

    MIP的对应值存在cost_dict的对应key的list中，按照样本123.。。100的顺序
    cost_dict = {
        "runtime": [],
        "best_objective": [],
        "Makespan": [],
        "MachineEC": [],
        "MachineIdleT": [],
        "TransEC": [],
    }
    """
    MIP_result_file = mip_path + "MO_FJSP_MIP_result_(J%s_M%s_seed%s_sample%s_w%s%s%s).csv" % \
                      (n_job, n_machine, seed, 100,
                       int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
    MIP_cost_dict = read_MIP_result_from_csv(MIP_result_file)

    # TODO 新增add idea_refer = [mk, pt, transT, idleT] 平衡多指标不同量级之间的训练有效性 + shape = 4指标真值元素，各元素100个结果值
    MIP_idea_cost_refer = [MIP_cost_dict["Makespan"], MIP_cost_dict["MachineEC"],
                           MIP_cost_dict["TransEC"], MIP_cost_dict["MachineIdleT"]]  # TODO 别忘了这里是平均能耗（MIP模型改成平均能耗了！）

    # TODO 按照MIP的值为参考，不用算了，MIP的全是0！笑死（除了4真值直接相加）
    """4指标分别和MIP对应指标求gap，然后进行加权和"""
    Obj_4rGap_list[-5] = [0] * samples  # 记录MIP的验证cost到对应lst(有权重的！！！)
    
    """4指标分别和MIP对应的4指标求gap"""
    pt_gap_list[-5] = [0] * samples
    tt_gap_list[-5] = [0] * samples
    mk_gap_list[-5] = [0] * samples
    it_gap_list[-5] = [0] * samples

    """4个指标的真实值 ： done之后的env输出的各个指标的真实值"""
    real_pt_list[-5] = MIP_idea_cost_refer[1][:samples] # 记录各个指标的真实值！
    real_tt_list[-5] = MIP_idea_cost_refer[2][:samples]
    real_mk_list[-5] = MIP_idea_cost_refer[0][:samples]
    real_it_list[-5] = MIP_idea_cost_refer[3][:samples]

    time_list[-5] = [x for x in MIP_cost_dict["runtime"][:samples]]  # 记录MIP的运行时间，后续的话时间 = limit time，没有最优解了

    """done之后的真实的4指标"""
    sum_4target = [x + y + z + k for x, y, z, k in zip(MIP_cost_dict["MachineEC"][:samples],
                                                        MIP_cost_dict["TransEC"][:samples],
                                                        MIP_cost_dict["Makespan"][:samples],
                                                        MIP_cost_dict["MachineIdleT"][:samples])]  # 4指标直接相加:还是list

    # TODO 1110-改成4指标直接加权和，看看效果
    weighted_4target = [weight_ec * x + weight_tt * y + weight_mk * z + weight_ec * k
                           for x, y, z, k in zip(MIP_cost_dict["MachineEC"][:samples],
                                                 MIP_cost_dict["TransEC"][:samples],
                                                 MIP_cost_dict["Makespan"][:samples],
                                                 MIP_cost_dict["MachineIdleT"][:samples])]  # 4指标直接加权和:还是list
    """done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0"""
    Obj_allmethod_lst[-5] = weighted_4target  # 直接4指标加权和

    
    """
    此时是循环samples次数，来验证训练好的PPO和PDRs
    """
    for i_episode in range(samples):  # 实验次数 = test的次数 || # for i_episode in range(65,samples):  # 从idx = 65开始运行
    
        """[========================================================12800ins-PPO=============================================================]"""
        
        print("n"*100)
        print("250510- New 12800 ins PPO greedy Start")
        print("n"*100)
        
        """
        相同变量会被重新赋值，记录不会串行
        
        1、记录当前轨迹的各指标累加cost，累加的，TODO ！= 真实值，初始的r0不是0
        cumsum_4r_dict = {
            "opr_Gt": 0,
            "opr_mk": 0,
            "opr_idleT": 0,
            "opr_pt": 0,
            "opr_transT": 0
        }
        2、真实值：real_4r = [mkt, pt, transT, idleT]  # 轨迹走完的最终4个指标 = 4 cost
        3、4个真实指标的加权和obj
        """
        ppo_st = time.time()
        cumsum_4r_dict, real_4r, Obj = validate_cost_gcn_jointActor_GAT(
                    ppo=ppo_12800ins,
                    gantt_flag=False,  # TODO 1130-不用画gantt图了，m有不能做的，大概率不可能同一个m！加快训练速度！
                    data=test_instance,  # 传入的是dataset，定义的可以直接.x读取数据
                    data_index=i_episode,
                    data_type="random",
                    greedy=True,
                    args=args
                    )  # 固定seed进行test，对所有方法进行验证
        ppo_end = time.time()

        # TODO 相对MIP差值的指标 + real_4r = [mk, pt, transT, idleT] = 最后step的真实输出值（+） + 每次产生1episode的值，跑samples次！！！
        ppo_4r_gap = (np.array(real_4r) - np.array(MIP_idea_cost_refer)[:, i_episode]) \
                               / np.array(MIP_idea_cost_refer)[:, i_episode]  # idea里边是所有100个数据，我们只要对应episode的！！！差值为正数，越小越好！！！
        ppo_obj_gap = weight_mk * ppo_4r_gap[0] \
                            + weight_ec * (ppo_4r_gap[1] + ppo_4r_gap[3]) \
                            + weight_tt * ppo_4r_gap[2]

        """4指标分别和MIP对应的4指标求gap，然后进行加权和"""
        Obj_4rGap_list[-1].append(ppo_obj_gap)  # 记录4指标分别和MIP对应指标求gap，然后进行加权和

        """4真实指标和MIP对应的4指标的gap"""
        pt_gap_list[-1].append(ppo_4r_gap[1])  # 对应[mk, pt, transT, idleT]来选
        tt_gap_list[-1].append(ppo_4r_gap[2])
        mk_gap_list[-1].append(ppo_4r_gap[0])
        it_gap_list[-1].append(ppo_4r_gap[3])

        """4个指标的真实值 ： done之后的env输出的各个指标的真实值"""
        real_pt_list[-1].append(real_4r[1])  # 记录各个指标的真实值！
        real_tt_list[-1].append(real_4r[2])
        real_mk_list[-1].append(real_4r[0])
        real_it_list[-1].append(real_4r[3])

        """done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0"""
        ppo_eval_Wed_raw_cost = weight_mk * real_4r[0] \
                                + weight_ec * (real_4r[1] + real_4r[3]) \
                                + weight_tt * real_4r[2]
        Obj_allmethod_lst[-1].append(ppo_eval_Wed_raw_cost)  # 直接4指标加权和

        time_list[-1].append(ppo_end - ppo_st)  # 记录结果的用时

        Logger.log("Testing/12800ins-ppo", f"sample={i_episode}, Obj={Obj_allmethod_lst[-1][-1]}/MIP={weighted_4target[i_episode]}, mk={real_mk_list[-1][-1]}/MIP={MIP_idea_cost_refer[0][i_episode]}, pt={real_pt_list[-1][-1]}/MIP={MIP_idea_cost_refer[1][i_episode]}, it={real_it_list[-1][-1]}/MIP={MIP_idea_cost_refer[3][i_episode]}, tt={real_tt_list[-1][-1]}/MIP={MIP_idea_cost_refer[2][i_episode]}", print_true=1)
        
        print("n"*100)
        print("250510- New 12800 ins PPO greedy Done")
        print("n"*100)
        
        """[====================================================P-G=====================================================================]"""
        
        print("n"*100)
        print("PPO greedy Start")
        print("n"*100)

        ppo_st = time.time()
        cumsum_4r_dict, real_4r_0, Obj = validate_cost_gcn_jointActor_GAT(
                    ppo=ppo_test_greedy,
                    gantt_flag=False,  # TODO 1130-不用画gantt图了，m有不能做的，大概率不可能同一个m！加快训练速度！
                    data=test_instance,  # 传入的是dataset，定义的可以直接.x读取数据
                    data_index=i_episode,
                    data_type="random",
                    greedy=True,
                    args=args
                    )  # 固定seed进行test，对所有方法进行验证
        ppo_end = time.time()

        # TODO 相对MIP差值的指标 + real_4r = [mk, pt, transT, idleT] = 最后step的真实输出值（+） + 每次产生1episode的值，跑samples次！！！
        ppo_4r_gap = (np.array(real_4r_0) - np.array(MIP_idea_cost_refer)[:, i_episode]) \
                               / np.array(MIP_idea_cost_refer)[:, i_episode]  # idea里边是所有100个数据，我们只要对应episode的！！！差值为正数，越小越好！！！
        ppo_obj_gap = weight_mk * ppo_4r_gap[0] \
                            + weight_ec * (ppo_4r_gap[1] + ppo_4r_gap[3]) \
                            + weight_tt * ppo_4r_gap[2]

        """4指标分别和MIP对应的4指标求gap，然后进行加权和"""
        Obj_4rGap_list[-3].append(ppo_obj_gap)  # 记录4指标分别和MIP对应指标求gap，然后进行加权和

        """4真实指标和MIP对应的4指标的gap"""
        pt_gap_list[-3].append(ppo_4r_gap[1])  # 对应[mk, pt, transT, idleT]来选
        tt_gap_list[-3].append(ppo_4r_gap[2])
        mk_gap_list[-3].append(ppo_4r_gap[0])
        it_gap_list[-3].append(ppo_4r_gap[3])

        """4个指标的真实值 ： done之后的env输出的各个指标的真实值"""
        real_pt_list[-3].append(real_4r_0[1])  # 记录各个指标的真实值！
        real_tt_list[-3].append(real_4r_0[2])
        real_mk_list[-3].append(real_4r_0[0])
        real_it_list[-3].append(real_4r_0[3])

        """done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0"""
        ppo_eval_Wed_raw_cost = weight_mk * real_4r_0[0] \
                                + weight_ec * (real_4r_0[1] + real_4r_0[3]) \
                                + weight_tt * real_4r_0[2]
        Obj_allmethod_lst[-3].append(ppo_eval_Wed_raw_cost)  # 直接4指标加权和

        time_list[-3].append(ppo_end - ppo_st)  # 记录结果的用时

        Logger.log("Testing/P-G", f"sample={i_episode}, Obj={Obj_allmethod_lst[-3][-1]}/MIP={weighted_4target[i_episode]}, mk={real_mk_list[-3][-1]}/MIP={MIP_idea_cost_refer[0][i_episode]}, pt={real_pt_list[-3][-1]}/MIP={MIP_idea_cost_refer[1][i_episode]}, it={real_it_list[-3][-1]}/MIP={MIP_idea_cost_refer[3][i_episode]}, tt={real_tt_list[-3][-1]}/MIP={MIP_idea_cost_refer[2][i_episode]}", print_true=1)
        
        print("n"*100)
        print("PPO greedy Done")
        print("n"*100)
        
        """[======================================================P-S===================================================================]"""
        
        
        # TODO 1217-新增随机采样的PPO的test
        print("n"*100)
        print("PPO sampling Start")
        print("n"*100)
        
        ppo_st_1 = time.time()
        cumsum_4r_dict, real_4r_1, Obj = validate_cost_gcn_jointActor_GAT(
                    ppo=ppo_test_sampling,
                    gantt_flag=False,  # TODO 1130-不用画gantt图了，m有不能做的，大概率不可能同一个m！加快训练速度！
                    data=test_instance,  # 传入的是dataset，定义的可以直接.x读取数据
                    data_index=i_episode,
                    data_type="random",
                    greedy=False,
                    args=args
                    )  # 固定seed进行test，对所有方法进行验证
        ppo_end_1 = time.time()

        # TODO 新增相对MIP差值的指标 + untilNow = [mk, pt, transT, idleT] = 最后step的真实输出值（+） + 每次产生1episode的值，跑samples次！！！
        ppo_new_related_cost1 = (np.array(real_4r_1) - np.array(MIP_idea_cost_refer)[:, i_episode]) \
                                / np.array(MIP_idea_cost_refer)[:, i_episode]  # idea里边是所有100个数据，我们只要对应episode的！！！差值为正数，越小越好！！！
        ppo_eval_Wed_cost1 = weight_mk * ppo_new_related_cost1[0] \
                            + weight_ec * (ppo_new_related_cost1[1] + ppo_new_related_cost1[3]) \
                            + weight_tt * ppo_new_related_cost1[2]

        """4指标分别和MIP对应的4指标求gap，然后进行加权和"""
        Obj_4rGap_list[-2].append(ppo_eval_Wed_cost1)  # 记录ppo的验证cost到对应lst(有权重的！！！)

        """4指标分别和MIP对应的4指标求gap"""
        pt_gap_list[-2].append(ppo_new_related_cost1[1])  # 对应[mk, pt, transT, idleT]来选
        tt_gap_list[-2].append(ppo_new_related_cost1[2])
        mk_gap_list[-2].append(ppo_new_related_cost1[0])
        it_gap_list[-2].append(ppo_new_related_cost1[3])

        """4个指标的真实值 ： done之后的env输出的各个指标的真实值"""
        real_pt_list[-2].append(real_4r_1[1])    # 记录各个指标的原始值！
        real_tt_list[-2].append(real_4r_1[2])
        real_mk_list[-2].append(real_4r_1[0])
        real_it_list[-2].append(real_4r_1[3])
        """done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0"""
        ppo_eval_Wed_raw_cost1 = weight_mk * real_4r_1[0] \
                                + weight_ec * (real_4r_1[1] + real_4r_1[3]) \
                                + weight_tt * real_4r_1[2]
        Obj_allmethod_lst[-2].append(ppo_eval_Wed_raw_cost1)  # 直接4指标加权和

        time_list[-2].append(ppo_end_1 - ppo_st_1)  # 记录结果的用时

        Logger.log("Testing/P-S", f"sample={i_episode}, Obj={Obj_allmethod_lst[-2][-1]}/MIP={weighted_4target[i_episode]}, mk={real_mk_list[-2][-1]}/MIP={MIP_idea_cost_refer[0][i_episode]}, pt={real_pt_list[-2][-1]}/MIP={MIP_idea_cost_refer[1][i_episode]}, it={real_it_list[-2][-1]}/MIP={MIP_idea_cost_refer[3][i_episode]}, tt={real_tt_list[-2][-1]}/MIP={MIP_idea_cost_refer[2][i_episode]}", print_true=1)
        
        print("n"*100)
        print("PPO sampling Done")
        print("n"*100)
        
        """[==========================================================ESA===============================================================]"""

        # TODO 1225-新增ESA的test结果，greedy方式
        # 单次测试，反馈的累加误差Gt = cost
        # cumsum_4r_dict, real_4r, untilNow = validate_cost_gcn_jointActor(
        print("n"*100)
        print("ESA-PPO greedy Start")
        print("n"*100)
        
        ppo_st_2 = time.time()
        cumsum_4r_dict, real_4r_2, Obj = esa_validate_cost_gcn_jointActor_GAT(
                    ppo=esa_ppo_test,
                    gantt_flag=False,  # TODO 1130-不用画gantt图了，m有不能做的，大概率不可能同一个m！加快训练速度！
                    data=test_instance,  # 传入的是dataset，定义的可以直接.x读取数据
                    data_index=i_episode,
                    data_type="random",
                    greedy=True,
                    args=args
                    )  # 固定seed进行test，对所有方法进行验证
        ppo_end_2 = time.time()

        # TODO 新增相对MIP差值的指标 + untilNow = [mk, pt, transT, idleT] = 最后step的真实输出值（+） + 每次产生1episode的值，跑samples次！！！
        ppo_new_related_cost2 = (np.array(real_4r_2) - np.array(MIP_idea_cost_refer)[:, i_episode]) \
                                / np.array(MIP_idea_cost_refer)[:,i_episode]  # idea里边是所有100个数据，我们只要对应episode的！！！差值为正数，越小越好！！！
        ppo_eval_Wed_cost2 = weight_mk * ppo_new_related_cost2[0] \
                             + weight_ec * (ppo_new_related_cost2[1] + ppo_new_related_cost2[3]) \
                             + weight_tt * ppo_new_related_cost2[2]

        """4指标分别和MIP对应的4指标求gap，然后进行加权和"""
        Obj_4rGap_list[-4].append(ppo_eval_Wed_cost2)  # 记录ppo的验证cost到对应lst(有权重的！！！)

        """4指标分别和MIP对应的4指标求gap"""
        pt_gap_list[-4].append(ppo_new_related_cost2[1])  # 对应[mk, pt, transT, idleT]来选
        tt_gap_list[-4].append(ppo_new_related_cost2[2])
        mk_gap_list[-4].append(ppo_new_related_cost2[0])
        it_gap_list[-4].append(ppo_new_related_cost2[3])

        """4个指标的真实值 ： done之后的env输出的各个指标的真实值"""
        real_pt_list[-4].append(real_4r_2[1])  # 记录各个指标的原始值！
        real_tt_list[-4].append(real_4r_2[2])
        real_mk_list[-4].append(real_4r_2[0])
        real_it_list[-4].append(real_4r_2[3])
        
        """done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0"""
        ppo_eval_Wed_raw_cost2 = weight_mk * real_4r_2[0] \
                                 + weight_ec * (real_4r_2[1] + real_4r_2[3]) \
                                 + weight_tt * real_4r_2[2]
        Obj_allmethod_lst[-4].append(ppo_eval_Wed_raw_cost2)  # 直接4指标加权和

        time_list[-4].append(ppo_end_2 - ppo_st_2)  # 记录结果的用时

        Logger.log("Testing/ESA", f"sample={i_episode}, Obj={Obj_allmethod_lst[-4][-1]}/MIP={weighted_4target[i_episode]}, mk={real_mk_list[-4][-1]}/MIP={MIP_idea_cost_refer[0][i_episode]}, pt={real_pt_list[-4][-1]}/MIP={MIP_idea_cost_refer[1][i_episode]}, it={real_it_list[-4][-1]}/MIP={MIP_idea_cost_refer[3][i_episode]}, tt={real_tt_list[-4][-1]}/MIP={MIP_idea_cost_refer[2][i_episode]}", print_true=1)
        
        print("n"*100)
        print("ESA-PPO greedy Done")
        print("n"*100)
        
        """[=======================================================PDRs======================================================================]"""

        o_rules = 6 # TODO 1217-现在有7个task的选择方式，都列出来！(注意：最后的RA随机要额外处理，不循环选就完事了！)
        m_rules = 2  
        # 遍历所有的Rule的组合
        for o_i in range(o_rules):  # 6
            for m_i in range(m_rules):  # 2
                # 返回单次episode的Gt + 对应的cost的dict

                print("n"*100)
                print(f"[o_rule{o_i},m_rule{m_i}] PDRs Start")
                print("n"*100)
                
                pdrs_st = time.time()
                """ 
                返回一组样本跑完一个rules的累加即时r（加权后） + 即时r（加权后）和即时4指标的累加dict + 真实4指标值 
                pdrs_real_4r = [mk, pt, transT, idleT]
                """
                _, _, pdrs_real_4r = run_Rules_jointActions_withMinus_1217(
                            args=args,
                            o_rule=o_i,
                            m_rule=m_i,
                            data=test_instance,
                            data_index=i_episode,
                            data_type=None, # TODO 暂时我没有用到这2个参数，我在rule里边手动设置了
                            MLWKR_type=None)  # 循环6*2的PDRs，并记录,返回的是列表
                pdrs_end = time.time()

                """
                存储比较绕，但是就是对应rule组合存在对应位置
                1、按照0123来进行组合 
                2、返回的是单次episode的单个rule组合的4个指标的真实值
                """
                # TODO pdrs_real_4r = [mk, pt, transT, idleT] 一样的四指标 + 区别是还在循环不同的PDRs + 注意Rules里边要返回平均能耗
                pdrs_new_related_cost = (np.array(pdrs_real_4r) - np.array(MIP_idea_cost_refer)[:,i_episode]) \
                                        / np.array(MIP_idea_cost_refer)[:,i_episode]  # idea里边是所有100个数据，我们只要对应episode的！！！差值为正数，越小越好！！！
                pdrs_eval_Wed_cost = weight_mk * pdrs_new_related_cost[0] \
                                     + weight_ec * (pdrs_new_related_cost[1] + pdrs_new_related_cost[3]) \
                                     + weight_tt * pdrs_new_related_cost[2]

                """4指标分别和MIP对应的4指标求gap，然后进行加权和"""
                Obj_4rGap_list[m_rules * o_i + m_i].append(pdrs_eval_Wed_cost)  # 对应的总cost(有权重的！！！)

                """4指标分别和MIP对应的4指标求gap"""
                pt_gap_list[m_rules * o_i + m_i].append(pdrs_new_related_cost[1])  # 对应[mk, pt, transT, idleT]来选
                tt_gap_list[m_rules * o_i + m_i].append(pdrs_new_related_cost[2])
                mk_gap_list[m_rules * o_i + m_i].append(pdrs_new_related_cost[0])
                it_gap_list[m_rules * o_i + m_i].append(pdrs_new_related_cost[3])

                """4个指标的真实值 ： done之后的env输出的各个指标的真实值"""
                real_pt_list[m_rules * o_i + m_i].append(pdrs_real_4r[1])    # 记录各个指标的原始值！
                real_tt_list[m_rules * o_i + m_i].append(pdrs_real_4r[2])
                real_mk_list[m_rules * o_i + m_i].append(pdrs_real_4r[0])
                real_it_list[m_rules * o_i + m_i].append(pdrs_real_4r[3])

                """done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0"""
                pdrs_eval_Wed_raw_cost = weight_mk * pdrs_real_4r[0] \
                                         + weight_ec * (pdrs_real_4r[1] + pdrs_real_4r[3]) \
                                         + weight_tt * pdrs_real_4r[2]
                Obj_allmethod_lst[m_rules * o_i + m_i].append(pdrs_eval_Wed_raw_cost)  # 直接4指标加权和

                time_list[m_rules * o_i + m_i].append(pdrs_end - pdrs_st)  # 记录时间

                Logger.log("Testing/PDRs", f"sample={i_episode}, [o_rule{o_i},m_rule{m_i}], Obj={Obj_allmethod_lst[m_rules * o_i + m_i][-1]}/MIP={weighted_4target[i_episode]}, mk={real_mk_list[m_rules * o_i + m_i][-1]}/MIP={MIP_idea_cost_refer[0][i_episode]}, pt={real_pt_list[m_rules * o_i + m_i][-1]}/MIP={MIP_idea_cost_refer[1][i_episode]}, it={real_it_list[m_rules * o_i + m_i][-1]}/MIP={MIP_idea_cost_refer[3][i_episode]}, tt={real_tt_list[m_rules * o_i + m_i][-1]}/MIP={MIP_idea_cost_refer[2][i_episode]}", print_true=1)
                
                print("n"*100)
                print(f"[o_rule{o_i},m_rule{m_i}] PDRs Done")
                print("n"*100)
                
        """[=======================================================RA======================================================================]"""
        print("n"*100)
        print(f"[RA,RA] PDRs Start")
        print("n"*100)
        
        """ 
        返回一组样本跑完一个rules的累加即时r（加权后） + 即时r（加权后）和即时4指标的累加dict + 真实4指标值 
        pdrs_real_4r_1 = [mk, pt, transT, idleT]
        """
        pdrs_st1 = time.time()
        _, _, pdrs_real_4r_1 = run_Rules_jointActions_withMinus_1217(
                    args=args,
                    o_rule=6,  # 对应的就是随机选择
                    m_rule=2,
                    data=test_instance,
                    data_index=i_episode,
                    data_type=None,# TODO 暂时我没有用到这2个参数，我在rule里边手动设置了
                    MLWKR_type=None)  # 指定6，2对应RA+RA的PDRs，并记录,返回的是列表
        pdrs_end1 = time.time()

        """
        存储比较绕，但是就是对应rule组合存在对应位置
        1、按照0123来进行组合 
        2、返回的是单次episode的单个rule组合的4个指标的真实值！！！
        """
        # TODO pdrs_real_4r_1 = [mk, pt, transT, idleT] 一样的四指标 + 区别是还在循环不同的PDRs + 注意Rules里边要返回平均能耗
        pdrs_new_related_cost = (np.array(pdrs_real_4r_1) - np.array(MIP_idea_cost_refer)[:, i_episode]) \
                                / np.array(MIP_idea_cost_refer)[:,i_episode]  # idea里边是所有100个数据，我们只要对应episode的！！！差值为正数，越小越好！！！
        pdrs_eval_Wed_cost = weight_mk * pdrs_new_related_cost[0] \
                             + weight_ec * (pdrs_new_related_cost[1] + pdrs_new_related_cost[3]) \
                             + weight_tt * pdrs_new_related_cost[2]

        """4指标分别和MIP对应的4指标求gap，然后进行加权和"""
        Obj_4rGap_list[-6].append(pdrs_eval_Wed_cost)  # 对应的总cost(有权重的！！！)

        """4指标分别和MIP对应的4指标求gap"""
        pt_gap_list[-6].append(pdrs_new_related_cost[1])  # 对应[mk, pt, transT, idleT]来选
        tt_gap_list[-6].append(pdrs_new_related_cost[2])
        mk_gap_list[-6].append(pdrs_new_related_cost[0])
        it_gap_list[-6].append(pdrs_new_related_cost[3])

        """4个指标的真实值 ： done之后的env输出的各个指标的真实值"""
        real_pt_list[-6].append(pdrs_real_4r_1[1])    # 记录各个指标的原始值！
        real_tt_list[-6].append(pdrs_real_4r_1[2])
        real_mk_list[-6].append(pdrs_real_4r_1[0])
        real_it_list[-6].append(pdrs_real_4r_1[3])
        """done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0"""
        pdrs_eval_Wed_raw_cost = weight_mk * pdrs_real_4r_1[0] \
                                 + weight_ec * (pdrs_real_4r_1[1] + pdrs_real_4r_1[3]) \
                                 + weight_tt * pdrs_real_4r_1[2]
        Obj_allmethod_lst[-6].append(pdrs_eval_Wed_raw_cost)  # 直接4指标加权和

        time_list[-6].append(pdrs_end1 - pdrs_st1)  # 记录时间

        Logger.log("Testing/PDRs", f"sample={i_episode}, [RA,RA], Obj={Obj_allmethod_lst[-6][-1]}/MIP={weighted_4target[i_episode]}, mk={real_mk_list[-6][-1]}/MIP={MIP_idea_cost_refer[0][i_episode]}, pt={real_pt_list[-6][-1]}/MIP={MIP_idea_cost_refer[1][i_episode]}, it={real_it_list[-6][-1]}/MIP={MIP_idea_cost_refer[3][i_episode]}, tt={real_tt_list[-6][-1]}/MIP={MIP_idea_cost_refer[2][i_episode]}", print_true=1)
                
        print("n"*100)
        print(f"[RA,RA] PDRs Done")
        print("n"*100)
        """[===============================================================================================================================]"""
        
        """==============================================
        保存数据用于画箱型图（有均值！）: 
        1、for循环里边，小lst会不断增加至100个元素
        2、一个大list，其中每个小lst是每一个方法，每个方法有100组数据，对应100个test样例
        3、mean和std之后，每个方法只有1个数据了，就不画箱型图了
        =================================================="""
        list_fig = []  # for循环每次都清0了！只保留当前的结果

        """4个指标的真实值 ： done之后的env输出的各个指标的真实值"""
        list_fig.append(real_pt_list)
        list_fig.append(real_tt_list)
        list_fig.append(real_mk_list)
        list_fig.append(real_it_list)
        """4指标分别和MIP对应的4指标求gap"""
        list_fig.append(pt_gap_list)
        list_fig.append(tt_gap_list)
        list_fig.append(mk_gap_list)
        list_fig.append(it_gap_list)
        """done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0"""
        list_fig.append(Obj_allmethod_lst)
        obj_mean_show = [np.mean(sublist1) for sublist1 in Obj_allmethod_lst]  # !TODO For循环里边，实时计算mean
        """4指标分别和MIP对应的4指标求gap，然后进行加权和"""
        list_fig.append(Obj_4rGap_list)  # 4指标分别和MIP对应的4指标求gap，然后进行加权和
        Obj_4rGap_mean_show = [np.mean(sublist) for sublist in Obj_4rGap_list]
        
        Logger.log("Testing/result_each_episode", f"sample={i_episode}, realtime_mean_Obj_allmethods={[round(x, 1) for x in obj_mean_show]},realtime_mean_Obj_4rgap_allmethods={[round(x, 1) for x in Obj_4rGap_mean_show]}", print_true=1) # 函数 round(x, 1) 将数字 x 四舍五入到小数点后一位。
        
        list_fig.append(time_list)  # 时间也画出来！
        list_fig.append([])  #  #  TODO 因为是4行3列，所以多加一个空元素，防止idx报错
        
        time_show = [np.mean(sublist1) for sublist1 in time_list] # 记录的运行时间，各个方法的结果
        std_show = [np.std(sublist1) for sublist1 in Obj_allmethod_lst]  # 记录的加权和的标准差std，各个方法的结果，看稳定程度
        Logger.log("Testing/result_inference_time", f"sample={i_episode}, realtime_mean_time_allmethods={[round(x, 3) for x in time_show]},realtime_std_Obj_allmethods={[round(x, 1) for x in std_show]}", print_true=1) # 函数 round(x, 1) 将数字 x 四舍五入到小数点后一位。
        

        result = [obj_mean_show, time_show, std_show, Obj_4rGap_mean_show] # 实时记录：obj的均值 + 推理时间均值 + std + 各指标对mip的gap的加权和的均值
        result_file = save_pth + "Results_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
                  % (n_job, n_machine, n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
        # 将数据存储到CSV文件中; 每次写入都是覆盖的方式，最终展示的是跑完100次的结果
        with open(result_file, 'w', newline='') as csvfile1:  # `newline`参数设置为空字符串`''`，以确保写入CSV文件时不进行额外的换行操作
            writer = csv.writer(csvfile1)
            writer.writerows(result)

        # 画图
        # result_box_plot(list_fig, 2, 3)  # 画图1行3列
        
        keys_name =  ['FIFO+SPT', 'FIFO+SEC',
                    'MOR+SPT', 'MOR+SEC',
                    'LWKR_T+SPT', 'LWKR_T+SEC',
                    'LWKR_PT+SPT', 'LWKR_PT+SEC',
                    'MWKR_T+SPT',  'MWKR_T+SEC',
                    'MWKR_PT+SPT', 'MWKR_PT+SEC',
                    'RA+RA','MIP_Solver', 'ESA-G', 
                    'PPO-G','PPO-S', 'new12800']
        box_title = ['Real_PT','Real_TT','Real_MK','Real_IT', 
                    'Real_PT_gap','Real_TT_gap','Real_MK_gap','Real_IT_gap', 
                    'Obj','Obj_4rgap','Inference_time', 'None']  #  因为是4行3列，所以多加一个空元素，防止idx报错
        fig_pth = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/results/test_results/Fig/"
        if i_episode == samples - 1:  #  ！TODO 250511-改为最后一个数据episode才画箱型图
            show = True
            result_box_plot_eachEpisode_1217(args=args,
                                            list=list_fig, 
                                            rows1=4, # 画图几行 
                                            cols1=3,  # 画图几列
                                            title_name=box_title,
                                            fig_pth=fig_pth,
                                            epi=i_episode,
                                            method_name=keys_name, 
                                            show_on=show)  # 画图1行3列
        else:
            show = False

    """
    wandb在线画图：
    1、传入字典dict，key就是图title，然后value就是数值点，
    2、相同key循环传入100次，就是在同一个图画出100个这个数值的折线
    
    本节，因为要每个指标对比18个方法，所以采用数据都采集完自定义图表，然后wandb绘图方式，不需要循环记录
        4个指标的真实值 ： done之后的env输出的各个指标的真实值
        4指标分别和MIP对应的4指标求gap
        done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0
        4指标分别和MIP对应的4指标求gap，然后进行加权和
    """
    data_lst = [real_pt_list, real_tt_list, real_mk_list, real_it_list,
                pt_gap_list, tt_gap_list, mk_gap_list, it_gap_list,
                Obj_allmethod_lst, Obj_4rGap_list, time_list]   # 要画的指标图，每个指标包含18个方法的lst，每个小lst有100个数据
    
    # PDRs + RA + MIP + ESA-G + G + S + TODO 250510-new12800样本训练
    for i in range(11):  # 遍历指标，每个指标1个图，对比18个方法
        # 创建一个图表
        data = data_lst[i]
        chart = wandb.plot.line_series(
            xs=list(range(samples)),  # X轴数据, 01234，。。。，100-1
            ys=[data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15], data[16], data[17]],
            keys=keys_name,
            title=f"Testing/100instances/{box_title[i]}",
            xname="Instance_idx"
        )
        # 将图表记录到 wandb
        wandb.log({box_title[i]: copy.deepcopy(chart)})
 
    """==============================================
    epi done之后储存数据 : 一个大list，其中每个小lst是每一个方法，每个方法有100组数据，对应100个test样例
    =================================================="""
    # TODO 1109-新增数据存储和读取（如果你想要将数据追加到文件末尾而不覆盖已有内容，你可以使用追加模式 `'a'` 来打开文件）
    """4指标分别和MIP对应的4指标求gap"""
    mk_file = save_pth + "MK_gap_mip_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
              % (n_job, n_machine, n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
    
    pt_file = save_pth + "PT_gap_mip_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
              % (n_job, n_machine,  n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
    
    tt_file = save_pth + "TT_gap_mip_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
              % (n_job, n_machine, n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
    
    it_file = save_pth + "IT_gap_mip_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
              % (n_job, n_machine, n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
    
    """4个指标的真实值 ： done之后的env输出的各个指标的真实值"""
    mk_file_raw = save_pth + "Real_MK_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
              % (n_job, n_machine,  n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
    
    pt_file_raw = save_pth + "Real_PT_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
              % (n_job, n_machine, n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
    
    tt_file_raw = save_pth + "Real_TT_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
              % (n_job, n_machine, n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
    
    it_file_raw = save_pth + "Real_IT_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
              % (n_job, n_machine,  n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
   
    """done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0"""
    w_raw_cost_file = save_pth + "Obj_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
                    % (n_job, n_machine, n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
    
    """4指标分别和MIP对应的4指标求gap，然后进行加权和"""
    W_gap_cost_file = save_pth + "Weighted_4r_gap_mip4r_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
                     % (n_job, n_machine,  n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
    
    Runtime_file = save_pth + "Runtime_J%s_M%s_E%s_Seed%s_Weight%s%s%s.csv" \
                   % (n_job, n_machine,  n_edge, seed, int(weight_mk * 10), int(weight_ec * 10), int(weight_tt * 10))
    
    # 将数据存储到CSV文件中
    """4指标分别和MIP对应的4指标求gap"""
    with open(mk_file, 'w', newline='') as csvfile:  # `newline`参数设置为空字符串`''`，以确保写入CSV文件时不进行额外的换行操作
        writer = csv.writer(csvfile)
        writer.writerows(mk_gap_list)
    with open(pt_file, 'w', newline='') as csvfile:  
        writer = csv.writer(csvfile)
        writer.writerows(pt_gap_list)
    with open(tt_file, 'w', newline='') as csvfile:  
        writer = csv.writer(csvfile)
        writer.writerows(tt_gap_list)
    with open(it_file, 'w', newline='') as csvfile:  
        writer = csv.writer(csvfile)
        writer.writerows(it_gap_list)

    """4个指标的真实值 ： done之后的env输出的各个指标的真实值"""
    with open(mk_file_raw, 'w', newline='') as csvfile:  
        writer = csv.writer(csvfile)
        writer.writerows(real_mk_list)
    with open(pt_file_raw, 'w', newline='') as csvfile:  
        writer = csv.writer(csvfile)
        writer.writerows(real_pt_list)
    with open(tt_file_raw, 'w', newline='') as csvfile:  
        writer = csv.writer(csvfile)
        writer.writerows(real_tt_list)
    with open(it_file_raw, 'w', newline='') as csvfile:  
        writer = csv.writer(csvfile)
        writer.writerows(real_it_list)

    """done之后直接4指标的直接加权和 = obj ！= 累加reward的加权和，因为初始r是预估值不为0"""
    with open(w_raw_cost_file, 'w', newline='') as csvfile:  
        writer = csv.writer(csvfile)
        writer.writerows(Obj_allmethod_lst)  # TODO-这里已经是原始指标的直接加权和，没有和MIP放缩的！
    
    """4指标分别和MIP对应的4指标求gap，然后进行加权和"""
    with open(W_gap_cost_file, 'w', newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows(Obj_4rGap_list) # 和mip放缩之后求解的gap的加权和
    
    with open(Runtime_file, 'w', newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerows(time_list)

    # # 重新按照原格式读取数据
    # new_data = []
    # with open(mk_file, 'r') as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         new_data.append(list(map(int, row)))
    #
    # # 打印重新读取的数据
    # print(new_data)

    # TODO 1108-新增一个3D结果图，用来比较多目标是再合适不过了！
    data_3d = {}
    for method in range(len(Obj_allmethod_lst)):  # 表明现在采用了多少种方法，[[samples],[samples],.....], 其中samples对应采用多少样本来进行test
        data_3d[keys_name[method]] = {'MK': real_mk_list[method],
                                    'PT': real_pt_list[method],
                                    'TT': real_tt_list[method],
                                    'IT': real_it_list[method]}  # samples对应采用多少样本来进行test(查看真实值！)
    # print(data_3d)
    plot_test_3d_cross_fig(data=data_3d, fig_pth=fig_pth, label_name=keys_name)  # TODO 画3D图，暂时关闭

    """
    画3d图的时候输入data的格式
    data = {
            'Method 1': {
                'MK': np.random.rand(10),
                'PT': np.random.rand(10),
                'TT': np.random.rand(10),
                'IT': np.random.rand(10),
            },
            'Method 2': {
                'MK': np.random.rand(10),
                'PT': np.random.rand(10),
                'TT': np.random.rand(10),
                'IT': np.random.rand(10),
            },
            'Method 3': {
                'MK': np.random.rand(10),
                'PT': np.random.rand(10),
                'TT': np.random.rand(10),
                'IT': np.random.rand(10),
            },
        }
        """
    


if __name__ == '__main__':  # 直接运行时，执行以下代码；导入该模块时，不会运行以下代码


    model_path = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/trained_model/can_use/No_lr_decay/"  # model存放位置
    result_save_pth = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/results/test_results/"
    mip_path = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/tester/Solver_seed3/"
    

    PPO_test_jointActor_PDRs_MIP_withMinus_DeltaIndicator_ESA_1225(
                args=vars(args),
                samples=100, 
                seed=args.test_seed, 
                model_typy="top1",
                model_pth=model_path, 
                result_pth=result_save_pth,
                mip_pth=mip_path,
                MLWKR_type="max", 
                greedy=True)
    
    pth2 = '/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/'
    testing_log_file = pth2 + f"/testing_log_J{args.n_job}M{args.n_machine}E{args.n_edge}_Weight{int(args.weight_mk*10)}{int(args.weight_mk*10)}{int(args.weight_mk*10)}.txt"
    Logger.get_logs(testing_log_file)
    
    print('=' * 250)
