import numpy as np
import random
import copy
import torch
import csv
import algorithm.ppo_algorithm
from algorithm.agent_func import greedy_select_machine_action, select_machine_action
from model.gcn_mlp import g_pool_cal
from graph_jsp_env.disjunctive_graph_jsp_env_singlestep import DisjunctiveGraphJspEnv_singleStep
from instance.generate_allsize_mofjsp_dataset import Logger, Result_Logger
from trainer.train_device import device


def read_MIP_result_from_csv(filepath):
    filename = filepath

    # 读取CSV文件中的数据
    data = []
    with open(filename, 'r') as file:
        """
        从data中遍历每一行读到的数据：
        ['runtime', 'best_objective', 'Makespan', 'MachineEC', 'MachineIdleT', 'TransEC']  第一行！！！需要跳过！
        ['25.853000164031982', '6627.449897232078', '352.49188700236493', '15574.64764122035', '523.0219814395231', '236.9264668359088']
        """
        reader = csv.reader(file)
        next(reader)  # 跳过第一行 , 第一行是那个表示数据的字符串啊！！！！
        for row in reader:  # 遍历每一行的data
            # 将每个元素从字符串转换为浮点数
            row = [float(value) for value in row]
            data.append(row)

    """
    注意：
    1、画箱型图：一个大list，里边都是不同方法的结果的list！转换形式
    """
    cost_dict = {
        "runtime": [],
        "best_objective": [],
        "Makespan": [],
        "MachineEC": [],
        "MachineIdleT": [],
        "TransEC": [],
    }
    for row in data: # 已经排除第一行是数据名称的row data；[1:]可以从第二个数据开始
        cost_dict["runtime"].append(row[0])    # 运行时间
        cost_dict["best_objective"].append(row[1])  # 带权重没有归一化处理的总目标
        cost_dict["Makespan"].append(row[2])
        cost_dict["MachineEC"].append(row[3])
        cost_dict["MachineIdleT"].append(row[4])
        cost_dict["TransEC"].append(row[5])
    # print(f"{var.varName} = {var.x}")

    # 打印读取的数据
    for row in data:
        print("从csv中读取到的数据：\n", row)

    return cost_dict


def validate_cost_gcn_jointActor_GAT(ppo, gantt_flag, data, data_index, data_type, greedy=True, args=None):
    """
    1、传入ppo是因为当前训练的网络的参数都在此中，验证时要看当前的网络的效果
    加载选择m的环境
    2、dataloader里边已经是按照env_batch进行拆分的样本数据了
    """
    configs = args
    n_job = configs['n_job']
    n_machine = configs['n_machine']
    n_total_task = n_job * n_machine
    n_edge = configs['n_edge']
    mask_value = configs['mask_value']
    reward_dict = configs['reward_scaling']
    m_scaling = configs['m_scaling']

    #记录当前轨迹的各指标累加cost
    cost_dict_cumsum = {
        "opr_Gt": 0,
        "opr_mk": 0,
        "opr_idleT": 0,
        "opr_pt": 0,
        "opr_transT": 0
    }

    # 注意： ability_instance这个代码如果设置seed，那么每一次运行一遍，产生的数据都是一样的！！！（不是一口气产生多个，也不是连续产生的！）
    
    """
    初始化graph_pool_avg：
    1、graph_pool_avg 全图节点嵌入求均值的矩阵：batch，batch*task  （只用初始化一次，按jm场景即可，不用每次都循环）
    2、batch = 1
    """
    va_graph_pool_avg = g_pool_cal(graph_pool_type=configs['neighbor_pooling_type'],  # average的type：1/n_nodes
                                batch_size=1,  # 指定批次的形状。[batch_size, n_j * n_m, n_j * n_m]。每个批次中有batch_size个图,每个图具有n_j * n_m个节点
                                n_nodes=n_total_task,  # task的个数
                                device=device)

    """
    验证数据集：
    1、现在是在外生成好，然后传入数据和当前的index来进行训练
    2、传回的data是dataset (第一维度是samples！！！所有的data都传入)，定义的可以直接.x读取数据
    3、index固定是0-99的变化，为了固定这100个测试样本；data传进来的不一样：随机seed，100个 + 训练样本前100个 + 训练样本后100个
    """
    if data_type == "same_samples":  # 采用同样样本的后100个数据
        va_ability_t = data.t_100_last[data_index]   # task * m
        va_ability_p = data.p_100_last[data_index]
        va_ability_tt = data.transT_100_last[data_index]
        va_edge_info = data.edge_100_last[data_index]
    else:  # 其他的都是直接按顺序的取数据
        va_ability_t = data.t[data_index]
        va_ability_p = data.p[data_index]  # pt加工能耗
        va_ability_tt= data.transT[data_index]
        va_edge_info = data.edge[data_index]

    # 打印出当前的验证集的samples
    if data_index >= 99:
        Logger.log("Evaluation/eval_instance_last", f"va_ability_t={va_ability_t.shape}, va_ability_p={va_ability_p.shape}, va_ability_tt={va_ability_tt.shape}, va_edge_info={np.array(va_edge_info).shape}", print_true=1)  # 
        
    """初始化DGFJSPEnv，反馈state和训练用reward"""
    jsp_instance = np.array([va_ability_t, va_ability_p])  # (记录能力矩阵：加工时间t和加工能耗p) = task * m
    va_env = DisjunctiveGraphJspEnv_singleStep(
                jps_instance=jsp_instance,
                reward_function_parameters=reward_dict,# makespan of the optimal solution for this instance
                default_visualisations=["gantt_console", "graph_console"],
                reward_function='wrk', 
                ability_tr_mm=va_ability_tt, # 运输能力矩阵
                perform_left_shift_if_possible=True,  # 打开左移的机制
                # perform_left_shift_if_possible=False  # 打开左移的机制
                configs=configs
                )

    """
    1、邻居矩阵adj = tasks * tasks
    2、各个task节点的节点特征向量 = tasks * 12 
        返回batch的nparray：adj =（batch，tasks，tasks） + tasks_fea = （batch*tasks， 12）
    3、candidate：batch * job (注意：需要记录的是可选的task的索引值index，为了方便gather提取特征！)
    4、mask：batch * job (注意：原有的是float01的张量，这里转成false和true的张量，mask_value=1来赋值！！)
    5、候选m节点va_machine_fea2 =  env_batch * m * 8 （被调度后）
    
    # TODO 新增task_fea的12特征版本=va_tasks_fea_1101, 整形一个bs=1，shape=（bs*task，12）
    # TODO 当前ENV的reset函数会返回self._state_array() = 其中有变量是对当前样本的ft和pt的idea矩阵（j*m）
    """
    _, _, _, va_adj, _, va_machine_fea2, va_tasks_fea_1101, \
        ft_idea_refer, pt_idea_refer = va_env.reset(Random_weight_type="eval")  # 初始化ft和it的状态 TODO 1108-验证的时候，采用的不是随机权重，而是固定的权重

    
    """++++++++++++++++++++++++++++++++++++++++++
        人为定义理想值：希望fjsp的结果
    +++++++++++++++++++++++++++++++++++++++++++++"""
    #  TODO 计算当前样本的理想值, 就是预估的ft和pt
    idea_mk = np.amax(ft_idea_refer.flatten())  # task个元素，已经被flatten！找最大，就是预估的整体完工时间   amax针对一维数组找最大高效，否则就用通用的mean了
    idea_pt = np.sum(pt_idea_refer.flatten()) / n_total_task # task元素求和，作为当前的真实的能耗输出！（覆盖原值即可），除以task求平均能耗！
    idea_tt, idea_it = 0, 0  # 运输和空闲，期望是0，但是根本不可能
    idea_cost_refer = [idea_mk, idea_pt, idea_tt, idea_it]  # TODO 分母有0，不好搞！要找当前样本的最差值，那不如干脆用MIP的值作为idea基准好了！！！

    """++++++++++++++++++++++++++++++++++++++++++
        初始化S0相关参数：batch=1
    +++++++++++++++++++++++++++++++++++++++++++++"""
    # 整形成batch = 1，防止网络出错
    va_adj = va_adj.reshape(1, va_adj.shape[0], va_adj.shape[1])
    va_tasks_fea_1101 = va_tasks_fea_1101.reshape(1 * va_tasks_fea_1101.shape[0], va_tasks_fea_1101.shape[1])
     
    # 创建pool_task_list和pool_task_dict
    pool_task_list = [1 + n_machine * i for i in range(n_job)] # 每一行的首个task，一直不会变，也不会修改 = [1, 3, 5]
    va_candidate = np.array(pool_task_list) - 1  # 将字典的value转成列表转为np，然后-1作为task的索引值
    va_candidate = va_candidate.reshape(1, -1)  # 转成2维，batch=1
    
    va_mask = torch.zeros(n_job, dtype=torch.float32).cuda().bool()  # tensor([0., 0., 0.], device='cuda:0'),01float张量转成true和false
    va_mask = va_mask.reshape(1, -1)  # 转成2维，batch=1
    
    va_machine_fea2 = va_machine_fea2.reshape(1, n_machine, va_machine_fea2.shape[-1])  # 整形加一个维度batch, bs * m * 8 = 1 * m * 8

    # TODO：初始化的样本数据的t能力的mask_machine，将矩阵中的元素大于等于0的设置为True，小于0的设置为False
    va_mask_machine_batch = va_ability_t >= 0  # t能力矩阵转为bool，shape = task * m 矩阵
    va_mask_machine_batch = torch.tensor(va_mask_machine_batch).to(device)
    va_mask_machine_batch0 = ~va_mask_machine_batch  # mask作用是，true代表不能选，# 将布尔张量取反
    va_mask_machine_batch0 = va_mask_machine_batch0.unsqueeze(0) # # 使用unsqueeze()添加维度，增加第一个维度 = 1*task*m
    
    # Logger.log("Evaluation/state0_init", f"va_adj={va_adj.shape}, va_tasks_fea_1101={va_tasks_fea_1101.shape}, va_candidate={va_candidate.shape}, va_mask={va_mask.shape}, va_machine_fea2={va_machine_fea2.shape}, va_mask_machine_batch0={va_mask_machine_batch0.shape}", print_true=1)  # 

    va_rewards = []
    va_h_mch_pooled = None # 在while之前初始化，只赋值一次！为了保证进入到选择operation的网络时是None的，用可学习的参数；后续都是实际的m的全图特征
    
    while True:
        with torch.no_grad():
            """先选择一个task节点！task_idx = (bs=1,)"""
            va_task_index, va_action_index, _, _, h_g_o_pooled, _ = ppo.job_actor(
                        x_fea=va_tasks_fea_1101,  # TODO task节点的最新的12特征  # bs * task * 12
                        graph_pool_avg=va_graph_pool_avg,  # 全局只有一个
                        padded_nei=None,
                        adj=va_adj,
                        candidate=va_candidate,
                        h_g_m_pooled=va_h_mch_pooled,  # 这个m节点的全图池化均值 ！= 选择o的动作啊（选m是按顺序，选o是按概率，不对应先不用）
                        mask_operation=va_mask,
                        use_greedy=greedy  # validate的时候需要使用greedy，只选择最大max概率的！!!!!!!!!!!
                        )

            # TODO: 选择machine的mask, bs_task_m 选择变为 batch_1_m（扩维然后expand, batch=1 = 1 * 1 * m
            va_mask_machine_batch_ = torch.gather(va_mask_machine_batch0,   # 1*task*m
                                                  1, # dim=1,维度2
                                                  va_task_index.unsqueeze(-1).unsqueeze(-1).expand(va_mask_machine_batch0.size(0), -1, va_mask_machine_batch0.size(2)))  # 每一批的样本都不会对mask有影响，只有job选择不同的task，gather之后的mask不一样，传入到machine网络中！
            

            # TODO： 新增当前task的对应可选m的节点特征 = [Im=mask, t, deltaP*T, delta_transT, p, edge] = bs * m * 6
            va_m_fea1 = va_cal_cur_task_machine_feature(va_task_index, va_mask_machine_batch_, va_tasks_fea_1101,
                                                        va_ability_t, va_ability_p, va_ability_tt, va_edge_info)
            
            # 返回的是batch * m的选择m的概率 + 返回所有m节点的全局嵌入（求均值）：batch * hidden_dim
            va_mch_prob, va_h_mch_pooled, _ = ppo.machine_actor_gcn(machine_fea_1=va_m_fea1,   # 当前task的m的候选特征，作为自注意力的目标节点
                                                                    machine_fea_2=va_machine_fea2,
                                                                    h_pooled_o=h_g_o_pooled,
                                                                    machine_mask=va_mask_machine_batch_)  # 选择m的action，选择概率最高的！
            
        """选择m_id"""
        if greedy:
            va_action_id, _ = greedy_select_machine_action(va_mch_prob)
        else:
            va_action_id, _ = select_machine_action(va_mch_prob)

        """进行task和machine的step"""
        # env  step
        va_joint_actions = [va_task_index.data.item(), va_action_id.data.item()] # 先把张量的数值提取出来，task_index和m_idx
        _, o_r, o_done, _, rmk, ridle, renergy_m, renergy_transM, _, _, \
            va_adj_, _, va_machine_fea2_, va_tasks_fea_1101_ = va_env.step(joint_action=va_joint_actions)  # 传入的都要是从0开始的！！action = task_id - 1 TODO 改名输出了m_fea2_ + 新增了task_1101_的输出

        """更新选择task节点之后的candidate和mask,(self, paralenv, action_batch, mask_value)"""
        """
        1、candidate：batch * job (注意：需要记录的是可选的task的索引值，为了方便gather提取特征！)
        2、mask：batch * job (注意：原有的是float01的张量，这里转成false和true的张量，mask_value=1来赋值！！)
        """
        va_candidate_, va_mask_ = ppo.Eval_esa_update_chosenTaskID_CandidateTaskIDx_JobMask(
                    env=va_env, 
                    action_batch=va_action_index,
                    mask_value=1)  # 更新剩余可选任务（每行=每个job）+ 显示对应的task的ID
        va_candidate_ = va_candidate_.reshape(1, -1)  # 转成2维，batch=1
        va_mask_ = va_mask_.reshape(1, -1)  # 转成2维，batch=1

        # TODO：判断是否会出现选择的t是负数
        if va_ability_t[va_task_index.data.item()][va_action_id.data.item()] < 0:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"=============test Minus: [{va_task_index.data.item()},{va_action_id.data.item()}], t={va_ability_t[va_task_index.data.item()][va_action_id.data.item()]}, p= {va_ability_p[va_task_index.data.item()][va_action_id.data.item()]}")

        # 整形成batch = 1，防止网络出错
        va_adj_ = va_adj_.reshape(1, va_adj_.shape[0], va_adj_.shape[1])
        va_tasks_fea_1101_ = va_tasks_fea_1101_.reshape(1 * va_tasks_fea_1101_.shape[0], va_tasks_fea_1101_.shape[1])
        va_machine_fea2_ = va_machine_fea2_.reshape(1, n_machine, va_machine_fea2_.shape[-1])  # 整形加一个维度batch

        """
        记录选择operation的累加Gt, while中记录所有step，直到done
        """
        cost_dict_cumsum["opr_Gt"] += o_r  # 带权重加权和之后的
        cost_dict_cumsum["opr_mk"] += rmk  # 不带权重的
        cost_dict_cumsum["opr_idleT"] += ridle
        cost_dict_cumsum["opr_pt"] += renergy_m
        cost_dict_cumsum["opr_transT"] += renergy_transM

        """
        实时更新下一次的 state 变量, va_fx不用变化
        """
        va_adj = va_adj_
        va_tasks_fea_1101 = va_tasks_fea_1101_  # TODO 新的task_fea_记得重新赋值
        va_candidate = va_candidate_
        va_mask = va_mask_
        va_machine_fea2 = va_machine_fea2_   # TODO m_fea2_新名称，注意改写！

        """现在只关注DGFJSPEnv的done + r + 4指标"""
        if o_done:

            """至今为止：一定要在done之后reset之前读取：全选完的当前this step的cost，没有权重，不用累加，不用相减"""
            """ 
            ! TODO 
            xxx_previous_step不会清0，reset的时候会变成理想预估初值，所以累加就不是=真实值，因为初始xxx_previous_step！=0 ！！！！
            这里done之后选这个变量，this_step就是计算的mk真值，pt累加，transT累加和idleT累加）
            """
            mk = va_env.makespan_previous_step  # 上次的makespan
            pt = va_env.total_e1_previous_step / n_total_task  # 上次的加工能耗之和（已分配设备的） TODO 别忘了这里是平均能耗！
            transT = va_env.trans_t_previous_step  # 至今为止的运输时间t （暂时没有乘以运输设备的e）
            idleT = va_env.idle_t_previous_step  # 至今为止的空闲时间之和
            Final_4cost = [mk, pt, transT, idleT]  # 轨迹走完的最终指标 = cost
            
            w1 = configs['weight_mk']
            w2 = configs['weight_ec']
            w3 = configs['weight_tt']

            Objective =  w1 * mk + w2 * (pt + idleT) +  w3 * transT

            if gantt_flag:  # validate的画图版本
                va_env.render()
                Logger.log("Evaluation/done", f"Objective={Objective}, mk={mk}, pt={pt}, transT={transT}, idleT={idleT}", print_true=1)  # eval的结果

            ppo.set_to_0(None)  # 清0，while的done已经清0，这里也清0，不影响
            va_env.reset()
            break
    
    # print("********************************* Validate Done *****************************************")
    # TODO 新增 = 更新4个cost的指标变成相对的差值的占比！[mk, pt, transT, idleT] ！！！分母有0，不行！！！
    # new_related_cost = (np.array(Final_4cost) - np.array(idea_cost_refer)) / np.array(idea_cost_refer)

    return cost_dict_cumsum, Final_4cost, Objective


def va_cal_cur_task_machine_feature(task_index, m_mask, all_task_fea, t, p, tt, edge):
    """

    :param task_index:  （张量）bs个元素，选择的task的index
    :param m_mask:  （张量）shape = bs_1_m, 当前task的对应的machine的mask，F可选，T是负数
    :param all_task_fea:  （np.array）（bs*task, 12）  上一次ENV更新之后的所有task的特征值 =  [Io=mask, Est, Eft, Ept, j_id, m_id, t, p, n_in_edge] = [可参考依据 + 被选之后的状态] 12维度
    :param t:  （np.array）（task, m）  t能力矩阵
    :param p:  （np.array）（task, m）  p能力矩阵
    :param tt:  （np.array）（m, m）  tt能力矩阵
    :param edge:  （np.array）[[0,1],[2,3]]  分属边
    :return:
    """
    batch_size = 1 # validate和test的时候，是一个样本一个样本来的，没有bs
    nmachines = m_mask.shape[-1]

    m_feas = np.zeros((batch_size, nmachines, 6)) # 初始化特征矩阵
    all_task_fea = all_task_fea.reshape(batch_size, -1, all_task_fea.shape[-1])  # bs * task * 12

    task_index = task_index.cpu().numpy()  # 先转成array数组
    m_mask = m_mask.cpu().numpy()  # 先转成array数组
    
    for i in range(batch_size):  # bs=1,只会跑一遍！
        t_ins = t  # task*m
        p_ins = p
        tt_ins = tt  # m*m
        edge_ins = edge  # 二维[[0,1],[2,3]] 2边均分m

        pt_ins = np.multiply(t_ins, np.abs(p_ins))  # task * m  P*T 有正负！

        """对那些不能选择的m，能力值用mean表示，防止0无意义的参数"""
        # 找到不为0的元素并计算均值
        over_zero_elements_t = t_ins[task_index[i]][t_ins[task_index[i]] > 0]
        mean_t = np.mean(over_zero_elements_t)  # 当前所选task的可以做的m的时间t均值

        over_zero_elements_pt = pt_ins[task_index[i]][pt_ins[task_index[i]] > 0]
        mean_pt = np.mean(over_zero_elements_pt)  # 当前所选task的可以做的m的能耗p*t均值

        over_zero_elements_p = p_ins[task_index[i]][p_ins[task_index[i]] > 0]
        mean_p = np.mean(over_zero_elements_p)  # 当前所选task的可以做的m的功率p均值

        
        """
        候选m的节点特征（bs，m，6）= 做当前task的<能力t, 能力pt, 所需transt, 可选Im, 能力p, 属于edge>(有则真，无则估计mean)
        """
        for m_index in range(nmachines):  # TODO 不是累加值，是可依据判断值，所有的m的所有的fea都要更新
            
            m_feas[i][m_index][0] = t_ins[task_index[i]][m_index] if t_ins[task_index[i]][m_index] > 0 else mean_t  # ！TODO 1 候选m节点的能力t，不能做用能做m的mean值代替

            # TODO 虽然最后会有mask屏蔽掉，但是这些都输入到网络，还是有点影响的吧？能选的m是真实值，不能选的就用能选的mean来代替！（用minus可能会NaN）
            m_feas[i][m_index][1] = pt_ins[task_index[i]][m_index] if pt_ins[task_index[i]][m_index] > 0 else mean_pt  # ！TODO 2 候选m节点的能力pt，不能做用能做m的mean值代替

            if task_index[i] % nmachines == 0:  # action从0开始，表明是每个job的首位，不会有运输
                new_avail_transT = 0
            else:  # 既然选到了中间位置的task，其前置task必定被选，不然不满足先后工序！！！
                # TODO 上一个task 和 当前task，是否会有运输时间！函数中会始终判断都是同一个job的！！！ 前置必然被选，此时的运输就是新增的运输时间
                """
                注意：
                1、task_fea里边存的是m_id，从1开始的，要转成从0开始（task_fea是在第6个元素！！！）
                2、task_fea= bs*task*12，task是按照顺序的，此时是在job的中间task，所以上一个task就是再同一个job里！
                3、task_fea中没有被调度的节点，m_id=0！此时查表就不对了？（放心，我们查找的是上一个task的m，和当前task候选m之间的运输时间，必存在）
                """
                new_avail_transT = tt_ins[int(all_task_fea[i][task_index[i] - 1][5]) - 1][m_index]  # TODO 同job前一个task的m的index，和当前遍历的m_index,查表就知道选了当前m会有多少的运输时间新增
            m_feas[i][m_index][2] = new_avail_transT   # 新增的运输时间    TODO 3 选择当前候选m所需的transT

            m_feas[i][m_index][3] = 1 - int(m_mask[i][0][m_index])  # 注意：mask中true=1是不能干！所以取反，当前m节点是否能选，bool转成01   TODO 4 可选~mask

            m_feas[i][m_index][4] = p_ins[task_index[i]][m_index] if p_ins[task_index[i]][m_index] > 0 else mean_p  # ！ TODO 5 候选m节点对应task的p能力，不能干用mean  
            m_feas[i][m_index][5] = np.where(edge_ins == m_index)[0][0] + 1  # 找到m节点在edge_ins的索引，返回元组，包含两个数组，行和列数组，取行数组的第一个满足条件的元素的index， +1 作为id，从1开始的边  TODO 6 属于edge

    return m_feas  # bs * m * 6   np.array



"""
ESWA的方法：主要是换了job和machine的网络！其他完全一样
为了不影响原程序，直接复制新的
"""
def esa_validate_cost_gcn_jointActor_GAT(ppo, gantt_flag, data, data_index, data_type, greedy=True, args=None):
    """
    1、传入ppo是因为当前训练的网络的参数都在此中，验证时要看当前的网络的效果
    加载选择m的环境
    2、dataloader里边已经是按照env_batch进行拆分的样本数据了
    """
    configs = args
    n_job = configs['n_job']
    n_machine = configs['n_machine']
    n_total_task = n_job * n_machine
    n_edge = configs['n_edge']
    mask_value = configs['mask_value']
    reward_dict = configs['reward_scaling']
    m_scaling = configs['m_scaling']

    #记录当前轨迹的各指标累加cost
    cost_dict_cumsum = {
        "opr_Gt": 0,
        "opr_mk": 0,
        "opr_idleT": 0,
        "opr_pt": 0,
        "opr_transT": 0
    }

    # 注意： ability_instance这个代码如果设置seed，那么每一次运行一遍，产生的数据都是一样的！！！（不是一口气产生多个，也不是连续产生的！）
    
    """
    初始化graph_pool_avg：
    1、graph_pool_avg 全图节点嵌入求均值的矩阵：batch，batch*task  （只用初始化一次，按jm场景即可，不用每次都循环）
    2、batch = 1
    """
    va_graph_pool_avg = g_pool_cal(graph_pool_type=configs['neighbor_pooling_type'],  # average的type：1/n_nodes
                                batch_size=1,  # 指定批次的形状。[batch_size, n_j * n_m, n_j * n_m]。每个批次中有batch_size个图,每个图具有n_j * n_m个节点
                                n_nodes=n_total_task,  # task的个数
                                device=device)

    """
    验证数据集：
    1、现在是在外生成好，然后传入数据和当前的index来进行训练
    2、传回的data是dataset (第一维度是samples！！！所有的data都传入)，定义的可以直接.x读取数据
    3、index固定是0-99的变化，为了固定这100个测试样本；data传进来的不一样：随机seed，100个 + 训练样本前100个 + 训练样本后100个
    """
    if data_type == "same_samples":  # 采用同样样本的后100个数据
        va_ability_t = data.t_100_last[data_index]   # task * m
        va_ability_p = data.p_100_last[data_index]
        va_ability_tt = data.transT_100_last[data_index]
        va_edge_info = data.edge_100_last[data_index]
    else:  # 其他的都是直接按顺序的取数据
        va_ability_t = data.t[data_index]
        va_ability_p = data.p[data_index]  # pt加工能耗
        va_ability_tt= data.transT[data_index]
        va_edge_info = data.edge[data_index]

    # 打印出当前的验证集的samples
    if data_index >= 99:
        Logger.log("Evaluation/eval_instance_last", f"va_ability_t={va_ability_t.shape}, va_ability_p={va_ability_p.shape}, va_ability_tt={va_ability_tt.shape}, va_edge_info={np.array(va_edge_info).shape}", print_true=1)  # 
        
    """初始化DGFJSPEnv，反馈state和训练用reward"""
    jsp_instance = np.array([va_ability_t, va_ability_p])  # (记录能力矩阵：加工时间t和加工能耗p) = task * m
    va_env = DisjunctiveGraphJspEnv_singleStep(
                jps_instance=jsp_instance,
                reward_function_parameters=reward_dict,# makespan of the optimal solution for this instance
                default_visualisations=["gantt_console", "graph_console"],
                reward_function='wrk', 
                ability_tr_mm=va_ability_tt, # 运输能力矩阵
                perform_left_shift_if_possible=True,  # 打开左移的机制
                # perform_left_shift_if_possible=False  # 打开左移的机制
                configs=configs
                )

    """
    1、邻居矩阵adj = tasks * tasks
    2、各个task节点的节点特征向量 = tasks * 12 
        返回batch的nparray：adj =（batch，tasks，tasks） + tasks_fea = （batch*tasks， 12）
    3、candidate：batch * job (注意：需要记录的是可选的task的索引值index，为了方便gather提取特征！)
    4、mask：batch * job (注意：原有的是float01的张量，这里转成false和true的张量，mask_value=1来赋值！！)
    5、候选m节点va_machine_fea2 =  env_batch * m * 8 （被调度后）
    
    # TODO 新增task_fea的12特征版本=va_tasks_fea_1101, 整形一个bs=1，shape=（bs*task，12）
    # TODO 当前ENV的reset函数会返回self._state_array() = 其中有变量是对当前样本的ft和pt的idea矩阵（j*m）
    """
    _, _, _, va_adj, _, va_machine_fea2, va_tasks_fea_1101, \
        ft_idea_refer, pt_idea_refer = va_env.reset(Random_weight_type="eval")  # 初始化ft和it的状态 TODO 1108-验证的时候，采用的不是随机权重，而是固定的权重

    
    """++++++++++++++++++++++++++++++++++++++++++
        人为定义理想值：希望fjsp的结果
    +++++++++++++++++++++++++++++++++++++++++++++"""
    #  TODO 计算当前样本的理想值, 就是预估的ft和pt
    idea_mk = np.amax(ft_idea_refer.flatten())  # task个元素，已经被flatten！找最大，就是预估的整体完工时间   amax针对一维数组找最大高效，否则就用通用的mean了
    idea_pt = np.sum(pt_idea_refer.flatten()) / n_total_task # task元素求和，作为当前的真实的能耗输出！（覆盖原值即可），除以task求平均能耗！
    idea_tt, idea_it = 0, 0  # 运输和空闲，期望是0，但是根本不可能
    idea_cost_refer = [idea_mk, idea_pt, idea_tt, idea_it]  # TODO 分母有0，不好搞！要找当前样本的最差值，那不如干脆用MIP的值作为idea基准好了！！！

    """++++++++++++++++++++++++++++++++++++++++++
        初始化S0相关参数：batch=1
    +++++++++++++++++++++++++++++++++++++++++++++"""
    # 整形成batch = 1，防止网络出错
    va_adj = va_adj.reshape(1, va_adj.shape[0], va_adj.shape[1])
    va_tasks_fea_1101 = va_tasks_fea_1101.reshape(1 * va_tasks_fea_1101.shape[0], va_tasks_fea_1101.shape[1])
     
    # 创建pool_task_list和pool_task_dict
    pool_task_list = [1 + n_machine * i for i in range(n_job)] # 每一行的首个task，一直不会变，也不会修改 = [1, 3, 5]
    va_candidate = np.array(pool_task_list) - 1  # 将字典的value转成列表转为np，然后-1作为task的索引值
    va_candidate = va_candidate.reshape(1, -1)  # 转成2维，batch=1
    
    va_mask = torch.zeros(n_job, dtype=torch.float32).cuda().bool()  # tensor([0., 0., 0.], device='cuda:0'),01float张量转成true和false
    va_mask = va_mask.reshape(1, -1)  # 转成2维，batch=1
    
    va_machine_fea2 = va_machine_fea2.reshape(1, n_machine, va_machine_fea2.shape[-1])  # 整形加一个维度batch, bs * m * 8 = 1 * m * 8

    # TODO：初始化的样本数据的t能力的mask_machine，将矩阵中的元素大于等于0的设置为True，小于0的设置为False
    va_mask_machine_batch = va_ability_t >= 0  # t能力矩阵转为bool，shape = task * m 矩阵
    va_mask_machine_batch = torch.tensor(va_mask_machine_batch).to(device)
    va_mask_machine_batch0 = ~va_mask_machine_batch  # mask作用是，true代表不能选，# 将布尔张量取反
    va_mask_machine_batch0 = va_mask_machine_batch0.unsqueeze(0) # # 使用unsqueeze()添加维度，增加第一个维度 = 1*task*m
    
    # Logger.log("Evaluation/state0_init", f"va_adj={va_adj.shape}, va_tasks_fea_1101={va_tasks_fea_1101.shape}, va_candidate={va_candidate.shape}, va_mask={va_mask.shape}, va_machine_fea2={va_machine_fea2.shape}, va_mask_machine_batch0={va_mask_machine_batch0.shape}", print_true=1)  # 

    va_rewards = []
    va_h_mch_pooled = None # 在while之前初始化，只赋值一次！为了保证进入到选择operation的网络时是None的，用可学习的参数；后续都是实际的m的全图特征
    
    while True:
        with torch.no_grad():
            """先选择一个task节点！task_idx = (bs=1,)"""
            va_task_index, va_action_index, _, _, h_g_o_pooled, _ = ppo.esa_job_actor(
                        x_fea=va_tasks_fea_1101,  # TODO task节点的最新的12特征  # bs * task * 12
                        graph_pool_avg=va_graph_pool_avg,  # 全局只有一个
                        padded_nei=None,
                        adj=va_adj,
                        candidate=va_candidate,
                        h_g_m_pooled=va_h_mch_pooled,  # 这个m节点的全图池化均值 ！= 选择o的动作啊（选m是按顺序，选o是按概率，不对应先不用）
                        mask_operation=va_mask,
                        use_greedy=greedy  # validate的时候需要使用greedy，只选择最大max概率的！!!!!!!!!!!
                        )
            
            # TODO: 选择machine的mask, bs_task_m 选择变为 batch_1_m（扩维然后expand, batch=1 = 1 * 1 * m
            va_mask_machine_batch_ = torch.gather(va_mask_machine_batch0,   # 1*task*m
                                                  1, # dim=1,维度2
                                                  va_task_index.unsqueeze(-1).unsqueeze(-1).expand(va_mask_machine_batch0.size(0), -1, va_mask_machine_batch0.size(2)))  # 每一批的样本都不会对mask有影响，只有job选择不同的task，gather之后的mask不一样，传入到machine网络中！
            

            # TODO： 新增当前task的对应可选m的节点特征 = [Im=mask, t, deltaP*T, delta_transT, p, edge] = bs * m * 6
            va_m_fea1 = va_cal_cur_task_machine_feature(va_task_index, va_mask_machine_batch_, va_tasks_fea_1101,
                                                        va_ability_t, va_ability_p, va_ability_tt, va_edge_info)
            
            # 返回的是batch * m的选择m的概率 + 返回所有m节点的全局嵌入（求均值）：batch * hidden_dim
            va_mch_prob, va_h_mch_pooled, _ = ppo.esa_machine_actor_gcn(machine_fea_1=va_m_fea1,   # 当前task的m的候选特征，作为自注意力的目标节点
                                                                    machine_fea_2=va_machine_fea2,
                                                                    h_pooled_o=h_g_o_pooled,
                                                                    machine_mask=va_mask_machine_batch_)  # 选择m的action，选择概率最高的！
            
        """选择m_id"""
        if greedy:
            va_action_id, _ = greedy_select_machine_action(va_mch_prob)
        else:
            va_action_id, _ = select_machine_action(va_mch_prob)

        """进行task和machine的step"""
        # env  step
        va_joint_actions = [va_task_index.data.item(), va_action_id.data.item()] # 先把张量的数值提取出来，task_index和m_idx
        _, o_r, o_done, _, rmk, ridle, renergy_m, renergy_transM, _, _, \
            va_adj_, _, va_machine_fea2_, va_tasks_fea_1101_ = va_env.step(joint_action=va_joint_actions)  # 传入的都要是从0开始的！！action = task_id - 1 TODO 改名输出了m_fea2_ + 新增了task_1101_的输出

        """更新选择task节点之后的candidate和mask,(self, paralenv, action_batch, mask_value)"""
        """
        1、candidate：batch * job (注意：需要记录的是可选的task的索引值，为了方便gather提取特征！)
        2、mask：batch * job (注意：原有的是float01的张量，这里转成false和true的张量，mask_value=1来赋值！！)
        """
        va_candidate_, va_mask_ = ppo.Eval_esa_update_chosenTaskID_CandidateTaskIDx_JobMask(
                    env=va_env, 
                    action_batch=va_action_index,
                    mask_value=1)  # 更新剩余可选任务（每行=每个job）+ 显示对应的task的ID
        va_candidate_ = va_candidate_.reshape(1, -1)  # 转成2维，batch=1
        va_mask_ = va_mask_.reshape(1, -1)  # 转成2维，batch=1

        # TODO：判断是否会出现选择的t是负数
        if va_ability_t[va_task_index.data.item()][va_action_id.data.item()] < 0:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"=============test Minus: [{va_task_index.data.item()},{va_action_id.data.item()}], t={va_ability_t[va_task_index.data.item()][va_action_id.data.item()]}, p= {va_ability_p[va_task_index.data.item()][va_action_id.data.item()]}")

        # 整形成batch = 1，防止网络出错
        va_adj_ = va_adj_.reshape(1, va_adj_.shape[0], va_adj_.shape[1])
        va_tasks_fea_1101_ = va_tasks_fea_1101_.reshape(1 * va_tasks_fea_1101_.shape[0], va_tasks_fea_1101_.shape[1])
        va_machine_fea2_ = va_machine_fea2_.reshape(1, n_machine, va_machine_fea2_.shape[-1])  # 整形加一个维度batch

        """
        记录选择operation的累加Gt, while中记录所有step，直到done
        """
        cost_dict_cumsum["opr_Gt"] += o_r  # 带权重加权和之后的
        cost_dict_cumsum["opr_mk"] += rmk  # 不带权重的
        cost_dict_cumsum["opr_idleT"] += ridle
        cost_dict_cumsum["opr_pt"] += renergy_m
        cost_dict_cumsum["opr_transT"] += renergy_transM

        """
        实时更新下一次的 state 变量, va_fx不用变化
        """
        va_adj = va_adj_
        va_tasks_fea_1101 = va_tasks_fea_1101_  # TODO 新的task_fea_记得重新赋值
        va_candidate = va_candidate_
        va_mask = va_mask_
        va_machine_fea2 = va_machine_fea2_   # TODO m_fea2_新名称，注意改写！

        """现在只关注DGFJSPEnv的done + r + 4指标"""
        if o_done:

            """至今为止：一定要在done之后reset之前读取：全选完的当前this step的cost，没有权重，不用累加，不用相减"""
            """ 
            ! TODO 
            xxx_previous_step不会清0，reset的时候会变成理想预估初值，所以累加就不是=真实值，因为初始xxx_previous_step！=0 ！！！！
            这里done之后选这个变量，this_step就是计算的mk真值，pt累加，transT累加和idleT累加）
            """
            mk = va_env.makespan_previous_step  # 上次的makespan
            pt = va_env.total_e1_previous_step / n_total_task  # 上次的加工能耗之和（已分配设备的） TODO 别忘了这里是平均能耗！
            transT = va_env.trans_t_previous_step  # 至今为止的运输时间t （暂时没有乘以运输设备的e）
            idleT = va_env.idle_t_previous_step  # 至今为止的空闲时间之和
            Final_4cost = [mk, pt, transT, idleT]  # 轨迹走完的最终指标 = cost
            
            w1 = configs['weight_mk']
            w2 = configs['weight_ec']
            w3 = configs['weight_tt']

            Objective =  w1 * mk + w2 * (pt + idleT) +  w3 * transT

            if gantt_flag:  # validate的画图版本
                va_env.render()
                Logger.log("Evaluation/done", f"Objective={Objective}, mk={mk}, pt={pt}, transT={transT}, idleT={idleT}", print_true=1)  # eval的结果

            ppo.set_to_0(None)  # 清0，while的done已经清0，这里也清0，不影响
            va_env.reset()
            break
    
    # print("********************************* Validate Done *****************************************")
    # TODO 新增 = 更新4个cost的指标变成相对的差值的占比！[mk, pt, transT, idleT] ！！！分母有0，不行！！！
    # new_related_cost = (np.array(Final_4cost) - np.array(idea_cost_refer)) / np.array(idea_cost_refer)

    return cost_dict_cumsum, Final_4cost, Objective
