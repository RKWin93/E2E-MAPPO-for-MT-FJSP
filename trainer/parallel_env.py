import numpy as np
import random
import copy
from instance.generate_allsize_mofjsp_dataset import Logger
from algorithm.ppo_trick import RewardScaling
# ! TODO (LLM) root@01a43040a453:/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/graph-jsp-env# pip install -e .  (需要在新的根目录安装下env的环境)- https://github.com/RKWin93/graph-jsp-env
from graph_jsp_env.disjunctive_graph_jsp_env_singlestep import DisjunctiveGraphJspEnv_singleStep

"""
新建Env的并行运行类：（batch_siaze， x）堆叠
1、初始化batch_size个ENV
2、根据actor网络的输出的action在并行的Env中输出state，并堆叠在一起
3、输出reward并堆叠，用于更新网络

记得清0！！！
"""
# batch_size, njobs, nmachines, nedges

class Parallel_env(object):
    def __init__(self, args):  # 传入arg参数，已经转成dict形式了
        
        self.njobs = args['n_job']
        self.nmachines = args['n_machine']
        self.ntasks = self.njobs * self.nmachines
        self.nedges = args['n_edge']
        self.batch_size = args['env_batch']
        self.m_scaling = args['m_scaling'] # m选择中的reward放缩比例
        self.reward_dict = args['reward_scaling'] # DG图中direction选择中的reward放缩比例

        self.args = args  # 为了方便传递参数设置

        self.ability_instance = []  # 记录并行的ability instance的所有信息
        # TODO 新增针对不同样本的reward——scaling
        self.paral_Rscaling_instance = [] # 记录针对同一样本中的放缩类，相同样本运行多少次，都不会清0，除非换样本
        self.paral_env_DG = []  # 记录并行的DG的环境
        self.oenv_info = []  # 记录每一个batch的reward, done, r_t, r_idle, r_energy_m, r_energy_transM
        
    # 采样Batch_size个，然后赋值self.ability_instance（不再清0和变化！）
    def get_batch(self, dataset_dict): # dataloader的返回数据是我在__getitem__中定义的字典
        """
        初始化：获取每一个batch的样本数据（只会运行一次！）
        1、dataset_dict是一个dict，其中有4个key（t,p,tranT,edge）,每个key中元素是一个大的tensor：shape = env_batch * task*m（m*m）（edge_num * m/edge_num(均分)）
        2、t_batch里边储存的已经是env_batch个元素了
        """
        Logger.log("Training/Parallel_env/get_batch_scenario", f"job={self.njobs}, machine={self.nmachines}, edge={self.nedges}, tasks={ self.ntasks}", print_true=1)
        
        t_batch = dataset_dict["t"].numpy() # 先转成array数组  # bs * task * m
        p_batch = dataset_dict["p"].numpy() 
        transT_batch = dataset_dict["transT"].numpy() 
        edge_batch = dataset_dict["edge"].numpy() 

        self.ability_instance = [] # 进来此函数，表示已经换了样本，否则不会进来！！！重头开始APPEND！！

        for i_batch in range(self.batch_size):
            
            # t_batch里边储存的已经是env_batch个元素了
            instance = [copy.deepcopy(t_batch[i_batch]),
                        copy.deepcopy(p_batch[i_batch]),
                        copy.deepcopy(transT_batch[i_batch]),
                        copy.deepcopy(edge_batch[i_batch])]  # 一次采样的实例信息存到一个list中,元素为np.array
            self.ability_instance.append(copy.deepcopy(instance))  # 记录在总的列表中， batch_size个instance，每个instance有4元素  (记录的是数据，不是变量参数名字)

        Logger.log("Training/Parallel_env/get_batch_info", f"self.ability_instance={np.array(self.ability_instance).shape}, t[0].shape={self.ability_instance[0][0].shape}, p[0].shape={self.ability_instance[0][1].shape}, transT[0].shape={self.ability_instance[0][2].shape}", print_true=1)   # bs=1的样本数据 TODO tensor张量用size，array数组用shape，list列表用len（看大小）
    
    """
    reward动态放缩：
    1、每个env也就是batch都会初始化放缩系数
    2、# TODO 参考样本数据，此reset不清零，新样本重新赋值来替代
    """
    def init_RewardScaling_sameBATCH(self, shape):

        self.paral_Rscaling_instance = []
        for _ in range(self.batch_size):
            # TODO 初始化reward动态放缩的class(类中也会初始化RunningMeanStd类，其中self.n在整个episode跑完中都不会重新reset，反应更多样本数据)
            """
            1、原版的self.n只在重新运行main的时候初始化。
                原因是：这个游戏env相当于是不变的，在这个env中不断训练，我现在为了增加泛化性，换了一个厂就跑不好了，所以env的配置会不断初始化！样本数据都不一样的！
            2、我现在：相同场景的env是不会初始化self.n的，除非你换了env，那reward铁定不一样了啊！

            所以：我在更换样本数据这里进行动态计算mean和std的self.n的初始化 + 同时考虑env_bs的影响，同时初始化多个实例，针对不同来运行！！！！
            """
            Rscaling = RewardScaling(shape=shape, gamma=self.args['GAMMA'])  # TODO 每次的episode只会重新reset这个RewardScaling类中的self.R（记录当前episode的累加误差）！！
            self.paral_Rscaling_instance.append(copy.deepcopy(Rscaling)) # 不同样本的不同类，env_bs *

    #     不用return，可以直接读取的
    
    def init_DGFJSPEnv_state0(self):
        """
        初始化并行Env：
        1、adj_block = np.zeros((self.batch_size*self.ntasks, self.batch_size*self.ntasks))  # 初始化一个全0矩阵，用来存放所有env的adj
        """
        self.paral_env_DG = []  # 初始直接清0，防错
        adj_batch = []  # 初始化一个list，用来存放所有env的adj（注意：最终返回一个np.array就好，到时候在forward里边会转成tensor + 大的对角矩阵的！）
        tasks_fea_batch = [] # 记录每一个env的节点特征向量 = 12元素
        machine_fea_batch = [] # 记录machine的节点特征向量 = 8元素（被调度之后）
        for ii_batch in range(self.batch_size):
            """
            加载选择0的环境
            instance = [copy.deepcopy(t_batch[i_batch]),
                        copy.deepcopy(p_batch[i_batch]),
                        copy.deepcopy(transT_batch[i_batch]),
                        copy.deepcopy(edge_batch[i_batch])]  # 一次采样的实例信息存到一个list中,元素为np.array
            self.ability_instance.append(copy.deepcopy(instance))  # 记录在总的列表中，大小 1 * batch_size  (记录的是数据，不是变量参数名字)
            
            1、jps_instance = 2*tak*m (记录能力矩阵：加工时间t和加工能耗p)
            2、ability_tr_mm = m*m 运输能力矩阵
            """
            jsp_instance = np.array([self.ability_instance[ii_batch][0], self.ability_instance[ii_batch][1]]) # (记录能力矩阵：加工时间t和加工能耗p)
            # TODO 虽然有负数，初始化的进去，内部处理的时候会有判断的，问题不大！
            env = DisjunctiveGraphJspEnv_singleStep(jps_instance=jsp_instance,
                                                    reward_function_parameters=self.reward_dict,  # reward的放缩系数，改为全局动态放缩了
                                                    default_visualisations=["gantt_console", "graph_console"],
                                                    reward_function='wrk', 
                                                    ability_tr_mm=self.ability_instance[ii_batch][2],  # 运输能力矩阵
                                                    perform_left_shift_if_possible=True,  # 打开左移的机制
                                                    # perform_left_shift_if_possible=False  # 打开左移的机制
                                                    configs=self.args
                                                    )

            self.paral_env_DG.append(copy.deepcopy(env))  # 记录选择operation的env

            """
            都是单个env的反馈，np.array
                1、邻居矩阵adj = tasks * tasks  （运输时间（if）+空闲时间（if）  or  =1 有边！）
                2、各个节点的节点特征向量 = tasks * 12 （预估ST + 预估FT + 预估PT + if被调度I + 入边的个数in_dedge_n + 分配给m_id + m的t能力 +m的p能力 + 归属作业j_id + 固定随机权重*3）
                3、被调度的machine节点特征向量 = m_num* 8 （被调度FT_last_task + 被调度sumPT/task + 被调度sumTransT + 被调度sumIdleT + 同一m的被选次数的累加sumIm + 固定随机权重*3）
            
            返回batch的nparray：adj =（batch，tasks，tasks） + tasks_fea = （batch*tasks， 12） + machine_fea =（env_batch * m * 8 ）
            """
            _, _, _, adj, _, machine_fea, tasks_fea, *_ = self.paral_env_DG[-1].reset()  # reset写好了，直接初始化任务和设备节点的状态(-1=最新的ENV)
            
            adj_batch.append(copy.deepcopy(adj))
            tasks_fea_batch.append(copy.deepcopy(tasks_fea))  # 每一个env的节点特征矩阵存到list
            machine_fea_batch.append(copy.deepcopy(machine_fea))

        adj_batch = np.array(adj_batch)  # 转成np.arry  =（batch，tasks，tasks）
        tasks_fea_batch = np.concatenate(tasks_fea_batch, axis=0)  # 在第一维度进行合并, 转成np.arry = （batch*tasks， 12）
        machine_fea_batch = np.array(machine_fea_batch)  # m节点的特征向量 env_batch * m * 8
        
        # Logger.log("Training/init_states_0", f"Shape: adj_batch={adj_batch.shape}, tasks_fea_batch={tasks_fea_batch.shape}, machine_fea_batch={machine_fea_batch.shape}", print_true=1)
        
        return adj_batch, machine_fea_batch, tasks_fea_batch
    
    
    """
    我需要根据当前的选择的task，给出所有 m*6 的machine节点的可以参考的state
    1、一贯思路：平行的env，循环batch来生成bs*m*8的特征值
    2、self.ability_instance = batch个4能力矩阵，已经是np.array了
    
    当前task的对应可选（候选）m的节点特征 =[Im=mask, t, P, P*T, transT_need, edge] = bs * m * 6 
    """
    def cal_cur_task_machine_feature(self, task_index, m_mask, all_task_fea):
        """
        :param task_index:  （张量）bs个元素，选择的task的index
        :param m_mask:  （张量）shape = bs_1_m, 当前task的对应的machine的mask，F可选，T是负数
        :param all_task_fea:  （np.array）（bs*task, 12） = 当前ENV更新之后的所有task的特征值 =  [Io=mask, Est, Eft, Ept, j_id, m_id, t, p, n_in_edge] = [可参考依据 + 被选之后的状态] 9维度  + 动态权重
        :return:
        """
        m_feas = np.zeros((self.batch_size, self.nmachines, 6)) # 初始化特征矩阵，全0
        all_task_fea = all_task_fea.reshape(self.batch_size, -1, self.args['gcn_input_dim'])  # bs * task * 12(特征值的个数)
        # Logger.log("Training/paralle_env/m_candidate_fea_input", f"m_feas={m_feas.shape}, all_task_fea={all_task_fea.shape}", print_true=1)  

        task_index = task_index.cpu().numpy()  # 先转成array数组 = (bs,)
        m_mask = m_mask.cpu().numpy()  # 先转成array数组 = bs * 1 * m
        
        for i in range(self.batch_size):  # bs个并行环境
            t_ins = self.ability_instance[i][0]  # task*m
            p_ins = self.ability_instance[i][1]  # task*m
            tt_ins = self.ability_instance[i][2]  # m*m
            edge_ins = self.ability_instance[i][3]  # 二维[[0,1],[2,3]] 2边均分m
            # print("--step生成m_fea1: tt_ins = ", tt_ins, tt_ins.shape)

            pt_ins = np.multiply(t_ins, np.abs(p_ins)) # task * m  P*T 有正负！

            """对那些不能选择的m，能力值用mean表示，防止0无意义的参数"""
            # 找到不为0的元素并计算均值
            over_zero_elements_t = t_ins[task_index[i]][t_ins[task_index[i]] > 0]  # 当前bs选择了哪个task，找这个task的能力t中大于0的元素
            mean_t = np.mean(over_zero_elements_t )  # 当前所选task的可以做的m的时间t均值

            over_zero_elements_pt  = pt_ins[task_index[i]][pt_ins[task_index[i]] > 0]
            mean_pt = np.mean(over_zero_elements_pt )  # 当前所选task的可以做的m的能耗p*t均值

            over_zero_elements_p  = p_ins[task_index[i]][p_ins[task_index[i]] > 0]
            mean_p = np.mean(over_zero_elements_p )  # 当前所选task的可以做的m的功率p均值

            """
            候选m的节点特征（bs，m，6）= 做当前task的<能力t, 能力pt, 所需transt, 可选Im, 能力p, 属于edge>(有则真，无则估计mean)
            """
            for m_index in range(self.nmachines): # TODO 不是累加值，是可依据判断值，所有的m的所有的fea都要更新（初始都为0）
                
                m_feas[i][m_index][0] = t_ins[task_index[i]][m_index] if t_ins[task_index[i]][m_index] > 0 else mean_t  # ！TODO 1 候选m节点的能力t，不能做用能做m的mean值代替

                #TODO 虽然最后会有mask屏蔽掉，但是这些都输入到网络，还是有点影响的吧？能选的m是真实值，不能选的就用能选的mean来代替！（用minus可能会NaN）
                m_feas[i][m_index][1] = pt_ins[task_index[i]][m_index] if pt_ins[task_index[i]][m_index] > 0 else mean_pt # ！TODO 2 候选m节点的能力pt，不能做用能做m的mean值代替

                if task_index[i] % self.nmachines == 0:  # action从0开始，表明是每个job的首位task，不会有运输
                    new_avail_transT = 0
                else:  # 既然选到了中间位置的task，其前置task必定被选，不然不满足先后工序！！！
                    # TODO 上一个task 和 当前task，是否会有运输时间！函数中会始终判断都是同一个job的！！！ 前置必然被选，此时的运输就是新增的运输时间
                    """
                    注意：
                    1、task_fea里边存的是m_id，从1开始的，要转成从0开始（task_fea是在第6个元素！！！）
                    2、task_fea= bs*task*12，task是按照顺序的，此时是在job的中间task，所以上一个task就是再同一个job里！
                    3、task_fea中没有被调度的节点，m_id=0！此时查表就不对了？（放心，我们查找的是上一个task的m，和当前task候选m之间的运输时间，必存在）
                    """
                    new_avail_transT = tt_ins[int(all_task_fea[i][task_index[i]-1][5])-1][m_index] # TODO 同job前一个task的m的index，和当前遍历的m_index,查表就知道选了当前m会有多少的运输时间新增
                m_feas[i][m_index][2] = new_avail_transT  # 新增的运输时间    TODO 3 选择当前候选m所需的transT

                m_feas[i][m_index][3] = 1 - int(m_mask[i][0][m_index])  # 注意：mask中true=1是不能干！所以取反，当前m节点是否能选，bool转成01   TODO 4 可选~mask

                m_feas[i][m_index][4] = p_ins[task_index[i]][m_index] if p_ins[task_index[i]][m_index] > 0 else mean_p    # ！ TODO 5 候选m节点对应task的p能力，不能干用mean  
                m_feas[i][m_index][5] = np.where(edge_ins == m_index)[0][0] + 1  # 找到m节点在edge_ins的索引，返回元组，包含两个数组，行和列数组，取行数组的第一个满足条件的元素的index， +1 作为id，从1开始的边  TODO 6 属于edge

        return m_feas  # bs * m * 6   np.array
    
    """并行环境中的env.step, 输入action更新下一时刻的state和产生reward"""
    def DGFJSPEnv_paral_step(self, joint_actions):
        """
        1、actor输出：action = tensor([1, 3, 3, 2, 1, 1, 1, 1, 1, 2, 0, 3, 3, 1, 2, 3]) torch.Size([16])，idx从0开始,(bacth,)
        2、转成task_id： self.chosen_action_list_batch = [[5], [13], [13], [9], [5], [5], [5], [5], [5], [9], [1], [13], [13], [5], [9],[13]]  16个list，分别记录选择的task，append加在后边,id从1开始
        3、循环batch_size次数，每个并行env都有新state产生
        """
        a_lst = joint_actions  # batch_size * step_chose_a
        adj_batch_ = []  # 初始化一个list，用来存放所有env的adj（注意：最终返回一个np.array就好，到时候在forward里边会转成tensor + 大的对角矩阵的！！！！）
        tasks_fea_batch_ = []  # 记录每一个env的节点特征向量
        machine_fea_batch_ = [] # 记录每一个env的m节点的特征向量
        self.oenv_info = []  # 记得初始化，每一次的info都不一样的！
        
        for l_batch in range(self.batch_size):
            select_task_index = a_lst[l_batch][0]  # 每一个batch中，当前选择的task_index是list的第一个
            select_mch_index = a_lst[l_batch][1]  # 每一个batch中，当前选择的m_id是list的第二个
            joint_action = [select_task_index,select_mch_index]
            """
            1、一个env对应一个joint_action，循环选取
            2、self.ability_instance[k_batch] = [t,e1,e2,edge]
            3、新增邻居矩阵 + 各个节点特征的state返回
            4、此为联合动作[task_index,m_idx]。action = task_index = task_id - 1
            
            env.step:
            return state, reward, done, info, r_t, r_idle, r_pt, r_transT, ft_s, it_s, adj_wrk, tasks_fea, machine_fea, tasks_fea_1101     #每一步都会返回（state + reward + done + info），
            """
            _, r, o_done, _, rmk, ridle, renergy_m, renergy_transM, _, _, \
                adj_, _, machine_fea_, tasks_fea_ = self.paral_env_DG[l_batch].step(joint_action)  # env.step

            # TODO：检验是否选到了负数的t
            if self.ability_instance[l_batch][0][select_task_index][select_mch_index] < 0:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"============= 'DGFJSPEnv_paral_step' occur error: chose Minus: t={self.ability_instance[l_batch][0][select_task_index][select_mch_index]}, p= {self.ability_instance[l_batch][1][select_task_index][select_mch_index]}")

            adj_batch_.append(copy.deepcopy(adj_))  # 转成np.arry
            tasks_fea_batch_.append(copy.deepcopy(tasks_fea_))  # 每一个env的节点特征矩阵存到list
            machine_fea_batch_.append(copy.deepcopy(machine_fea_))  # 记录m节点特征向量

            """TODO 对每一个step返回的即时的4个r指标，进行动态放缩 + 相同样本运行越多，放缩效果越好 (np.array形式)"""
            r_vector = np.array([rmk, ridle, renergy_m, renergy_transM])  # 每个样本中的每一step的4个即时的r
            r_vector_scaling = self.paral_Rscaling_instance[l_batch](r_vector)  # 相当于调用实例化中的__call__函数
            # Logger.log("Training/paralle_env/batch_env_step/dynamic_reard_scalling", f"4 reward scaling: self.paral_Rscaling_instance[l_batch].running_ms.n={self.paral_Rscaling_instance[l_batch].running_ms.n}", print_true=0)  # 检查 self.n在同样本中是否会持续累加 

            # r_vector_scaling = [ rmk, ridle, renergy_m, renergy_transM], 已放缩且np.array
            oenv_step_info = [r, o_done, r_vector_scaling[0], r_vector_scaling[1], r_vector_scaling[2], r_vector_scaling[3]]  # 放缩之后的即时r的4指标，重新按照原顺序存进oenv_step_info
            self.oenv_info.append(oenv_step_info)  # batch_size个oenv_step_info

        adj_batch_ = np.array(adj_batch_)  # 转成np.arry = env_batch * tasks * tasks (np.array)
        tasks_fea_batch_ = np.concatenate(tasks_fea_batch_, axis=0)  # 在第一维度进行连接, 转成np.arry = (env_batch*tasks) * 12 (np.array)
        machine_fea_batch_ = np.array(machine_fea_batch_)  # 转成np.arry = bs*m*8 (np.array)
        # Logger.log("Training/paralle_env/batch_env_step_output", f"adj_batch_={adj_batch_.shape}, tasks_fea_batch_={tasks_fea_batch_.shape}, machine_fea_batch_={machine_fea_batch_.shape}", print_true=1)
    
        # bs个env中的邻接矩阵 + 4reweard + machine下一时刻状态 + task下一时刻状态
        return adj_batch_, self.oenv_info, machine_fea_batch_, tasks_fea_batch_  
    
    def reset_data(self):

        """
        注意：
        1、你的并行的所有env都需要reset一下，不然没办法继续跑！（理论上每次episode都是重新初始化，不reset也没事，保险起见，写上？）
        放到main里边去reset了！
        2、并行环境 + step的infor可以清0，因为重新episode的时候都会重新产生！！！！
        """
        # self.ability_instance = []  # TODO 相同的samples之间不会清零，除非换了下一组bs，会重新赋值覆盖！记录并行的ability instance的所有信息  (注意：这里不能清0！！不然没有instance来初始化env！整个batch期间不变，先不清0)
        self.paral_env_DG = []  # 记录并行的选择DG的环境
        self.oenv_info = []  # 记录每一个batch的reward, done, r_t, r_idle, r_energy_m, r_energy_transM
        # self.paral_Rscaling_instance   # 不清0，参考ability_instance直接覆盖
        

    

    

