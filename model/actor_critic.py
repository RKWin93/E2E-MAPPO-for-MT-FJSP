import copy
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from trainer.train_device import device
from model.gcn_mlp import Encoder, MLPActor, aggr_obs, MLPCritic
from instance.generate_allsize_mofjsp_dataset import Logger
from algorithm.agent_func import select_operation_action, greedy_select_action
from model.gat import GATLayer
from trainer.fig_kpi import get_GPU_usage





"""
传入之前的参数：
1、需要转成tensor，然后在GPU上
2、最大的场景，输出的状态参数也才：batch*job，图网络的好处；可能邻居和node_fea比较大，但是这只是输入网络的参数而已！
------------------------------一步双actor的方式---------------------------------------"""
"""
按照全局和局部的critic的方式，进行重新写
"""
class Operation_Actor_JointAction_selfCritic(nn.Module):
    def __init__(self, configs):
        super(Operation_Actor_JointAction_selfCritic, self).__init__()

        self.n_job = configs['n_job']  # 后续想调用的，可以用self.的形式，其他没必要
        self.n_machine = configs['n_machine']
        self.n_total_task = self.n_job * self.n_machine
        self.batch_size = configs['env_batch']

        self.GAMMA = configs['GAMMA']  # reward折扣率
        self.LAMDA = configs['LAMDA']  # GAE参数
        self.epsilon = configs['epsilon']  # 重要性采样的裁剪
        self.ENTROPY_BETA = configs['ENTROPY_BETA']  # 偏向于出现0001，那就调小！！！！  平衡熵正则化

        # device = torch.device(configs['device'] if torch.cuda.is_available() else "cpu")
        
        gcn_layer = configs['gcn_layer']
        num_mlp_layers = configs['mlp_fea_extract_layer']
        gcn_input_dim = configs['gcn_input_dim']
        gcn_hidden_dim = configs['gcn_hidden_dim']
        learn_eps = configs['learn_eps']
        neighbor_pooling_type = configs['neighbor_pooling_type']
        
        mlp_actor_layer = configs['mlp_actor_layer']  # 作为最终policy的mlp的层数

        """
        图卷积网络：
        1、初始化GIN网络：暂不用学习参数e，简单的图卷积：自身+邻居节点（入度）：因为只有前一时刻才有影响，后续没调度你卷积干什么？？（TII的就很无语，他俩都是乱试）
        2、正常node vector也需要mlp的网络，但是这里没有用。直接用的特征向量（TII用的GAT时，用的mlp先转换！）
        3、我不用old网络：自身网络采集作为old网络，比较时不重新采集！（同eswa！）
        4、直接用configs，全局统一，防止直接传参数，传的不对到时候理解复杂！
        
        OUTPUT：forward =  全图池化嵌入 + all节点的嵌入  # batch* hidden 和（batch*tasks）* hidden
        """
        self.encoder = Encoder(num_layers=gcn_layer,
                               num_mlp_layers=num_mlp_layers,
                               input_dim=gcn_input_dim,
                               hidden_dim=gcn_hidden_dim,
                               learn_eps=learn_eps,
                               neighbor_pooling_type=neighbor_pooling_type,
                               device=device).to(device)
        Logger.log("Training/ppo/init_job_encoder", f"GIN + MLP: self.encoder={self.encoder}", print_true=1)
        

        self._input = nn.Parameter(torch.Tensor(configs['gcn_hidden_dim']))  # 创建一个名为`_input`的`nn.Parameter`对象，它是一个可学习的张量参数。torch.Tensor(hidden_dim)创建一个形状为`(hidden_dim,)`的张量
        self._input.data.uniform_(-1, 1).to(device)  # 使用`uniform_(-1, 1)`方法对`_input.data`进行均匀分布的随机初始化。`uniform_(-1, 1)`会将`_input.data`中的元素从均匀分布中进行随机采样。

        """
        # class MLPActor(nn.Module):
        #     def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        1、Actor_policy输入参数：待选节点嵌入 + 全图节点嵌入 + M节点全图嵌入 = in 上述3个元素，so hidden*3
        """
        self.o_policy = MLPActor(mlp_actor_layer, gcn_hidden_dim * 3, gcn_hidden_dim, 1).to(device)  # 几层 + in + hidden + out
        Logger.log("Training/ppo/init_job_actor", f"MLP: self.o_policy={self.o_policy}", print_true=1)

        """
        actor_critic的MLP网络
        """
        mlp_critic_layer = configs['mlp_critic_layer']
        critic_input_dim = configs['critic_input_dim']
        critic_hidden_dim = configs['critic_hidden_dim']
        self.job_critic = MLPCritic(mlp_critic_layer, critic_input_dim, critic_hidden_dim, 2).to(device)  # 3层 + 128in + 128隐藏 + 2out  TODO 全图特征的输入= 1*128 + mk和it的反馈reward = 2！
        # TODO 0109-消融实验-不使用r向量和v向量，那么动态的权重01其实没用，不改问题不大，看看效果
        # self.job_critic = MLPCritic(mlp_critic_layer, critic_input_dim, critic_hidden_dim, 1).to(device)  # 3层 + 128in + 128隐藏 + 1out  
        Logger.log("Training/ppo/init_job_critic", f"MLP: self.job_critic={self.job_critic}", print_true=1)

        # 正交初始化！
        if configs['use_orthogonal']:  # 就是configs的init
            for name, p in self.named_parameters():  # 遍历模型的所有参数，通过`self.named_parameters()`方法获取参数的名称和值。
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p,
                                            gain=1)  # 对于参数名称中包含’weight’的参数，判断其维度是否大于等于2。如果是，则使用`nn.init.orthogonal_`方法对该参数进行正交初始化，其中`gain`参数指定了初始化的增益。
                        Logger.log("Training/ppo/init_o_orthogonal_gcn_weight", f"Job Actor: orthogonal_gain=1", print_true=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)  # 对于参数名称中包含’bias’的参数，使用`nn.init.constant_`方法将其初始化为常数0。
                    Logger.log("Training/ppo/init_o_orthogonal_gcn_bias", f"job Actor: constant_(p, 0)", print_true=1)

    def forward(self,
                x_fea,  # task节点的特征向量：                                               batch*task，12
                graph_pool_avg,  # 全图节点嵌入求均值的矩阵：                                 batch，batch*task
                padded_nei,  # max的pooling才有用
                adj,  # 邻居矩阵（带权重+j到i+自身置1）                                       batch*task，batch*task
                candidate,  # 可选task的id，                                                 batch，job
                h_g_m_pooled,  # 传入每一step的m节点的全图节点嵌入均值：                       batch，hidden
                mask_operation,  # 传入当前可选节点的mask位，某一job选完了，那就置为true：      batch，job
                use_greedy=False
                ):
        # if h_g_m_pooled is not None:
        #     Logger.log("Training/ppo/job_actor_forward_input", f"x_fea={x_fea.shape}, graph_pool_avg={graph_pool_avg.shape}, adj={adj.shape}, candidate={candidate.shape}, h_g_m_pooled={h_g_m_pooled.shape}, mask_operation={mask_operation.shape}", print_true=1)
        # else: 
        #     Logger.log("Training/ppo/job_actor_forward_input", f"x_fea={x_fea.shape}, graph_pool_avg={graph_pool_avg.shape}, adj={adj.shape}, candidate={candidate.shape}, mask_operation={mask_operation.shape}", print_true=1)

        """
        传入的是np.array，需要转成tensor + GPU
        :param x_fea:  batch*task，12
        :param graph_pool_avg: batch，batch*task  （已经是稀疏张量）
        :param padded_nei:
        :param adj:  batch，task，task
        :param candidate:  batch，job
        :param h_g_m_pooled:  batch，hidden （上一个net的输出：已经是张量）  不对应啊，123的选m，和按概率选o，这没法一起训练吧？？？
        :param mask_operation: batch，job (已经是张量)
        :param use_greedy: 默认是False
        :return:
        """

        # print("1.1:{}".format(get_GPU_usage()[1])) # 返回值的第二个参数就是显存使用量
        
        if torch.is_tensor(x_fea):  # 判断是否为张量，因为在update里边，buffer里边读取的都已经是tensor了
            # 原deepcopy：这部分代码首先使用`np.copy`创建了`fea`的副本，然后使用`torch.from_numpy`将其转换为PyTorch张量
            x_fea = copy.copy(x_fea).float().to(device)  
            candidate = copy.copy(candidate).long().to(device)  # 原deepcopy：转为long整型的张量 batch * job
            # 原deepcopy：这行代码首先使用`deepcopy`函数创建了`adj`的副本，然后将其转移到指定的`device`上，并将其稀疏化处理。最后，使用`aggr_obs`函数对稀疏的`adj`进行聚合操作，生成了`env_adj`=大对角adj稀疏矩阵 = （bs*task，bs*task）
            adj = aggr_obs(copy.copy(adj).to(device).to_sparse(),
                           self.n_total_task)  
        else:
            # 这部分代码首先使用`np.copy`创建了`fea`的副本，然后使用`torch.from_numpy`将其转换为PyTorch张量
            x_fea = torch.from_numpy(np.copy(x_fea)).float().to(device)  
            candidate = torch.from_numpy(np.copy(candidate)).long().to(device)  # 转为long整型的张量 batch * job
            # mask_operation = torch.from_numpy(np.copy(mask_operation)).to(device)  # 转为张量，可选节点的mask：batch* job
            """
            # all batch的所有的邻居矩阵，adj = torch.Size([4, 16, 16])  batch*tasks*tasks 张量
            # 然后aggr_obs聚合成大的稀疏批邻居矩阵
            # 估计是转成大型对角矩阵的形式，稀疏矩阵表示，即64*64 = batch*tasks, batch*tasks

            我传入的adj是np.array，所以我要先转成tensor，再计算!!!
            """
            # print("--观察数据：adj = ", adj, adj.shape)  # adj已经是张量了？？？我传入的还是np.array
            # 原deepcopy：这行代码首先使用`deepcopy`函数创建了`adj`的副本，然后将其转移到指定的`device`上，并将其稀疏化处理。最后，使用`aggr_obs`函数对稀疏的`adj`进行聚合操作，生成了`env_adj`=大对角adj稀疏矩阵 = （bs*task，bs*task）
            adj = aggr_obs(torch.from_numpy(copy.copy(adj)).to(device).to_sparse(),
                           self.n_total_task)  

        mask_operation = copy.copy(mask_operation).to(device)  # 原deepcopy：已经是张量，可选节点的mask：batch* job

        # print("1.2:{}".format(get_GPU_usage()[1]))

        """
        3层gin网络卷积之后，
        RETURn：全图的节点嵌入(维度4（batch） * 128) + 各个节点的嵌入（64（batch*tasks） * 128）
        """
        h_g_o_pooled, h_o_nodes = self.encoder(x=x_fea,  # 这里直接就GIN的节点特征，和全局的池化特征都有了
                                               graph_pool=graph_pool_avg,
                                               padded_nei=padded_nei,
                                               adj=adj)
        # Logger.log("Training/ppo/job_actor_forward_encoder", f"h_g_o_pooled={h_g_o_pooled.shape}, h_o_nodes={h_o_nodes.shape}", print_true=1)
        
        """
        现在的动作直接是对可选的node进行打分：
        1、传入所有env_batch中可选的job的具体task
        2、采用gather方式：从h_o_nodes所有节点嵌入中选择具体可选节点
        3、可选节点h_o_nodes：batch*task，128
        4、候选节点candidate：batch，job
        5、期望gather之后：batch，job，128

        这个是从外边传递进来的候选task的编号              
        candidate = tensor
        ([[ 0,  4,  8, 12],  
        [ 0,  4,  8, 12],
        [ 0,  4,  8, 12],
        [ 0,  4,  8, 12]], device='cuda:0')
        """
        
        """
        候选节点candidate：
        1、shape = （batch * job） 其中每一行代表剩下可选的task的id，0开始赋值（即我的可选task的id的列表）
        2、使用`unsqueeze(-1)`方法将`candidate`张量在最后一个维度上添加一个维度，将其形状变为`(batch_size, self.n_j, 1)`。
        3、使用`expand(-1, self.n_j, h_nodes.size(-1))`方法将`candidate`张量在第二个维度上进行扩展，使其形状与`h_nodes`的形状相同。这将生成一个形状为`(batch_size, self.n_j, h_nodes.size(-1))`的张量`dummy`。
            `expand`方法可以用来在指定的维度上复制张量的元素，从而改变张量的形状。
            `expand`方法并不实际复制张量的元素，而是通过改变张量的尺寸和视图来模拟扩展操作。这意味着扩展后的张量与原始张量共享相同的存储空间，因此修改其中一个张量的元素也会影响到另一个张量。（怪不得叫dummy，共享内存的，改一个其他都改了）
        """
        #  batch，job，128
        dummy = candidate.unsqueeze(-1).expand(-1, self.n_job, h_o_nodes.size(-1))  # batch，job，128。最后一个维度扩维 = 4,4,1，然后复制元素变成=4,4,128
        # Logger.log("Training/ppo/job_actor_forward_dummy", f"new: dummy={dummy.shape}, old：candidate={candidate.shape}", print_true=1)

        """
        h_o_nodes整形：batch，task，128
        1、out维度 = 索引张量dummy的维度： batch, job, 128 （所以从第二个维度挑选）
        2、out[i][j][k] = input[i][index[i][j][k]][k], dim=1          !!!!!!!!!!!!!! 快速理解
        """ 
        candidate_feature = torch.gather(h_o_nodes.reshape(dummy.size(0), -1, dummy.size(-1)),
                                         1,
                                         dummy)  # 类似于从每个样本中，挑选对应candidate中可选task的编号的节点的嵌入给挑选出来，重新组成变量！！！
        # Logger.log("Training/ppo/job_actor_forward_candidate_feature", f"candidate_feature={candidate_feature.shape}", print_true=1)

        """
        全图所有节点嵌入的均值，也作为输入网络的参数
        1、原：batch，hidden
        2、h_pooled：4*128 batch*hidden_dim ，第二个维度扩展1维：4,1,128
        3、然后扩展数据变成candidate_feature ：4,4,128的维度，  batch, job, 128
        4、那么就需要对第二个维度的数据进行复制，复制4份: 每个样本的全图数据 * 4 复制了4分一样的
        """
        h_g_o_pooled_repeated = h_g_o_pooled.unsqueeze(-2).expand_as(candidate_feature)  # batch, job, 128
        # Logger.log("Training/ppo/job_actor_forward_h_g_o_pooled_repeated", f"h_g_o_pooled_repeated={h_g_o_pooled_repeated.shape}", print_true=1)

        """
        machine网络的输出参数：m节点的全图节点嵌入均值：step，batch，hidden
        1、每次调用forward只传入一次step的h_g_m_pooled = batch，hidden
        2、candidate_feature = batch, job, 128 
        3、然后在倒数第二维度增加一个维度：batch * 1 * hidden_dim
        4、expand_as复制扩充变成candidate_feature = batch * job * 128 （每个batch中可以选择的job的对应的节点嵌入，mlp输出是128维度）

        先注释，选m和选o一点也不对应，这怎么能拿过来训练呢！
        """
        if h_g_m_pooled == None:  # 如果此次是初始选择动作，还没有mch_pool的结果：使用之前建立的128均匀分布的元素
            h_g_m_pooled_repeated = self._input[None, None, :].expand_as(candidate_feature).to(device)  # 并通过 `[None,None, :]` 在前两个维度上添加了维度，然后扩展复制维度到4,4,128
        else:  # 如果有选择m，就会不断更新m节点嵌入的全局池化值
            """# mch_pool全图的特征： batch * hidden_dim （mlp特征提取之后）"""
            h_g_m_pooled_repeated = h_g_m_pooled.unsqueeze(-2).expand_as(candidate_feature).to(device)  # 插入一个倒数第二维度，然后expand扩展复制成4,4,128=batch * job * 128
        # Logger.log("Training/ppo/job_actor_forward_h_g_m_pooled_repeated", f"Last_in: h_g_m_pooled_repeated={h_g_m_pooled_repeated.shape}", print_true=1)
        
        """
        合并特征值：输入到网络的参数：
        1、输入到actor网络的，除了全图所有节点的均值以外，还有对应当前可选的节点嵌入（我之前输入的都只是全图的状态，对应节点的状态也可选）
        2、统一shape = batch，job，hidden。（理解为，只用candidate的节点输入到网络，然后输出的就是每个候选节点的打分，转成概率就是可选择的动作）
        
        # 输入到Actor网络的是：待选节点的节点嵌入（gather挑选出来的） + 全图样本数据复制4分 + 设备数据：4,4,128
        # 因为是选择job，所以当前可以选择的节点就job=4个，所以shape = batch * job * hidden_dim*3（3个类型的特征）
        """      
        concateFea = torch.cat((candidate_feature,
                                h_g_o_pooled_repeated,
                                h_g_m_pooled_repeated),
                                dim=-1)  # 在最后一个维度进行合并：4,4,128*3 （batch * job * hidden*3）
        
        """
        o_policy就是MLPActor的网络
        输出logit值：理解为可选节点的打分score

        输入：4,4,128*3 （batch * job * hidden_dim）
        输出：4,4,1    1是因为输入的数据已经是可以每一个可以选择的job的节点表征了，输出1是针对每一个可选job的打分！（batch，job，1）
        """
        candidate_scores = self.o_policy(concateFea)  # Job的全连接层：3层，128*3输入，128隐藏，1输出（只看第二维度，最后一个维度为1，便于去掉）
        # candidate_scores = candidate_scores * 10  # 之前都是/1000作为归一化，可能是为了增加得分的范围或强调得分的重要性
        # Logger.log("Training/ppo/job_actor_forward_candidate_scores", f"candidate_scores={candidate_scores.shape}", print_true=1)

        """
        mask的问题：传入batch，job = 4，4
        1、都是加到logits层，没有本质区别，他直接置为-inf
        2、理论上：我设定的每一个m都可以做，那么mch_mask感觉就没有用了！！！（对的，他的m的mask一直全是False）
        3、为什么他的mask先第一列的job选完之后，后边的每次只能选择指定好的job（指定那个由env里边在判断！）
        """
        mask_operation_reshape = mask_operation.reshape(candidate_scores.size())  # 变成和打分结果一样：4,4,1（batch，job，1）
        # 将mask_reshape中对应位置为True的元素置为-inf
        candidate_scores[mask_operation_reshape] = float('-inf')
        # Logger.log("Training/ppo/job_actor_forward_mask_operation_reshape", f"mask_operation_reshape={mask_operation_reshape.shape}, new candidate_scores={candidate_scores}", print_true=1)

        """
        在这里对logit采用softmax激活函数：
        1、输出可选节点的动作的概率！
        2、candidate_scores = （batch，job，1），先.squeeze(-1)去除最后一个维度
        3、prob概率pi = batch，job
        4、候选节点candidate：batch，job
        """
        prob = F.softmax(candidate_scores.squeeze(-1), dim=-1)  # 维度是4,4 = batch*job  所以是在最后一个维度进行softmax操作
        # Logger.log("Training/ppo/job_actor_forward_operation_prob", f"prob={prob.shape}", print_true=1)
        
        if use_greedy:
            task_index, action_index, log_a = greedy_select_action(prob, candidate)  # 贪婪，那就是每次就选择最大prob = all (batch_size,)
        else:
            """
            # 张量taskindex = tasks的具体id（从0开始）：一维，4元素，（batch）
            # 张量action_index：一维，（batch），离散采样得到的动作index
            # 张量log_a：一维，（batch），采样动作的对应的离散概率
            """
            task_index, action_index, log_a = select_operation_action(prob, candidate)  # 否则就是dist离散分布的采样
        # Logger.log("Training/ppo/job_actor_forward_a_index", f"task_index={task_index.shape}, action_index={action_index.shape}, task_index={log_a.shape}", print_true=1)
        
        # TODO 1113-新增局部的job的critic，只关注mk和it = in hidden_size, out 2 reward's value
        job_v = self.job_critic(h_g_o_pooled) # h_g_o_pooled = bs * hidden  
        # Logger.log("Training/ppo/job_actor_forward_job_value", f"job_v={job_v.shape}", print_true=1)  # batch*2

        return task_index, action_index, log_a, prob, h_g_o_pooled, job_v  # 全图的特征batch*hidden


class Machine_Actor_JointAction_selfGAT_selfCritic(nn.Module):
    def __init__(self, configs):
        super(Machine_Actor_JointAction_selfGAT_selfCritic, self).__init__()

        self.n_machine = configs['n_machine']
        mlp_actor_layer = configs['mlp_actor_layer']
        self.machine_hidden_dim = configs['machine_hidden_dim']

        self.bn = torch.nn.BatchNorm1d(self.machine_hidden_dim).to(device)  # 批归一化层

        """候选m节点：6元素 + 已调度m节点：8元素"""
        self.m_fea_1_fcl = nn.Linear(6, self.machine_hidden_dim, bias=False).to(device)  # 全连接层来池化节点的特征，6in，128out，没有偏置项
        self.m_fea_2_fcl = nn.Linear(8, self.machine_hidden_dim, bias=False).to(device)  # 全连接层来池化节点的特征，6in，128out，没有偏置项
        self.activation = nn.ELU() #输入 x 中的负值将按照 ELU 的定义进行变换，而正值将保持不变 TODO 在m的不同维度转成hidden的FCL后边，新增一个激活函数，一层后跟一个！
        # self.activation = nn.LeakyReLU() #

        self.heads = 1   # 注意力机制的多头

        # 初始化GAT网络
        self.gat_layer = GATLayer(in_features=self.machine_hidden_dim,  # 输入特征向量的维度；如果所有节点的维度不同，先进行一个FCL进行下统一！
                            out_features=self.machine_hidden_dim,  # 输出的维度
                            n_heads= self.heads,  # 几个多头，可以为1
                            concat= False,  # 是否对多头就行concate TODO 多头机制需要拆分的，这里不用多头，head=1
                            dropout= 0,  # dropout的系数，防止过拟合！
                            leaky_relu_slope= 0.2  # `leaky_relu_slope`参数确定了负输入的斜率大小。解决训练过程中神经元对负输入变得不响应的问题，这被称为“dying ReLU”问题
                            ).to(device)
        Logger.log("Training/ppo/init_machine_GAT", f"GAT+Linear: self.gat_layer={self.gat_layer}", print_true=1)

        """
        machine这里：没有图结构 + 节点之间不连接，所以ESA那片直接用全连接层来池化/聚合节点的特征向量(直接一个线性层就ok了？GIN的MLP可是循环3次的！！！！！！！ + 为什么没有bias？？？)
        
        我这里直接用GAT方式！
        """
        self.fcl_pooling = nn.Linear(self.machine_hidden_dim, self.machine_hidden_dim, bias=False).to(device)  # GAT聚合之后，全连接层来池化节点的特征，128in，128out，没有偏置项
        self.m_policy = MLPActor(mlp_actor_layer, self.machine_hidden_dim * 3, self.machine_hidden_dim, 1).to(device)  # 选择m的actor网络：共3层（包括输入）+ 128*3in + 128hidden + 1out
        Logger.log("Training/ppo/init_machine_actor", f"MLP: self.fcl_pooling={self.fcl_pooling}, self.m_policy={self.m_policy}", print_true=1)

        """
        machine_critic的MLP网络
        """
        mlp_critic_layer = configs['mlp_critic_layer']
        critic_input_dim = configs['critic_input_dim']
        critic_hidden_dim = configs['critic_hidden_dim']
        self.machine_critic = MLPCritic(mlp_critic_layer, critic_input_dim, critic_hidden_dim, 2).to(device)  # 3层 + 128in + 128隐藏 + 2out  TODO 全图特征的输入= 1*128 + pt和tt的反馈reward = 2！
        # TODO 0109-消融实验-不使用r向量和v向量，那么动态的权重01其实没用，不改问题不大，看看效果
        # self.machine_critic = MLPCritic(mlp_critic_layer, critic_input_dim, critic_hidden_dim, 1).to(device)  # 3层 + 128in + 128隐藏 + 1out 
        Logger.log("Training/ppo/init_machine_critic", f"MLP: self.machine_critic={self.machine_critic}", print_true=1)

        # 正交初始化！
        if configs['use_orthogonal']:  # 就是configs的init
            for name, p in self.named_parameters():  # 遍历模型的所有参数，通过`self.named_parameters()`方法获取参数的名称和值。
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p,
                                            gain=1)  # 对于参数名称中包含’weight’的参数，判断其维度是否大于等于2。如果是，则使用`nn.init.orthogonal_`方法对该参数进行正交初始化，其中`gain`参数指定了初始化的增益。
                        Logger.log("Training/ppo/init_m_orthogonal_gcn_weight", f"Machine Actor: orthogonal_gain=1", print_true=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)  # 对于参数名称中包含’bias’的参数，使用`nn.init.constant_`方法将其初始化为常数0。
                    Logger.log("Training/ppo/init_m_orthogonal_gcn_bias", f"Machine Actor: constant_(p, 0)", print_true=1)

    def forward(self, machine_fea_1, machine_fea_2, h_pooled_o, machine_mask):  # m_fea2 指向 m_fea1： 有向边
        
        
        # Logger.log("Training/ppo/machine_actor_forward_input", f"machine_fea_1={machine_fea_1.shape}, machine_fea_2={machine_fea_2.shape}, h_pooled_o={h_pooled_o.shape}, machine_mask={machine_mask.shape}", print_true=1)
        
        """

        :param machine_fea_1:  候选m: (当前task的对应的m的特征)传入machine网络的m节点的特征向量 = env_batch * m * 6 （np.array）
        :param machine_fea_2:  已调度m：（选好m输入到ENV中的被选更新的state）传入machine网络的m节点的特征向量 = env_batch * m * 8 （np.array）
        :param h_pooled_o: 传入的上一个step的task节点的全图特征 = env_batch * hidden
        :param machine_mask: 当前job选择的task的mask = env_batch * 1 * m  好像直接就是张量？？？
        :return:
        """

        if torch.is_tensor(machine_fea_1):  # 判断是否为张量，因为在update里边，buffer里边读取的都已经是tensor了
            machine_fea_1 = copy.copy(machine_fea_1).float().to(device)  # 原deepcopy：转为张量+gpu  (env_batch * m * 6)
            machine_fea_2 = copy.copy(machine_fea_2).float().to(device)  # 原deepcopy：转为张量+gpu  (env_batch * m * 6)
        else:
            machine_fea_1 = torch.from_numpy(np.copy(machine_fea_1)).float().to(device)  # 转为张量+gpu  (env_batch * m * 6)
            machine_fea_2 = torch.from_numpy(np.copy(machine_fea_2)).float().to(device)  # 转为张量+gpu  (env_batch * m * 6)
        # print("Machine_Actor: forward： machine_fea = ", machine_fea, machine_fea.shape)
        # print("MActor: machine_fea_1 = ", machine_fea_1, machine_fea_1.shape)

        """将不同维度的m_fea转换成一致，然后按照（bs*m，2, hidden）进行转换： 代表两个m节点的聚合，2-->1"""
        m_fea1 = self.m_fea_1_fcl(machine_fea_1)  # bs * m * 128(hidden)
        m_fea2 = self.m_fea_2_fcl(machine_fea_2)
        # Logger.log("Training/ppo/machine_actor_forward_fcl_out", f"m_fea1={m_fea1.shape}, m_fea2={m_fea2.shape}", print_true=1)

        # 将 A 和 B 展平为形状为 (bs * m) 的一维张量
        m_fea1 = m_fea1.view(m_fea1.shape[0] * m_fea1.shape[1], m_fea1.shape[2])  # 三维变2维度：bs*m，hidden
        m_fea2 = m_fea2.view(m_fea2.shape[0] * m_fea2.shape[1], m_fea2.shape[2])
        
        # 交替堆叠 m_fea1 和 m_fea2的第二个维度的元素值，会变成 = bs*m，2，hidden，不放心再reshape一下！
        All_sameM_fea12 = torch.stack([m_fea1, m_fea2], dim=1).reshape(-1, 2, m_fea1.shape[-1])  # TODO 只有同个m的两个不同特征值进行聚合！(类比task_fea = bs*task，12的所有task和所有bs的特征值)
        # Logger.log("Training/ppo/machine_actor_forward_gat_in", f"All_sameM_fea12={All_sameM_fea12.shape}", print_true=1)

        """
        h = bs*m，2，hidden
        adj_m=(bs*m, head, 2, 2) 且ij=1表示j->i! + 默认有向边2->1，所以某一行全0，固定的
        
        1、创建一个形状为 (bs * m) * head * 2 * 2 的张量
        2、TODO 修改adj = [[1, 1], [0, 1]]让m_fea2自己聚合自己，虽然我不用这一行！（全0行变成全-inf行，然后softmax该行=Nan！）
        """
        adj_m = torch.zeros(All_sameM_fea12.shape[0], self.heads, 2, 2).to(device)
        # 将 2*2 的矩阵的第一行元素设置为 1
        adj_m[:, :, 0, :] = 1  # 第一行都是1
        adj_m[:, :, 1, 1] = 1  # 第二行第二列=1
        # Logger.log("Training/ppo/machine_actor_forward_adj_m", f"adj_m={adj_m.shape}", print_true=1)

        """h_prime_gat = （bs*m, 2, hidden）--- 其中2*hidden的第一行是我们想要的，第二行是m_fea2的自身聚合，我们只用第一行的参数 [:,0,:]= （bs*m, hidden）--reshape = （bs, m ,hiddden）"""
        h_prime_gat = self.activation(self.gat_layer(h=All_sameM_fea12, adj_mat=adj_m))  # TODO 严格按照公式，和Wh加权聚合之后，需要一个ELU激活！！输出m节点嵌入特征值
        # Logger.log("Training/ppo/machine_actor_forward_first_h_prime_gat", f"First_gat: h_prime_gat={h_prime_gat.shape}", print_true=1)

        # TODO GAT可以循环n次的！对应GCN，那就3次！
        h_prime_gat = self.activation(self.gat_layer(h=h_prime_gat, adj_mat=adj_m))
        h_prime_gat = self.gat_layer(h=h_prime_gat, adj_mat=adj_m)   # （bs, m ,hiddden）
        # Logger.log("Training/ppo/machine_actor_forward_last_h_prime_gat", f"Last_gat_no_activate: h_prime_gat={h_prime_gat.shape}", print_true=1)

        # TODO 为什么要单独只用一个？两个都用，或者全图的特征不是可以吗？（FCL或MLPActor的需求：bs*m*hidden）
        # h_machine_feas = h_prime_gat[:,0,:].reshape(-1, self.n_machine, self.machine_hidden_dim) # （bs, m ,hiddden）
        # TODO 参考GCN中的Average方式，求均值mean（dim=-2）！也可以reshape（bs*m，2*hidden）--out的时候只有一个hidden也行！
        h_machine_feas = h_prime_gat.mean(dim=-2).reshape(-1, self.n_machine, self.machine_hidden_dim) # （bs, m ,hiddden）  聚合方式！
        # Logger.log("Training/ppo/machine_actor_forward_h_machine_feas", f"h_machine_feas={h_machine_feas.shape}", print_true=1)
        
        # 直接数据放缩！考虑要不要加？因为p*t会很大！
        # machine_fea = machine_fea / configs['']et_normalize_coef  # 归一化处理: 直接原数据/1000 batch * m  当前所选择m的上一个工序的完工时间

        """
        1、fea直接输入到1线性层中，结果batch*m*128，转成2维：（batch*m）*128，输入到bn层中，结果再转回来：batch * m * 128
        2、类似于对m的节点进行了encoder，变成了池化之后节点嵌入 = (batch_size, n_m, hidden_size)
        """
        # TODO 此时就用一用bn层吧，fcl没必要了！
        # h_m_nodes = self.bn(self.fcl_pooling(h_machine_feas).reshape(-1, self.machine_hidden_dim)).reshape(-1,
        #                                                                                                 self.n_machine,
        #                                                                                                 self.machine_hidden_dim)
        h_m_nodes = self.bn(h_machine_feas.reshape(-1, self.machine_hidden_dim)).reshape(-1,
                                                                                       self.n_machine,
                                                                                       self.machine_hidden_dim)
        # Logger.log("Training/ppo/machine_actor_forward_h_m_nodes", f"h_m_nodes={h_m_nodes.shape}", print_true=1)

        """
        计算machine节点网络的全图特征：
        1、计算 `h_m_nodes` 张量在第2个维度上的均值，即对每个样本的 `n_m` 个节点特征进行平均。得到的 `pool` 张量形状为 `(batch_size, hidden_size)
        2、相当于我首先节点池化了一波，每个m都有128个特征，然后我对每个batch中的all m求他的平均的特征表征（类似于全图中各节点的嵌入的均值，池化过程！）
        """
        h_pooled = h_m_nodes.mean(dim=1)  # (batch_size, hidden_size)
        
        """"
        下述的变换操作：
        1、是为了和各个m节点的嵌入进行合并：batch*m*128
        2、从而进行expand的复制操作
        
        1、 `unsqueeze(1)` 在第二个维度上添加一个维度，并使用 `expand_as` 函数将 `pool` 张量复制扩展成和 `action_node` 张量相同的形状。
        2、得到的 `h_pooled_repeated` 张量形状为 `(batch_size, n_m, hidden_size)`。
        3、相当于：每个batch中的全m的节点嵌入复制了m=4份
        """
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(h_m_nodes)  # (batch_size, n_m, hidden_size)  m节点的全图节点嵌入的均值
        # Logger.log("Training/ppo/machine_actor_forward_h_pooled_repeated", f"h_pooled_repeated={h_pooled_repeated.shape}", print_true=1)
        
        """
        整形task节点的全图特征h_pooled_o：
        1、h_poooled_o: batch * hidden_dim , GIN之后的上一个状态的全图的节点嵌入的均值
        2、同上：第2维度加1，然后expand复制变成(batch_size, n_m, hidden_size) 4*4*128
        """
        o_pooled_repeated = h_pooled_o.unsqueeze(1).expand_as(h_m_nodes) # (batch_size, n_m, hidden_size)  o节点的全图节点嵌入的均值
        # Logger.log("Training/ppo/machine_actor_forward_o_pooled_repeated", f"Last_in: o_pooled_repeated={o_pooled_repeated.shape}", print_true=1)

        """
        合并特征值：
        1、维度不够就要进行expand的复制补充！
        2、输入变量： m节点的池化嵌入 + 所有m节点全局嵌入（mean） + 选择job的全图的all节点的嵌入（mean！） ： batch * m * 128*3
        """
        concateFea = torch.cat((h_m_nodes, 
                                h_pooled_repeated, 
                                o_pooled_repeated), dim=-1)  # 最后一个维度合并
        # Logger.log("Training/ppo/machine_actor_forward_concateFea", f"concateFea={concateFea.shape}", print_true=1)

        mch_scores = self.m_policy(concateFea)  # batch * m * 1 输出的是选择哪一个m的得分：放入到了MLPActor的网络中
        mch_scores = mch_scores.squeeze(-1) * 10  # 去掉最后一个维度：batch * m。数值放大10倍
        # Logger.log("Training/ppo/machine_actor_forward_mch_scores", f"mch_scores={mch_scores.shape}", print_true=1)
        

        """
        选择m的mask：(存在无法加工)
        1、# machine_mask: batch * 1 * m 所选择的task的对应的mask（false可选）
        2、squeeze（1）去掉维度为1的：batch * m，bool（）转换成布尔张量（若已经是bool，保持不变）：这将把非零值转换为 `True`，零值转换为 `False`。
        3、使用 `masked_fill` 函数，根据掩码张量的值将 `mch_scores` 张量中对应位置的元素填充为负无穷大。掩码张量中为 `True` 的位置将被填充，而为 `False` 的位置将保持不变。
        """
        mch_scores = mch_scores.masked_fill(machine_mask.squeeze(1).bool(), float("-inf"))
        # Logger.log("Training/ppo/machine_actor_forward_machine_mask", f"machine_mask={machine_mask.shape}, new mch_scores={mch_scores}", print_true=1)

        # 掩码加持之后的mch_scores，进行softmax输出选择m的概率：batch * m 数值变成概率了
        mch_prob = F.softmax(mch_scores, dim=-1)
        # Logger.log("Training/ppo/machine_actor_forward_mch_prob", f"prob={mch_prob.shape}", print_true=1)
        
        # TODO 1113-新增局部的machine的critic，只关注pt和tt
        machine_v = self.machine_critic(h_pooled)  # h_pooled = bs * hidden
        # Logger.log("Training/ppo/machine_actor_forward_machine_v", f"machine_v={machine_v.shape}", print_true=1)  # batch*2

        return mch_prob, h_pooled, machine_v  # batch*m的选择m的概率 + 返回所有m节点的全局嵌入（求均值）：batch * hidden_dim + m本地value=batch*2


"""
Critic的网络大部分是在Update的时候使用：
1、都是从ReplayBuffer里边获取的！
2、concat = [task节点的candidate + task全图特征 + m节点的特征值 + m节点的全图特征]！！！
"""
class Global_Critic_JointAction_GAT(nn.Module):
    def __init__(self, configs):
        super(Global_Critic_JointAction_GAT, self).__init__()

        gcn_layer = configs['gcn_layer']
        num_mlp_layers = configs['mlp_fea_extract_layer']
        gcn_input_dim = configs['gcn_input_dim']
        gcn_hidden_dim = configs['gcn_hidden_dim']
        learn_eps = configs['learn_eps']
        neighbor_pooling_type = configs['neighbor_pooling_type']
        # device = configs['device']

        self.n_job = configs['n_job']
        self.n_machine = configs['n_machine']
        self.n_total_task = self.n_job * self.n_machine

        self.heads = 1

        """
        图卷积网络：
        1、初始化GIN网络：暂不用学习参数e，简单的图卷积：自身+邻居节点（入度）：因为只有前一时刻才有影响，后续没调度你卷积干什么？？（TII的就很无语，他俩都是乱试）
        2、正常node vector也需要mlp的网络，但是这里没有用。直接用的特征向量（TII用的GAT时，用的mlp先转换！）
        3、我不用old网络：自身网络采集作为old网络，比较时不重新采集！（同eswa！）
        4、直接用configs，全局统一，防止直接传参数，传的不对到时候理解复杂！
        """
        self.encoder = Encoder(num_layers=gcn_layer,
                               num_mlp_layers=num_mlp_layers,
                               input_dim=gcn_input_dim,
                               hidden_dim=gcn_hidden_dim,
                               learn_eps=learn_eps,
                               neighbor_pooling_type=neighbor_pooling_type,
                               device=device).to(device)
        Logger.log("Training/ppo/init_global_critic_encoder", f"Global_Critic: self.encoder={self.encoder}", print_true=1)

        self.machine_hidden_dim = configs['machine_hidden_dim']

        self.bn = torch.nn.BatchNorm1d(self.machine_hidden_dim).to(device)  # 批归一化层

        self.m_fea_1_fcl = nn.Linear(6, self.machine_hidden_dim, bias=False).to(device)  # 全连接层来池化节点的特征，6in，128out，没有偏置项
        self.m_fea_2_fcl = nn.Linear(8, self.machine_hidden_dim, bias=False).to(device)  # 全连接层来池化节点的特征，6in，128out，没有偏置项
        self.activation = nn.ELU()  # 输入 x 中的负值将按照 ELU 的定义进行变换，而正值将保持不变 TODO 在m的不同维度转成hidden的FCL后边，新增一个激活函数，一层后跟一个！能这样子用？？？？
        # self.activation = nn.LeakyReLU()  #

        self.heads = 1

        # 初始化GAT网络
        self.gat_layer = GATLayer(in_features=self.machine_hidden_dim,  # 输入特征向量的维度；如果所有节点的维度不同，先进行一个FCL进行下统一！
                            out_features=self.machine_hidden_dim,  # 输出的维度
                            n_heads=self.heads,  # 几个多头，可以为1
                            concat=False,  # 是否对多头就行concate TODO 我不知道啥意思，但是我看默认都是false??
                            dropout=0,  # dropout的系数，防止过拟合！
                            leaky_relu_slope=0.2 # `leaky_relu_slope`参数确定了负输入的斜率大小。解决训练过程中神经元对负输入变得不响应的问题，这被称为“dying ReLU”问题
                            ).to(device)
        Logger.log("Training/ppo/init_global_gat_layer", f"Global_Critic: self.gat_layer={self.gat_layer}", print_true=1)

        # machine这里：没有图结构 + 节点之间不连接，所以直接用全连接层来池化/聚合节点的特征向量
        self.fcl_pooling = nn.Linear(self.machine_hidden_dim, self.machine_hidden_dim, bias=False).to(device)  # 全连接层来池化节点的特征，128in，128out，没有偏置项

        """
        critic的MLP网络
        """
        mlp_critic_layer = configs['mlp_critic_layer']
        critic_input_dim = configs['critic_input_dim']
        critic_hidden_dim = configs['critic_hidden_dim']
        # TODO 0109-消融实验-不使用r向量和v向量，那么动态的权重01其实没用，不改问题不大，看看效果
        # self.critic = MLPCritic(mlp_critic_layer, critic_input_dim * 2, critic_hidden_dim, 1).to(device)  # 2层 + 128in + 128隐藏 + 1out
        self.critic = MLPCritic(mlp_critic_layer, critic_input_dim * 2, critic_hidden_dim, 4).to(device)  # 3层 + 128in + 128隐藏 + 4out

        # 正交初始化！
        if configs['use_orthogonal']:  # 就是configs的init
            for name, p in self.named_parameters():  # 遍历模型的所有参数，通过`self.named_parameters()`方法获取参数的名称和值。
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p,
                                            gain=1)  # 对于参数名称中包含’weight’的参数，判断其维度是否大于等于2。如果是，则使用`nn.init.orthogonal_`方法对该参数进行正交初始化，其中`gain`参数指定了初始化的增益。
                        Logger.log("Training/ppo/init_v_orthogonal_gcn_weight", f"Global_Critic: orthogonal_gain=1", print_true=1)
                        
                elif 'bias' in name:
                    nn.init.constant_(p, 0)  # 对于参数名称中包含’bias’的参数，使用`nn.init.constant_`方法将其初始化为常数0。
                    Logger.log("Training/ppo/init_v_orthogonal_gcn_bias", f"Global_Critic: constant_(p, 0)", print_true=1)

    def forward(self,
                x_fea,
                graph_pool_avg,
                adj,
                candidate,
                machine_fea1,
                machine_fea2
                ):

        # Logger.log("Training/ppo/global_critic_forward_input", f"x_fea={x_fea.shape}, graph_pool_avg={graph_pool_avg.shape}, adj={adj.shape}, candidate={candidate.shape}, machine_fea1={machine_fea1.shape}, machine_fea2={machine_fea2.shape}", print_true=1)
        
        """
        注意： critic网络里边的传入参数，都是从memory里边读取的，已经是张量了啊！！！！!!!

        :param x_fea:  # task节点的特征向量：batch*task，9
        :param graph_pool_avg:   # 全图节点嵌入求均值的矩阵：batch，batch*task
        :param adj:  # 邻居矩阵（带权重+j到i+自身置1）batch*task*task  （后续会做处理的！变成bs*task，bs*task）
        :param candidate:  # 记录的当前可选的候选节点的index = batch*job
        :param machine_fea1:  当前task的：传入machine网络的m节点的特征向量 = env_batch * m * 6 （np.array）
        :param machine_fea2:  env反馈的：传入machine网络的m节点的特征向量 = env_batch * m * 8 （np.array）
        :return:
        """

        if torch.is_tensor(x_fea):  # 判断是否为张量，因为在update里边，buffer里边读取的都已经是tensor了

            x_fea = copy.copy(x_fea).float().to(device)  # 原deepcopy：这部分代码首先使用`np.copy`创建了`fea`的副本，然后使用`torch.from_numpy`将其转换为PyTorch张量
            candidate = copy.copy(candidate).long().to(device)  # 原deepcopy：转为long整型的张量 batch * job
            machine_fea1 = copy.copy(machine_fea1).float().to(device)  # 原deepcopy：转为张量+gpu  (env_batch * m * 6)
            machine_fea2 = copy.copy(machine_fea2).float().to(device)  # 原deepcopy：转为张量+gpu  (env_batch * m * 6)
            # print("Global_Critic: forward： machine_fea = ", machine_fea, machine_fea.shape)
            """
            # all batch的所有的邻居矩阵，adj = torch.Size([4, 16, 16])  batch*tasks*tasks 张量
            # 然后aggr_obs聚合成大的稀疏批邻居矩阵
            # 估计是转成大型对角矩阵的形式，稀疏矩阵表示，即64*64 = batch*tasks, batch*tasks

            我传入的adj是np.array，所以我要先转成tensor，再计算!!!
            """
            # print("--观察数据：adj = ", adj, adj.shape)  # adj已经是张量了？？？我传入的还是np.array
            adj = aggr_obs(copy.copy(adj).to(device).to_sparse(),
                           self.n_total_task)  # 原deepcopy：这行代码首先使用`deepcopy`函数创建了`adj`的副本，然后将其转移到指定的`device`上，并将其稀疏化处理。最后，使用`aggr_obs`函数对稀疏的`adj`进行聚合操作，生成了`env_adj`
        else:
            x_fea = torch.from_numpy(np.copy(x_fea)).float().to(device)  # 这部分代码首先使用`np.copy`创建了`fea`的副本，然后使用`torch.from_numpy`将其转换为PyTorch张量
            candidate = torch.from_numpy(np.copy(candidate)).long().to(device)  # 转为long整型的张量 batch * job
            machine_fea1 = torch.from_numpy(np.copy(machine_fea1)).float().to(device)  # 转为张量+gpu  (env_batch * m * 6)
            machine_fea2 = torch.from_numpy(np.copy(machine_fea2)).float().to(device)  # 转为张量+gpu  (env_batch * m * 6)
            adj = aggr_obs(torch.from_numpy(copy.copy(adj)).to(device).to_sparse(),
                           self.n_total_task)  # 原deepcopy：这行代码首先使用`deepcopy`函数创建了`adj`的副本，然后将其转移到指定的`device`上，并将其稀疏化处理。最后，使用`aggr_obs`函数对稀疏的`adj`进行聚合操作，生成了`env_adj`

        """
        求出task节点的全图平均池化嵌入：
        1、h_g_o_pooled: shape = 维度4（batch） * 128（hidden_dim）  （原始未整形）
        2、h_o_nodes = shape = bs*task,hidden
        """
        # 3层gin网络卷积之后，
        # 反馈：全图的节点嵌入(维度4（batch） * 128) + 各个节点的嵌入（64（batch*tasks） * 128）
        h_g_o_pooled, h_o_nodes = self.encoder(x=x_fea,  # 这里直接就GIN的节点特征，和全局的池化特征都有了
                                               graph_pool=graph_pool_avg,
                                               padded_nei=None,
                                               adj=adj)
        # Logger.log("Training/ppo/global_critic_forward_encoder", f"h_g_o_pooled={h_g_o_pooled.shape}, h_o_nodes={h_o_nodes.shape}", print_true=1)

        # batch*job --bs*j*1--bs*j*hidden
        dummy = candidate.unsqueeze(-1).expand(-1, self.n_job,
                                               h_o_nodes.size(-1))  # batch，job，128。最后一个维度扩维 = 4,4,1，然后复制元素变成=4,4,128
        # Logger.log("Training/ppo/global_critic_forward_dummy", f"new: dummy={dummy.shape}, old：candidate={candidate.shape}", print_true=1)

        # h_o_nodes = shape = bs*task,hidden--bs*task*hidden + dummy = bs*j*hidden ------最后输出=bs*j*hidden
        candidate_feature = torch.gather(h_o_nodes.reshape(dummy.size(0), -1, dummy.size(-1)),
                                         1,
                                         dummy)  # 类似于从每个样本中，挑选对应candidate中可选task的编号的节点的嵌入给挑选出来，重新组成变量！！！
        # Logger.log("Training/ppo/global_critic_forward_candidate_feature", f"new: candidate_feature={candidate_feature.shape}", print_true=1)

        """
        求出m节点的全图平均池化嵌入：
        1、h_g_m_pooled: shape = (batch_size, hidden_size)  （原始未整形）
        2、h_m_nodes = shape = bs * m * 128
        """

        """将不同维度的m_fea转换成一致，然后按照（bs*m，2, hidden）进行转换： 代表两个m节点的聚合，2-->1"""
        m_fea1 = self.m_fea_1_fcl(machine_fea1)  # bs * m * 128(hidden)
        m_fea2 = self.m_fea_2_fcl(machine_fea2)
        
        # 将 A 和 B 展平为形状为 (bs * m) 的一维张量
        m_fea1 = m_fea1.view(m_fea1.shape[0] * m_fea1.shape[1], m_fea1.shape[2])  # 三维变2维度：bs*m，hidden
        m_fea2 = m_fea2.view(m_fea2.shape[0] * m_fea2.shape[1], m_fea2.shape[2])
        # Logger.log("Training/ppo/global_critic_forward_m_fea12", f"m_fea1={m_fea1.shape}, m_fea2={m_fea2.shape}", print_true=1)
        
        # 交替堆叠 m_fea1 和 m_fea2的第二个维度的元素值，会变成 = bs*m，2，hidden，不放心再reshape一下！
        All_sameM_fea12 = torch.stack([m_fea1, m_fea2], dim=1).reshape(-1, 2, m_fea1.shape[-1])  # TODO 只有同个m的两个不同特征值进行聚合！(类比task_fea = bs*task，9的所有task和所有bs的特征值)
        # Logger.log("Training/ppo/global_critic_forward_gat_in", f"All_sameM_fea12={All_sameM_fea12.shape}", print_true=1)

        """
        h = bs*m，2，hidden
        adj_m=(bs*m, head, 2, 2) 且ij=1表示j->i! + 默认有向边2->1，所以某一行全0，固定的
        """
        # TODO 修改adj = [[1, 1], [0, 1]]让m_fea2自己聚合自己，虽然我不用这一行！（全0行变成全-inf行，然后softmax该行=Nan！）
        # 创建一个形状为 (bs * m) * head * 2 * 2 的张量
        adj_m = torch.zeros(All_sameM_fea12.shape[0], self.heads, 2, 2).to(device)
        # 将 2*2 的矩阵的第一行元素设置为 1
        adj_m[:, :, 0, :] = 1  # 第一行都是1
        adj_m[:, :, 1, 1] = 1  # 第二行第二列=1
        # Logger.log("Training/ppo/global_critic_forward_adj_m", f"adj_m={adj_m.shape}", print_true=1)

        """h_prime_gat = （bs*m, 2, hidden）--- 其中2*hidden的第一行是我们想要的，第二行是m_fea2的自身聚合，我们只用第一行的参数 [:,0,:]= （bs*m, hidden）--reshape = （bs, m ,hiddden）"""
        h_prime_gat = self.activation(self.gat_layer(h=All_sameM_fea12, adj_mat=adj_m))  # TODO 严格按照公式，和Wh加权聚合之后，需要一个ELU激活！！输出m节点嵌入特征值
        # TODO GAT可以循环n次的！对应GCN，那就3次！
        h_prime_gat = self.activation(self.gat_layer(h=h_prime_gat, adj_mat=adj_m))
        h_prime_gat = self.gat_layer(h=h_prime_gat, adj_mat=adj_m)
        # Logger.log("Training/ppo/global_critic_forward_h_prime_gat", f"Gat_out: h_prime_gat={h_prime_gat.shape}", print_true=1)

        # TODO 为什么要单独只用一个？两个都用，或者全图的特征不是可以吗？（FCL或MLPActor的需求：bs*m*hidden）
        # h_machine_feas = h_prime_gat[:,0,:].reshape(-1, self.n_machine, self.machine_hidden_dim) # （bs, m ,hiddden）
        # TODO 参考GCN中的Average方式，求均值mean（dim=-2）！也可以reshape（bs*m，2*hidden）--out的时候只有一个hidden也行！
        h_machine_feas = h_prime_gat.mean(dim=-2).reshape(-1, self.n_machine,self.machine_hidden_dim)  # （bs, m ,hiddden）
        # Logger.log("Training/ppo/global_critic_forward_h_machine_feas", f"h_machine_feas={h_machine_feas.shape}", print_true=1)
        
        # 直接数据放缩！考虑要不要加？因为p*t会很大！
        # machine_fea = machine_fea / configs['']et_normalize_coef  # 归一化处理: 直接原数据/1000 batch * m  当前所选择m的上一个工序的完工时间

        """
        1、fea直接输入到1线性层中，结果batch*m*128，转成2维：（batch*m）*128，输入到bn层中，结果再转回来：batch * m * 128
        2、类似于对m的节点进行了encoder，变成了池化之后节点嵌入 = (batch_size, n_m, hidden_size)
        """
        # TODO 此时就用一用bn层吧，fcl没必要了！
        # h_m_nodes = self.bn(self.fcl_pooling(h_machine_feas).reshape(-1, self.machine_hidden_dim)).reshape(
        h_m_nodes = self.bn(h_machine_feas.reshape(-1, self.machine_hidden_dim)).reshape(
                                                                                        -1,
                                                                                        self.n_machine,
                                                                                        self.machine_hidden_dim)
        # Logger.log("Training/ppo/global_critic_forward_h_m_nodes", f"h_m_nodes={h_m_nodes.shape}", print_true=1)

        """
        计算machine节点网络的全图特征：
        1、计算 `h_m_nodes` 张量在第2个维度上的均值，即对每个样本的 `n_m` 个节点特征进行平均。得到的 `pool` 张量形状为 `(batch_size, hidden_size)
        2、相当于我首先节点池化了一波，每个m都有128个特征，然后我对每个batch中的all m求他的平均的特征表征（类似于全图中各节点的嵌入的均值，池化过程！）
        """
        h_g_m_pooled = h_m_nodes.mean(dim=1)  # (batch_size, hidden_size)
        # Logger.log("Training/ppo/global_critic_forward_h_g_m_pooled", f"h_g_m_pooled={h_g_m_pooled.shape}", print_true=1)
        # print("观察数据：Global_Critic：h_pooled = ", h_g_m_pooled, h_g_m_pooled.shape)

        # TODO Critic为了和Actor严格对应，随意传入的数据都要一样！维度不一样怎么办？？？？？？？？？？？？？？？？？先不改了，还是两个全图特征！！！！
        """
        合并特征值：
        用：h_g_o_pooled = batch * hidden 全batch的task节点的全图的节点池化均值！
        candidate = bs * j * hidden  当前的候选task节点
        用：h_g_m_pooled = batch * hidden 全batch的m节点的全图的节点池化均值！
        m_nodes = bs * m * hidden  当前（所有）的m节点
        """
        # 合并输入的state： shape = batch * hidden （两个都一样）
        concateFea = torch.cat((h_g_m_pooled, h_g_o_pooled),
                               dim=-1)  # 在最后一个维度进行合并：4,128*2 （batch * hidden*2）
        # Logger.log("Training/ppo/global_critic_forward_concateFea", f"concateFea={concateFea.shape}", print_true=1)

        """
        唯一的和value有关的
        h_g_m_pooled = batch * hidden 全batch的m节点的全图的节点池化均值！
        h_g_o_pooled = batch * hidden 全batch的task节点的全图的节点池化均值！
        只有全图的信息输入到value中！！！！
        输出：2层+128*2in+128隐+4out  batch*4
        """
        v = self.critic(concateFea)
        # Logger.log("Training/ppo/global_critic_forward_global_v", f"v={v.shape}", print_true=1)  # batch*4 = 对应4个reward
        
        return v  # TODO 如果修改了value的输出维度=4，则shape = batch*4



class ablation_Machine_Actor_JointAction_selfGAT_selfCritic(nn.Module):
    def __init__(self, configs):
        super(ablation_Machine_Actor_JointAction_selfGAT_selfCritic, self).__init__()

        
        self.n_machine = configs['n_machine']
        

        mlp_actor_layer = configs['mlp_actor_layer']
        self.machine_hidden_dim = configs['machine_hidden_dim']

        self.bn = torch.nn.BatchNorm1d(self.machine_hidden_dim).to(device)  # 批归一化层

        self.m_fea_1_fcl = nn.Linear(6, self.machine_hidden_dim, bias=False).to(device)  # 全连接层来池化节点的特征，6in，128out，没有偏置项
        self.m_fea_2_fcl = nn.Linear(8, self.machine_hidden_dim, bias=False).to(device)  # 全连接层来池化节点的特征，6in，128out，没有偏置项
        self.activation = nn.ELU() #输入 x 中的负值将按照 ELU 的定义进行变换，而正值将保持不变 TODO 在m的不同维度转成hidden的FCL后边，新增一个激活函数，一层后跟一个！能这样子用？？？？
        # self.activation = nn.LeakyReLU() #

        self.heads = 1

        # 初始化GAT网络
        self.gat_layer = GATLayer(in_features=self.machine_hidden_dim,  # 输入特征向量的维度；如果所有节点的维度不同，先进行一个FCL进行下统一！
                            out_features=self.machine_hidden_dim,  # 输出的维度
                            n_heads= self.heads,  # 几个多头，可以为1
                            concat= False,  # 是否对多头就行concate TODO 我不知道啥意思，但是我看默认都是false??
                            dropout= 0,  # dropout的系数，防止过拟合！
                            leaky_relu_slope= 0.2  # `leaky_relu_slope`参数确定了负输入的斜率大小。解决训练过程中神经元对负输入变得不响应的问题，这被称为“dying ReLU”问题
                            ).to(device)

        # machine这里：没有图结构 + 节点之间不连接，所以直接用全连接层来池化/聚合节点的特征向量
        #TODO 这里的m节点的特征聚合，直接一个线性层就ok了？GIN的MLP可是循环3次的！！！！！！！ + 为什么没有bias？？？
        # 我怀疑这个网络都没有起到很大的作用！！！

        # self.fcl_pooling = nn.Linear(3, self.machine_hidden_dim, bias=False).to(device)  # 全连接层来池化节点的特征，6in，128out，没有偏置项
        self.fcl_pooling = nn.Linear(self.machine_hidden_dim, self.machine_hidden_dim, bias=False).to(device)  # GAT聚合之后，全连接层来池化节点的特征，128in，128out，没有偏置项
        self.m_policy = MLPActor(mlp_actor_layer, self.machine_hidden_dim * 3, self.machine_hidden_dim, 1).to(device)  # 选择m的actor网络：共3层（包括输入）+128*2in+128hidden+1out

        """
        critic的MLP网络
        """
        mlp_critic_layer = configs['mlp_critic_layer']
        critic_input_dim = configs['critic_input_dim']
        critic_hidden_dim = configs['critic_hidden_dim']
        # self.critic = MLPCritic(mlp_critic_layer, critic_input_dim * 2, critic_hidden_dim, 1).to(device)  # 2层 + 128in + 32隐藏 + 1out
        self.machine_critic = MLPCritic(mlp_critic_layer, critic_input_dim, critic_hidden_dim, 2).to(device)  # 3层 + 128in + 32隐藏 + 1out  TODO 全图特征的输入= 1*128 + pt和tt的反馈reward = 2！

        # 正交初始化！
        if configs['use_orthogonal']:  # 就是configs的init
            for name, p in self.named_parameters():  # 遍历模型的所有参数，通过`self.named_parameters()`方法获取参数的名称和值。
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p,
                                            gain=1)  # 对于参数名称中包含’weight’的参数，判断其维度是否大于等于2。如果是，则使用`nn.init.orthogonal_`方法对该参数进行正交初始化，其中`gain`参数指定了初始化的增益。
                        print("Machine Actor init.orthogonal_ gain=1")
                elif 'bias' in name:
                    nn.init.constant_(p, 0)  # 对于参数名称中包含’bias’的参数，使用`nn.init.constant_`方法将其初始化为常数0。
                    print("Machine Actor init.constant_(p, 0)")

    def forward(self, machine_fea_1, machine_fea_2, h_pooled_o, machine_mask):  # m_fea2 指向 m_fea1： 有向边
        """

        :param machine_fea_1:  (当前task的对应的m的特征)传入machine网络的m节点的特征向量 = env_batch * m * 6 （np.array）
        :param machine_fea_2:  （选好m输入到ENV中的被选更新的state）传入machine网络的m节点的特征向量 = env_batch * m * 5 （np.array）
        :param h_pooled_o: 传入的上一个step的task节点的全图特征 = env_batch * hidden
        :param machine_mask: 当前job选择的task的mask = env_batch * 1 * m  好像直接就是张量？？？
        :return:
        """

        if torch.is_tensor(machine_fea_1):  # 判断是否为张量，因为在update里边，buffer里边读取的都已经是tensor了
            machine_fea_1 = copy.copy(machine_fea_1).float().to(device)  # 原deepcopy：转为张量+gpu  (env_batch * m * 6)
            machine_fea_2 = copy.copy(machine_fea_2).float().to(device)  # 原deepcopy：转为张量+gpu  (env_batch * m * 6)
        else:
            machine_fea_1 = torch.from_numpy(np.copy(machine_fea_1)).float().to(device)  # 转为张量+gpu  (env_batch * m * 6)
            machine_fea_2 = torch.from_numpy(np.copy(machine_fea_2)).float().to(device)  # 转为张量+gpu  (env_batch * m * 6)
        # print("Machine_Actor: forward： machine_fea = ", machine_fea, machine_fea.shape)
        # print("MActor: machine_fea_1 = ", machine_fea_1, machine_fea_1.shape)

        """将不同维度的m_fea转换成一致，然后按照（bs*m，2, hidden）进行转换： 代表两个m节点的聚合，2-->1"""
        # m_fea1 = self.activation(self.m_fea_1_fcl(machine_fea_1)) # bs * m * 128(hidden) TODO 加上了激活函数ELU???????
        # m_fea2 = self.activation(self.m_fea_2_fcl(machine_fea_2))
        m_fea1 = self.m_fea_1_fcl(machine_fea_1)  # bs * m * 128(hidden)
        m_fea2 = self.m_fea_2_fcl(machine_fea_2)
        # print("MActor: m_fea1 = ",m_fea1, m_fea1.shape)
        # print("MActor: m_fea2 = ",m_fea2, m_fea2.shape)

        # 将 A 和 B 展平为形状为 (bs * m) 的一维张量
        m_fea1 = m_fea1.view(m_fea1.shape[0] * m_fea1.shape[1], m_fea1.shape[2])  # 三维变2维度：bs*m，hidden
        m_fea2 = m_fea2.view(m_fea2.shape[0] * m_fea2.shape[1], m_fea2.shape[2])
        # print("MActor: reshape: m_fea1 = ", m_fea1, m_fea1.shape)
        # print("MActor: reshape: m_fea2 = ", m_fea2, m_fea2.shape)

        # 交替堆叠 m_fea1 和 m_fea2的第二个维度的元素值，会变成 = bs*m，2，hidden，不放心再reshape一下！
        All_sameM_fea12 = torch.stack([m_fea1, m_fea2], dim=1).reshape(-1, 2, m_fea1.shape[-1])  # TODO 只有同个m的两个不同特征值进行聚合！(类比task_fea = bs*task，9的所有task和所有bs的特征值)
        # print("MActor: reshape: All_sameM_fea12 = ", All_sameM_fea12, All_sameM_fea12.shape)

        """
        h = bs*m，2，hidden
        adj_m=(bs*m, head, 2, 2) 且ij=1表示j->i! + 默认有向边2->1，所以某一行全0，固定的
        """
        # 创建一个形状为 (bs * m) * head * 2 * 2 的张量
        # TODO 修改adj = [[1, 1], [0, 1]]让m_fea2自己聚合自己，虽然我不用这一行！（全0行变成全-inf行，然后softmax该行=Nan！）
        adj_m = torch.zeros(All_sameM_fea12.shape[0], self.heads, 2, 2).to(device)
        # 将 2*2 的矩阵的第一行元素设置为 1
        adj_m[:, :, 0, :] = 1  # 第一行都是1
        adj_m[:, :, 1, 1] = 1  # 第二行第二列=1

        """h_prime_gat = （bs*m, 2, hidden）--- 其中2*hidden的第一行是我们想要的，第二行是m_fea2的自身聚合，我们只用第一行的参数 [:,0,:]= （bs*m, hidden）--reshape = （bs, m ,hiddden）"""
        h_prime_gat = self.activation(self.gat_layer(h=All_sameM_fea12, adj_mat=adj_m))  # TODO 严格按照公式，和Wh加权聚合之后，需要一个ELU激活！！输出m节点嵌入特征值

        # TODO GAT可以循环n次的！对应GCN，那就3次！
        h_prime_gat = self.activation(self.gat_layer(h=h_prime_gat, adj_mat=adj_m))
        h_prime_gat = self.gat_layer(h=h_prime_gat, adj_mat=adj_m)

        # TODO 为什么要单独只用一个？两个都用，或者全图的特征不是可以吗？（FCL或MLPActor的需求：bs*m*hidden）
        # h_machine_feas = h_prime_gat[:,0,:].reshape(-1, self.n_machine, self.machine_hidden_dim) # （bs, m ,hiddden）
        # TODO 参考GCN中的Average方式，求均值mean（dim=-2）！也可以reshape（bs*m，2*hidden）--out的时候只有一个hidden也行！
        h_machine_feas = h_prime_gat.mean(dim=-2).reshape(-1, self.n_machine, self.machine_hidden_dim) # （bs, m ,hiddden）  聚合方式！

        # print("MActor------单独取第一行的--------h_machine_feas =\n",h_machine_feas,h_machine_feas.shape)

        # 直接数据放缩！考虑要不要加？因为p*t会很大！
        # machine_fea = machine_fea / configs['']et_normalize_coef  # 归一化处理: 直接原数据/1000 batch * m  当前所选择m的上一个工序的完工时间

        # fea直接输入到1线性层中，结果batch*m*128，转成2维：（batch*m）*128，输入到bn层中，结果再转回来：batch * m * 128
        # 类似于对m的节点进行了encoder，变成了池化之后节点嵌入 = (batch_size, n_m, hidden_size)
        # TODO 此时就用一用bn层吧，fcl没必要了！---- 0108-消融实验，只用1个m的state，然后没有图结构，只能直接MLP！！！
        h_m_nodes = self.bn(self.m_fea_1_fcl(machine_fea_1).reshape(-1, self.machine_hidden_dim)).reshape(-1,
                                                                                                           self.n_machine,
                                                                                                           self.machine_hidden_dim)
        # h_m_nodes = self.bn(self.fcl_pooling(h_machine_feas).reshape(-1, self.machine_hidden_dim)).reshape(-1,
        #                                                                                                 self.n_machine,
        #                                                                                                 self.machine_hidden_dim)
        """h_m_nodes = self.bn(h_machine_feas.reshape(-1, self.machine_hidden_dim)).reshape(-1,
                                                                                       self.n_machine,
                                                                                       self.machine_hidden_dim)"""

        # 计算 `h_m_nodes` 张量在第二个维度上的均值，即对每个样本的 `n_m` 个节点特征进行平均。得到的 `pool` 张量形状为 `(batch_size, hidden_size)`。
        # 相当于我首先节点池化了一波，每个m都有128个特征，然后我对每个batch中的all m求他的平均的特征表征（类似于全图中各节点的嵌入的均值，池化过程！）
        h_pooled = h_m_nodes.mean(dim=1)  # (batch_size, hidden_size)

        """"
        下述的变换操作：
        1、是为了和各个m节点的嵌入进行合并：batch*m*128
        2、从而进行expand的复制操作
        """
        # 这一行代码使用 `unsqueeze(1)` 在第二个维度上添加一个维度，并使用 `expand_as` 函数将 `pool` 张量复制扩展成和 `action_node` 张量相同的形状。
        # 得到的 `h_pooled_repeated` 张量形状为 `(batch_size, n_m, hidden_size)`。
        # 相当于：每个batch中的全m的节点嵌入复制了m=4份
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(h_m_nodes)  # (batch_size, n_m, hidden_size)  m节点的全图节点嵌入的均值

        # h_poooled_m: batch * hidden_dim GIN之后的上一个状态的全图的节点嵌入的均值
        # 同上：第2维度加1，然后expand复制变成(batch_size, n_m, hidden_size) 4*4*128
        pooled_repeated = h_pooled_o.unsqueeze(1).expand_as(h_m_nodes) # (batch_size, n_m, hidden_size)  o节点的全图节点嵌入的均值

        # 维度不够就要进行expand的复制补充！
        # 输入变量： m节点的池化嵌入 + 所有m节点全局嵌入（mean） + 选择job的全图的all节点的嵌入（mean，暂无，先去掉！） ： batch * m * 128*2
        concateFea = torch.cat((h_m_nodes, h_pooled_repeated, pooled_repeated), dim=-1)  # 最后一个维度合并
        # print("MCH_Actor查看数据： concateFea = ", concateFea, concateFea.shape)  # batch * m * 128*2

        mch_scores = self.m_policy(concateFea)  # batch * m * 1 输出的是选择哪一个m的得分：放入到了MLPActor的网络中
        mch_scores = mch_scores.squeeze(-1) * 10  # 去掉最后一个维度：batch * m。数值放大10倍
        # mask_reshape = mask_mch_action.reshape(candidate_scores.size())
        # print("MCH_Actor查看数据： 还没没有mask：mch_scores = ", mch_scores, mch_scores.shape)

        """选择m的mask：暂无，省略此步"""
        # TODO: add machine选择时候的mask
        # machine_mask: batch * 1 * m 所选择的task的对应的mask（false可选）
        # squeeze（1）去掉维度为1的：batch * m，bool（）转换成布尔张量（若已经是bool，保持不变）：这将把非零值转换为 `True`，零值转换为 `False`。
        # 使用 `masked_fill` 函数，根据掩码张量的值将 `mch_scores` 张量中对应位置的元素填充为负无穷大。掩码张量中为 `True` 的位置将被填充，而为 `False` 的位置将保持不变。
        # print("MCH_Actor查看数据： mask之后： machine_mask = ", machine_mask, machine_mask.shape)
        mch_scores = mch_scores.masked_fill(machine_mask.squeeze(1).bool(), float("-inf"))
        # print("MCH_Actor查看数据： mask之后： mch_scores = ", mch_scores, mch_scores.shape)


        # 掩码加持之后的mch_scores，进行softmax输出选择m的概率：batch * m 数值变成概率了
        mch_prob = F.softmax(mch_scores, dim=-1)
        # print("MCH_Actor查看数据： mch_prob = ", mch_prob, mch_prob.shape)

        # TODO 1113-新增局部的machine的critic，只关注pt和tt
        machine_v = self.machine_critic(h_pooled)  # h_pooled = bs * hidden
        # print("观察数据：machine_Critic：machine_v = ", machine_v, machine_v.shape) # batch*2

        return mch_prob, h_pooled, machine_v  # 返回的是batch*m的选择m的概率 + 返回所有m节点的全局嵌入（求均值）：batch * hidden_dim



# TODO 1223-===========  以下是ESA[1]的复现代码-嵌入到我的场景！  =======================================================================================================================================================================

"""
按照全局和局部的critic的方式，进行重新写
"""
class esa_Operation_Actor_Critic (nn.Module):
    def __init__(self, configs):
        super(esa_Operation_Actor_Critic, self).__init__()

        self.n_job = configs['n_job']  # 后续想调用的，可以用self.的形式，其他没必要
        self.n_machine = configs['n_machine']
        self.n_total_task = self.n_job * self.n_machine
        self.batch_size = configs['env_batch']

        self.GAMMA = configs['GAMMA']  # reward折扣率
        self.LAMDA = configs['LAMDA']  # GAE参数
        self.epsilon = configs['epsilon']  # 重要性采样的裁剪
        self.ENTROPY_BETA = configs['ENTROPY_BETA']  # 偏向于出现0001，那就调小！！！！  平衡熵正则化

        gcn_layer = configs['gcn_layer']
        num_mlp_layers = configs['mlp_fea_extract_layer']
        gcn_input_dim = configs['gcn_input_dim']
        gcn_hidden_dim = configs['gcn_hidden_dim']
        learn_eps = configs['learn_eps']
        neighbor_pooling_type = configs['neighbor_pooling_type']
        # device = configs['']device

        mlp_actor_layer = configs['mlp_actor_layer']

        """
        图卷积网络：
        1、初始化GIN网络：暂不用学习参数e，简单的图卷积：自身+邻居节点（入度）：因为只有前一时刻才有影响，后续没调度你卷积干什么？？（TII的就很无语，他俩都是乱试）
        2、正常node vector也需要mlp的网络，但是这里没有用。直接用的特征向量（TII用的GAT时，用的mlp先转换！）
        3、我不用old网络：自身网络采集作为old网络，比较时不重新采集！（同eswa！）
        4、直接用configs，全局统一，防止直接传参数，传的不对到时候理解复杂！
        """
        self.encoder = Encoder(num_layers=gcn_layer,
                               num_mlp_layers=num_mlp_layers,
                               input_dim=gcn_input_dim,
                               hidden_dim=gcn_hidden_dim,
                               learn_eps=learn_eps,
                               neighbor_pooling_type=neighbor_pooling_type,
                               device=device).to(device)
        # print("GCN网络打印: Operation_Actor: self.encoder = ", self.encoder)

        self._input = nn.Parameter(torch.Tensor(configs['gcn_hidden_dim']))  # 创建一个名为`_input`的`nn.Parameter`对象，它是一个可学习的张量参数。torch.Tensor(hidden_dim)创建一个形状为`(hidden_dim,)`的张量
        self._input.data.uniform_(-1, 1).to( device)  # 使用`uniform_(-1, 1)`方法对`_input.data`进行均匀分布的随机初始化。`uniform_(-1, 1)`会将`_input.data`中的元素从均匀分布中进行随机采样。

        """
        # class MLPActor(nn.Module):
        #     def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        1、暂时输入参数：待选节点嵌入 + 全图节点嵌入 + M节点全图嵌入
        """
        # self.o_policy = MLPActor(mlp_actor_layer, gcn_hidden_dim * 3, gcn_hidden_dim, 1).to(device)  # 几层 + in + hidden + out
        self.o_policy = MLPActor(mlp_actor_layer, gcn_hidden_dim * 3, gcn_hidden_dim, 1).to(device)  # 几层 + in + hidden + out

        # TODO 1223-job的critic正好对应的了ESA的critic，所以说不用修改，直接使用，注意在loss里边的用法！

        """
        critic的MLP网络
        """
        mlp_critic_layer = configs['mlp_critic_layer']
        critic_input_dim = configs['critic_input_dim']
        critic_hidden_dim = configs['critic_hidden_dim']
        # self.critic = MLPCritic(mlp_critic_layer, critic_input_dim * 2, critic_hidden_dim, 1).to(device)  # 2层 + 128in + 32隐藏 + 1out
        # TODO 1223-ESA方法复现：这里的critic要输出4个变量，对应4个reward！
        self.job_critic = MLPCritic(mlp_critic_layer, critic_input_dim, critic_hidden_dim, 4).to(device)  # 3层 + 128in + 32隐藏 + 1out  TODO 全图特征的输入= 1*128 + mk和it的反馈reward = 2！

        # 正交初始化！
        if configs['use_orthogonal']:  # 就是configs的init
            for name, p in self.named_parameters():  # 遍历模型的所有参数，通过`self.named_parameters()`方法获取参数的名称和值。
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p,
                                            gain=1)  # 对于参数名称中包含’weight’的参数，判断其维度是否大于等于2。如果是，则使用`nn.init.orthogonal_`方法对该参数进行正交初始化，其中`gain`参数指定了初始化的增益。
                        print("GCN init.orthogonal_ gain=1")
                elif 'bias' in name:
                    nn.init.constant_(p, 0)  # 对于参数名称中包含’bias’的参数，使用`nn.init.constant_`方法将其初始化为常数0。
                    print("GCN init.constant_(p, 0)")

    def forward(self,
                x_fea,  # task节点的特征向量：batch*task，4
                graph_pool_avg,  # 全图节点嵌入求均值的矩阵：batch，batch*task
                padded_nei,  # max的pooling才有用
                adj,  # 邻居矩阵（带权重+j到i+自身置1）batch*task，batch*task
                candidate,  # 可选task的id，batch，job
                h_g_m_pooled,  # 传入每一step的m节点的全图节点嵌入均值：batch，hidden
                mask_operation,  # 传入当前可选节点的mask位，某一job选完了，那就置为true：batch，job
                use_greedy=False
                ):

        # print('sssssssssssssssssssssss',x.size(),graph_pool.size(),padded_nei,adj.size(),candidate.size(),mask.size())

        """
        传入的是np.array，需要转成tensor + GPU
        :param x_fea:  batch*task，4
        :param graph_pool_avg: batch，batch*task  （已经是稀疏张量）
        :param padded_nei:
        :param adj:  batch，task，task
        :param candidate:  batch，job
        :param h_g_m_pooled:  batch，hidden （上一个net的输出：已经是张量）  不对应啊，123的选m，和按概率选o，这没法一起训练吧？？？
        :param mask_operation: batch，job (已经是张量)
        :param use_greedy: 默认是False
        :return:
        """

        # print("1.1:{}".format(get_GPU_usage()[1])) # 返回值的第二个参数就是显存使用量
        if torch.is_tensor(x_fea):  # 判断是否为张量，因为在update里边，buffer里边读取的都已经是tensor了
            x_fea = copy.copy(x_fea).float().to(device)  # 原deepcopy：这部分代码首先使用`np.copy`创建了`fea`的副本，然后使用`torch.from_numpy`将其转换为PyTorch张量
            candidate = copy.copy(candidate).long().to(device)  # 原deepcopy：转为long整型的张量 batch * job
            adj = aggr_obs(copy.copy(adj).to(device).to_sparse(),
                           self.n_total_task)  # 原deepcopy：这行代码首先使用`deepcopy`函数创建了`adj`的副本，然后将其转移到指定的`device`上，并将其稀疏化处理。最后，使用`aggr_obs`函数对稀疏的`adj`进行聚合操作，生成了`env_adj`
        else:
            x_fea = torch.from_numpy(np.copy(x_fea)).float().to(device)  # 这部分代码首先使用`np.copy`创建了`fea`的副本，然后使用`torch.from_numpy`将其转换为PyTorch张量
            candidate = torch.from_numpy(np.copy(candidate)).long().to(device)  # 转为long整型的张量 batch * job
            # mask_operation = torch.from_numpy(np.copy(mask_operation)).to(device)  # 转为张量，可选节点的mask：batch* job
            """
            # all batch的所有的邻居矩阵，adj = torch.Size([4, 16, 16])  batch*tasks*tasks 张量
            # 然后aggr_obs聚合成大的稀疏批邻居矩阵
            # 估计是转成大型对角矩阵的形式，稀疏矩阵表示，即64*64 = batch*tasks, batch*tasks

            我传入的adj是np.array，所以我要先转成tensor，再计算!!!
            """
            # print("--观察数据：adj = ", adj, adj.shape)  # adj已经是张量了？？？我传入的还是np.array
            adj = aggr_obs(torch.from_numpy(copy.copy(adj)).to(device).to_sparse(),
                           self.n_total_task)  # 原deepcopy：这行代码首先使用`deepcopy`函数创建了`adj`的副本，然后将其转移到指定的`device`上，并将其稀疏化处理。最后，使用`aggr_obs`函数对稀疏的`adj`进行聚合操作，生成了`env_adj`

        mask_operation = copy.copy(mask_operation).to(device)  # 原deepcopy：已经是张量，可选节点的mask：batch* job

        # print("1.2:{}".format(get_GPU_usage()[1]))

        # 3层gin网络卷积之后，
        # 反馈：全图的节点嵌入(维度4（batch） * 128) + 各个节点的嵌入（64（batch*tasks） * 128）
        h_g_o_pooled, h_o_nodes = self.encoder(x=x_fea,  # 这里直接就GIN的节点特征，和全局的池化特征都有了
                                               graph_pool=graph_pool_avg,
                                               padded_nei=padded_nei,
                                               adj=adj)
        # print("观察数据：h_o_nodes = ", h_o_nodes, h_o_nodes.shape)
        """
        现在的动作直接是对可选的node进行打分：
        1、传入所有env_batch中可选的job的具体task
        2、采用gather方式：从h_o_nodes所有节点嵌入中选择具体可选节点
        3、可选节点h_o_nodes：batch*task，128
        4、候选节点candidate：batch，job
        5、期望gather之后：batch，job，128

        这个是从外边传递进来的候选task的编号              
        candidate = tensor
        ([[ 0,  4,  8, 12],  
        [ 0,  4,  8, 12],
        [ 0,  4,  8, 12],
        [ 0,  4,  8, 12]], device='cuda:0')
        """
        #  shape = （batch * job） 其中每一行代表剩下可选的task的id，0开始赋值（即我的可选task的id的列表）
        # 使用`unsqueeze(-1)`方法将`candidate`张量在最后一个维度上添加一个维度，将其形状变为`(batch_size, self.n_j, 1)`。
        # 使用`expand(-1, self.n_j, h_nodes.size(-1))`方法将`candidate`张量在第二个维度上进行扩展，使其形状与`h_nodes`的形状相同。这将生成一个形状为`(batch_size, self.n_j, h_nodes.size(-1))`的张量`dummy`。
        # `expand`方法可以用来在指定的维度上复制张量的元素，从而改变张量的形状。
        # `expand`方法并不实际复制张量的元素，而是通过改变张量的尺寸和视图来模拟扩展操作。这意味着扩展后的张量与原始张量共享相同的存储空间，因此修改其中一个张量的元素也会影响到另一个张量。
        # 怪不得叫dummy，共享内存的，改一个其他都改了
        #  batch，job，128
        dummy = candidate.unsqueeze(-1).expand(-1, self.n_job,
                                               h_o_nodes.size(-1))  # batch，job，128。最后一个维度扩维 = 4,4,1，然后复制元素变成=4,4,128
        # print("观察数据：dummy = ", dummy, dummy.shape)

        # h_o_nodes整形：batch，task，128
        # out维度 = 索引张量dummy的维度： batch, job, 128 （所以从第二个维度挑选）
        # out[i][j][k] = input[i][index[i][j][k]][k],dim=1 !!!!!!!!!!!!!! 快速理解
        candidate_feature = torch.gather(h_o_nodes.reshape(dummy.size(0), -1, dummy.size(-1)),
                                         1,
                                         dummy)  # 类似于从每个样本中，挑选对应candidate中可选task的编号的节点的嵌入给挑选出来，重新组成变量！！！
        # print("观察数据：candidate_feature = ", candidate_feature, candidate_feature.shape) #打印太大也会报错

        """
        全图所有节点嵌入的均值，也作为输入网络的参数
        1、原：batch，hidden
        """
        # h_pooled：4*128 batch*hidden_dim ，第二个维度扩展1维：4,1,128
        # 然后扩展数据变成candidate_feature ：4,4,128的维度
        # 那么就需要对第二个维度的数据进行复制，复制4份: 每个样本的全图数据 * 4 复制了4分一样的
        h_g_o_pooled_repeated = h_g_o_pooled.unsqueeze(-2).expand_as(candidate_feature)
        # print("观察数据：h_g_o_pooled_repeated = ", h_g_o_pooled_repeated, h_g_o_pooled_repeated.shape) #打印太大也会报错

        """
        machine网络的输出参数：m节点的全图节点嵌入均值：step，batch，hidden
        1、每次调用forward只传入一次step的h_g_m_pooled = batch，hidden
        2、candidate_feature = batch, job, 128 

        先注释，选m和选o一点也不对应，这怎么能拿过来训练呢！
        """
        if h_g_m_pooled == None:  # 如果此次是初始选择动作，还没有mch_pool的结果：使用之前建立的128均匀分布的元素
            h_g_m_pooled_repeated = self._input[None, None, :].expand_as(candidate_feature).to(device)  # 并通过 `[None,None, :]` 在前两个维度上添加了维度，然后扩展复制维度到4,4,128
        else:  # 如果有选择m，就会不断更新m节点嵌入的全局池化值
            """# mch_pool全图的特征： batch * hidden_dim （mlp特征提取之后）"""
            # 然后在倒数第二维度增加一个维度：batch * 1 * hidden_dim
            # expand_as复制扩充变成candidate_feature = batch * job * 128 （每个batch中可以选择的job的对应的节点嵌入，mlp输出是128维度）
            h_g_m_pooled_repeated = h_g_m_pooled.unsqueeze(-2).expand_as(candidate_feature).to(device)  # 插入一个倒数第二维度，然后expand扩展复制成4,4,128=batch * job * 128
        # print("观察数据：h_g_m_pooled_repeated = ", h_g_m_pooled_repeated, h_g_m_pooled_repeated.shape)  # 打印太大也会报错
        """
        输入到网络的参数：
        1、输入到actor网络的，除了全图所有节点的均值以外，还有对应当前可选的节点嵌入（我之前输入的都只是全图的状态，对应节点的状态也可选）
        2、统一shape = batch，job，hidden。（理解为，只用candidate的节点输入到网络，然后输出的就是每个候选节点的打分，转成概率就是可选择的动作）
        """
        # 输入到Actor网络的是：待选节点的节点嵌入（gather挑选出来的） + 全图样本数据复制4分 + 设备数据：4,4,128
        # 因为是选择job，所以当前可以选择的节点就job=4个，所以shape = batch * job * hidden_dim*3（3个特征）
        concateFea = torch.cat((candidate_feature,
                                h_g_o_pooled_repeated,
                                h_g_m_pooled_repeated),
                                dim=-1)  # 在最后一个维度进行合并：4,4,128*3 （batch * job * hidden*3）
        # concateFea = torch.cat((candidate_feature,
        #                         h_g_o_pooled_repeated),
        #                        dim=-1)  # 在最后一个维度进行合并：4,4,128*2 （batch * job * hidden*2）

        """
        o_policy就是MLPActor的网络
        输出logit值：理解为可选节点的打分

        输入：4,4,128*3 （batch * job * hidden_dim）
        输出：4,4,1    1是因为输入的数据已经是可以每一个可以选择的job的节点表征了，输出1是针对每一个可选job的打分！（batch，job，1）
        """
        candidate_scores = self.o_policy(concateFea)  # Job的全连接层：3层，128*3输入，128隐藏，1输出（只看第二维度，最后一个维度为1，便于去掉）
        # candidate_scores = self.attn(decoder_input, candidate_feature)
        # candidate_scores = candidate_scores * 10  # 之前都是/1000作为归一化，可能是为了增加得分的范围或强调得分的重要性
        # print("Job_Aactor:查看数据 candidate_scores = ", candidate_scores, candidate_scores.shape)

        """
        mask的问题：传入batch，job
        1、都是加到logits层，没有本质区别，他直接置为-inf
        2、理论上：我设定的每一个m都可以做，那么mch_mask感觉就没有用了！！！（对的，他的m的mask一直全是False）
        3、为什么他的mask先第一列的job选完之后，后边的每次只能选择指定好的job（指定那个由env里边在判断！）
        """
        # mask传入：4,4 = batch*job
        mask_operation_reshape = mask_operation.reshape(candidate_scores.size())  # 变成和打分结果一样：4,4,1（batch，job，1）
        # 将mask_reshape中对应位置为True的元素置为-inf
        candidate_scores[mask_operation_reshape] = float('-inf')
        # print("Job_Aactor:查看数据 mask_operation_reshape = ", mask_operation_reshape, mask_operation_reshape.shape)
        # print("Job_Aactor:查看数据 candidate_scores(logit层，还未softmax) = ", candidate_scores, candidate_scores.shape)

        """
        在这里对logit采用softmax激活函数：
        1、输出可选节点的动作的概率！
        2、candidate_scores = （batch，job，1），先.squeeze(-1)去除最后一个维度
        3、prob概率pi = batch，job
        4、候选节点candidate：batch，job
        """
        prob = F.softmax(candidate_scores.squeeze(-1), dim=-1)  # 维度是4,4 = batch*job  所以是在最后一个维度进行softmax操作
        # print("Job_Aactor:查看数据 prob = ", prob, prob.shape)
        if use_greedy:
            task_index, action_index, log_a = greedy_select_action(prob, candidate)  # 如果是贪婪，那就是每次就选择最大prob
            # log_a = 0
            # action_index = 0
        else:
            """
            # 张量taskindex = tasks的具体id（从0开始）：一维，4元素，（batch）
            # 张量action_index：一维，（batch），离散采样得到的动作index
            # 张量log_a：一维，（batch），采样动作的对应的离散概率
            """
            task_index, action_index, log_a = select_operation_action(prob, candidate)  # 否则就是dist离散分布的采样
            # print("Job_Actor: select_action1查看数据 action = ", task_id, task_id.shape)
            # print("Job_Actor: select_action1查看数据 index = ", action_index, action_index.shape)
            # print("Job_Actor: select_action1查看数据 log_a = ", log_a, log_a.shape)

        # TODO 1113-新增局部的job的critic，只关注mk和it
        job_v = self.job_critic(h_g_o_pooled) # h_g_o_pooled = bs * hidden
        # print("观察数据：job_Critic：job_v = ", job_v, job_v.shape) # batch*2

        return task_index, action_index, log_a, prob, h_g_o_pooled, job_v  # 全图的特征batch*hidden

class esa_Machine_Actor(nn.Module):
    def __init__(self, configs):
        super(esa_Machine_Actor, self).__init__()

        
        self.n_machine = configs['n_machine']
        
        mlp_actor_layer = configs['mlp_actor_layer']
        self.machine_hidden_dim = configs['machine_hidden_dim']

        self.bn = torch.nn.BatchNorm1d(self.machine_hidden_dim).to(device)  # 批归一化层

        self.m_fea_1_fcl = nn.Linear(6, self.machine_hidden_dim, bias=False).to(device)  # 全连接层来池化节点的特征，6in，128out，没有偏置项
        self.m_fea_2_fcl = nn.Linear(8, self.machine_hidden_dim, bias=False).to(device)  # 全连接层来池化节点的特征，6in，128out，没有偏置项
        self.activation = nn.ELU() #输入 x 中的负值将按照 ELU 的定义进行变换，而正值将保持不变 TODO 在m的不同维度转成hidden的FCL后边，新增一个激活函数，一层后跟一个！能这样子用？？？？
        # self.activation = nn.LeakyReLU() #

        self.heads = 1

        # 初始化GAT网络
        self.gat_layer = GATLayer(in_features=self.machine_hidden_dim,  # 输入特征向量的维度；如果所有节点的维度不同，先进行一个FCL进行下统一！
                            out_features=self.machine_hidden_dim,  # 输出的维度
                            n_heads= self.heads,  # 几个多头，可以为1
                            concat= False,  # 是否对多头就行concate TODO 我不知道啥意思，但是我看默认都是false??
                            dropout= 0,  # dropout的系数，防止过拟合！
                            leaky_relu_slope= 0.2  # `leaky_relu_slope`参数确定了负输入的斜率大小。解决训练过程中神经元对负输入变得不响应的问题，这被称为“dying ReLU”问题
                            ).to(device)

        # machine这里：没有图结构 + 节点之间不连接，所以直接用全连接层来池化/聚合节点的特征向量
        #TODO 这里的m节点的特征聚合，直接一个线性层就ok了？GIN的MLP可是循环3次的！！！！！！！ + 为什么没有bias？？？
        # 我怀疑这个网络都没有起到很大的作用！！！

        # self.fcl_pooling = nn.Linear(3, self.machine_hidden_dim, bias=False).to(device)  # 全连接层来池化节点的特征，6in，128out，没有偏置项
        self.fcl_pooling = nn.Linear(self.machine_hidden_dim, self.machine_hidden_dim, bias=False).to(device)  # GAT聚合之后，全连接层来池化节点的特征，128in，128out，没有偏置项
        self.m_policy = MLPActor(mlp_actor_layer, self.machine_hidden_dim * 4, self.machine_hidden_dim, 1).to(device)  # 选择m的actor网络：共3层（包括输入）+128*2in+128hidden+1out

        """
        critic的MLP网络
        """
        mlp_critic_layer = configs['mlp_critic_layer']
        critic_input_dim = configs['critic_input_dim']
        critic_hidden_dim = configs['critic_hidden_dim']
        # self.critic = MLPCritic(mlp_critic_layer, critic_input_dim * 2, critic_hidden_dim, 1).to(device)  # 2层 + 128in + 32隐藏 + 1out
        self.machine_critic = MLPCritic(mlp_critic_layer, critic_input_dim, critic_hidden_dim, 2).to(device)  # 3层 + 128in + 32隐藏 + 1out  TODO 全图特征的输入= 1*128 + pt和tt的反馈reward = 2！

        # 正交初始化！
        if configs['use_orthogonal']:  # 就是configs的init
            for name, p in self.named_parameters():  # 遍历模型的所有参数，通过`self.named_parameters()`方法获取参数的名称和值。
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p,
                                            gain=1)  # 对于参数名称中包含’weight’的参数，判断其维度是否大于等于2。如果是，则使用`nn.init.orthogonal_`方法对该参数进行正交初始化，其中`gain`参数指定了初始化的增益。
                        print("Machine Actor init.orthogonal_ gain=1")
                elif 'bias' in name:
                    nn.init.constant_(p, 0)  # 对于参数名称中包含’bias’的参数，使用`nn.init.constant_`方法将其初始化为常数0。
                    print("Machine Actor init.constant_(p, 0)")

    def forward(self, machine_fea_1, machine_fea_2, h_pooled_o, machine_mask):  # m_fea2 指向 m_fea1： 有向边
        """

        :param machine_fea_1:  (当前task的对应的m的特征)传入machine网络的m节点的特征向量 = env_batch * m * 6 （np.array）
        :param machine_fea_2:  （选好m输入到ENV中的被选更新的state）传入machine网络的m节点的特征向量 = env_batch * m * 5 （np.array）
        :param h_pooled_o: 传入的上一个step的task节点的全图特征 = env_batch * hidden
        :param machine_mask: 当前job选择的task的mask = env_batch * 1 * m  好像直接就是张量？？？
        :return:
        """

        if torch.is_tensor(machine_fea_1):  # 判断是否为张量，因为在update里边，buffer里边读取的都已经是tensor了
            machine_fea_1 = copy.copy(machine_fea_1).float().to(device)  # 原deepcopy：转为张量+gpu  (env_batch * m * 6)
            machine_fea_2 = copy.copy(machine_fea_2).float().to(device)  # 原deepcopy：转为张量+gpu  (env_batch * m * 6)
        else:
            machine_fea_1 = torch.from_numpy(np.copy(machine_fea_1)).float().to(device)  # 转为张量+gpu  (env_batch * m * 6)
            machine_fea_2 = torch.from_numpy(np.copy(machine_fea_2)).float().to(device)  # 转为张量+gpu  (env_batch * m * 6)
        # print("Machine_Actor: forward： machine_fea = ", machine_fea, machine_fea.shape)
        # print("MActor: machine_fea_1 = ", machine_fea_1, machine_fea_1.shape)

        """将不同维度的m_fea转换成一致，然后按照（bs*m，2, hidden）进行转换： 代表两个m节点的聚合，2-->1"""
        # m_fea1 = self.activation(self.m_fea_1_fcl(machine_fea_1)) # bs * m * 128(hidden) TODO 加上了激活函数ELU???????
        # m_fea2 = self.activation(self.m_fea_2_fcl(machine_fea_2))
        """m_fea1 = self.m_fea_1_fcl(machine_fea_1)  # bs * m * 128(hidden)
        m_fea2 = self.m_fea_2_fcl(machine_fea_2)"""
        # print("MActor: m_fea1 = ",m_fea1, m_fea1.shape)
        # print("MActor: m_fea2 = ",m_fea2, m_fea2.shape)

        # TODO 1223-因为m现在有两个变量，所以分别无偏fcl，然后bn，然后再cancat！

        # fea12直接输入到1线性层中，结果batch*m*128，转成2维：（batch*m）*128，输入到bn层中，结果再转回来：batch * m * 128
        # 类似于对m的节点进行了encoder，变成了池化之后节点嵌入 = (batch_size, n_m, hidden_size)
        # TODO 1223-ESA-直接两个m的fea都是先fcl然后bn，直接concat完事！
        h_m_nodes1 = self.bn(self.m_fea_1_fcl(machine_fea_1).reshape(-1, self.machine_hidden_dim)).reshape(-1,
                                                                                                        self.n_machine,
                                                                                                        self.machine_hidden_dim)
        h_m_nodes2 = self.bn(self.m_fea_2_fcl(machine_fea_2).reshape(-1, self.machine_hidden_dim)).reshape(-1,
                                                                                                           self.n_machine,
                                                                                                           self.machine_hidden_dim)




        # # 将 A 和 B 展平为形状为 (bs * m) 的一维张量
        # m_fea1 = m_fea1.view(m_fea1.shape[0] * m_fea1.shape[1], m_fea1.shape[2])  # 三维变2维度：bs*m，hidden
        # m_fea2 = m_fea2.view(m_fea2.shape[0] * m_fea2.shape[1], m_fea2.shape[2])
        # # print("MActor: reshape: m_fea1 = ", m_fea1, m_fea1.shape)
        # # print("MActor: reshape: m_fea2 = ", m_fea2, m_fea2.shape)
        #
        # # 交替堆叠 m_fea1 和 m_fea2的第二个维度的元素值，会变成 = bs*m，2，hidden，不放心再reshape一下！
        # All_sameM_fea12 = torch.stack([m_fea1, m_fea2], dim=1).reshape(-1, 2, m_fea1.shape[-1])  # TODO 只有同个m的两个不同特征值进行聚合！(类比task_fea = bs*task，9的所有task和所有bs的特征值)
        # # print("MActor: reshape: All_sameM_fea12 = ", All_sameM_fea12, All_sameM_fea12.shape)
        #
        # """
        # h = bs*m，2，hidden
        # adj_m=(bs*m, head, 2, 2) 且ij=1表示j->i! + 默认有向边2->1，所以某一行全0，固定的
        # """
        # # 创建一个形状为 (bs * m) * head * 2 * 2 的张量
        # # TODO 修改adj = [[1, 1], [0, 1]]让m_fea2自己聚合自己，虽然我不用这一行！（全0行变成全-inf行，然后softmax该行=Nan！）
        # adj_m = torch.zeros(All_sameM_fea12.shape[0], self.heads, 2, 2).to(device)
        # # 将 2*2 的矩阵的第一行元素设置为 1
        # adj_m[:, :, 0, :] = 1  # 第一行都是1
        # adj_m[:, :, 1, 1] = 1  # 第二行第二列=1
        #
        # """h_prime_gat = （bs*m, 2, hidden）--- 其中2*hidden的第一行是我们想要的，第二行是m_fea2的自身聚合，我们只用第一行的参数 [:,0,:]= （bs*m, hidden）--reshape = （bs, m ,hiddden）"""
        # h_prime_gat = self.activation(self.gat_layer(h=All_sameM_fea12, adj_mat=adj_m))  # TODO 严格按照公式，和Wh加权聚合之后，需要一个ELU激活！！输出m节点嵌入特征值
        #
        # # TODO GAT可以循环n次的！对应GCN，那就3次！
        # h_prime_gat = self.activation(self.gat_layer(h=h_prime_gat, adj_mat=adj_m))
        # h_prime_gat = self.gat_layer(h=h_prime_gat, adj_mat=adj_m)
        #
        # # TODO 为什么要单独只用一个？两个都用，或者全图的特征不是可以吗？（FCL或MLPActor的需求：bs*m*hidden）
        # # h_machine_feas = h_prime_gat[:,0,:].reshape(-1, self.n_machine, self.machine_hidden_dim) # （bs, m ,hiddden）
        # # TODO 参考GCN中的Average方式，求均值mean（dim=-2）！也可以reshape（bs*m，2*hidden）--out的时候只有一个hidden也行！
        # h_machine_feas = h_prime_gat.mean(dim=-2).reshape(-1, self.n_machine, self.machine_hidden_dim) # （bs, m ,hiddden）  聚合方式！
        #
        # # print("MActor------单独取第一行的--------h_machine_feas =\n",h_machine_feas,h_machine_feas.shape)
        #
        # # 直接数据放缩！考虑要不要加？因为p*t会很大！
        # # machine_fea = machine_fea / configs['']et_normalize_coef  # 归一化处理: 直接原数据/1000 batch * m  当前所选择m的上一个工序的完工时间
        #
        # # fea直接输入到1线性层中，结果batch*m*128，转成2维：（batch*m）*128，输入到bn层中，结果再转回来：batch * m * 128
        # # 类似于对m的节点进行了encoder，变成了池化之后节点嵌入 = (batch_size, n_m, hidden_size)
        # # TODO 此时就用一用bn层吧，fcl没必要了！
        # # h_m_nodes = self.bn(self.fcl_pooling(h_machine_feas).reshape(-1, self.machine_hidden_dim)).reshape(-1,
        # #                                                                                                 self.n_machine,
        # #                                                                                                 self.machine_hidden_dim)
        # h_m_nodes = self.bn(h_machine_feas.reshape(-1, self.machine_hidden_dim)).reshape(-1,
        #                                                                                self.n_machine,
        #                                                                                self.machine_hidden_dim)

        # 计算 `h_m_nodes`= (batch_size, n_m, hidden_size) 张量在第二个维度上的均值，即对每个样本的 `n_m` 个节点特征进行平均。得到的 `pool` 张量形状为 `(batch_size, hidden_size)`。
        # 相当于我首先节点池化了一波，每个m都有128个特征，然后我对每个batch中的all m求他的平均的特征表征（类似于全图中各节点的嵌入的均值，池化过程！）
        h_pooled1 = h_m_nodes1.mean(dim=1)  # (batch_size, hidden_size)
        h_pooled2 = h_m_nodes2.mean(dim=1)  # (batch_size, hidden_size)
        h_pooled = (h_pooled1 + h_pooled2)/2  # TODO 1223-；两个m的特征，求一个均值作为全局的特征，用于传递到job网络中

        """"
        下述的变换操作：
        1、是为了和各个m节点的嵌入进行合并：batch*m*128
        2、从而进行expand的复制操作
        
        h_m_nodes1维度 = h_m_nodes2维度
        """
        # 这一行代码使用 `unsqueeze(1)` 在第二个维度上添加一个维度，并使用 `expand_as` 函数将 `pool` 张量复制扩展成和 `action_node` 张量相同的形状。
        # 得到的 `h_pooled_repeated` 张量形状为 `(batch_size, n_m, hidden_size)`。
        # 相当于：每个batch中的全m的节点嵌入复制了m=4份
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(h_m_nodes1)  # (batch_size, n_m, hidden_size)  m节点的全图节点嵌入的均值

        # h_poooled_m: batch * hidden_dim GIN之后的上一个状态的全图的节点嵌入的均值
        # 同上：第2维度加1，然后expand复制变成(batch_size, n_m, hidden_size) 4*4*128
        pooled_repeated = h_pooled_o.unsqueeze(1).expand_as(h_m_nodes1) # (batch_size, n_m, hidden_size)  o节点的全图节点嵌入的均值

        # 维度不够就要进行expand的复制补充！
        # 输入变量： m节点的池化嵌入 + 所有m节点全局嵌入（mean） + 选择job的全图的all节点的嵌入（mean，暂无，先去掉！） ： batch * m * 128*4
        concateFea = torch.cat((h_m_nodes1, h_m_nodes2, h_pooled_repeated, pooled_repeated), dim=-1)  # 最后一个维度合并
        # print("MCH_Actor查看数据： concateFea = ", concateFea, concateFea.shape)  # batch * m * 128*2

        mch_scores = self.m_policy(concateFea)  # batch * m * 1 输出的是选择哪一个m的得分：放入到了MLPActor的网络中
        mch_scores = mch_scores.squeeze(-1) * 10  # 去掉最后一个维度：batch * m。数值放大10倍
        # mask_reshape = mask_mch_action.reshape(candidate_scores.size())
        # print("MCH_Actor查看数据： 还没没有mask：mch_scores = ", mch_scores, mch_scores.shape)

        """选择m的mask：暂无，省略此步"""
        # TODO: add machine选择时候的mask
        # machine_mask: batch * 1 * m 所选择的task的对应的mask（false可选）
        # squeeze（1）去掉维度为1的：batch * m，bool（）转换成布尔张量（若已经是bool，保持不变）：这将把非零值转换为 `True`，零值转换为 `False`。
        # 使用 `masked_fill` 函数，根据掩码张量的值将 `mch_scores` 张量中对应位置的元素填充为负无穷大。掩码张量中为 `True` 的位置将被填充，而为 `False` 的位置将保持不变。
        # print("MCH_Actor查看数据： mask之后： machine_mask = ", machine_mask, machine_mask.shape)
        mch_scores = mch_scores.masked_fill(machine_mask.squeeze(1).bool(), float("-inf"))
        # print("MCH_Actor查看数据： mask之后： mch_scores = ", mch_scores, mch_scores.shape)


        # 掩码加持之后的mch_scores，进行softmax输出选择m的概率：batch * m 数值变成概率了
        mch_prob = F.softmax(mch_scores, dim=-1)
        # print("MCH_Actor查看数据： mch_prob = ", mch_prob, mch_prob.shape)

        # TODO 1113-新增局部的machine的critic，只关注pt和tt
        machine_v = self.machine_critic(h_pooled)  # h_pooled = bs * hidden
        # print("观察数据：machine_Critic：machine_v = ", machine_v, machine_v.shape) # batch*2

        return mch_prob, h_pooled, machine_v  # 返回的是batch*m的选择m的概率 + 返回所有m节点的全局嵌入（求均值）：batch * hidden_dim









