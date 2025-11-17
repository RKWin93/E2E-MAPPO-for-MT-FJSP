import torch.nn as nn
import torch
import torch.nn.functional as F
from trainer.train_device import device
from algorithm.ppo_trick import orthogonal_init
from instance.generate_allsize_mofjsp_dataset import Logger


class Encoder(nn.Module):  # GNN的网络，图神经网络的编码
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, learn_eps, neighbor_pooling_type, device):
        super(Encoder, self).__init__()
        self.feature_extract = GraphCNN(num_layers=num_layers,    # # GraphCNN图卷积的实例化，
                                        num_mlp_layers=num_mlp_layers,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)

    def forward(self, x, graph_pool, padded_nei, adj, ): # GraphCNN的forward函数
        h_pooled, h_nodes = self.feature_extract(x=x,    # 节点的特征向量
                                                 graph_pool=graph_pool,  # 求全图池化所需的均值矩阵
                                                 padded_nei=padded_nei,  # 求max池化所需，暂不用！！
                                                 adj=adj)

        return h_pooled, h_nodes  # 全图池化嵌入 + all节点的嵌入  # batch* hidden 和（batch*tasks）* hidden


class GraphCNN(nn.Module):   
    def __init__(self,
                 num_layers,   # 网络总层数：定义的时候把输入层也算到里边了，所以真正用来图卷积的层数需要-1，这是形参定义的原因，没有其他原因
                 num_mlp_layers,  # mlp的层数
                 input_dim,   # 输入特征的维度
                 hidden_dim,  # 隐藏层维度
                 # final_dropout,  # 最后线性层要不要dropout
                 learn_eps,  # 学习参数（自身节点的额外特征  or  直接相加的聚合）
                 neighbor_pooling_type,  # 如何聚合邻居节点，也叫pooling池化（平均，最大和最小）就是叠加
                 device):  # 部署设备
        '''
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        是否学习参数epsilon，用于区分中心节点和邻居节点。如果为True，则学习epsilon；如果为False，则将邻居节点和中心节点一起聚合。
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        device: which device to use
        '''

        super(GraphCNN, self).__init__()

        # self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        # common out the eps if you do not need to use it, otherwise the it will cause error "not in the computational graph"
        # if self.learn_eps:
              # 创建一个形状为 `(self.num_layers - 1,)` 的全零张量，将全零张量转换为可训练的参数对象
        #     self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))  # 通过将一个 `nn.Parameter` 对象赋值给模型的属性，可以使该属性成为模型的可训练参数。

        # List of MLPs  都是module的list形式
        self.mlps = torch.nn.ModuleList()
        self.bn = torch.nn.BatchNorm1d(input_dim)  # ppo调参说不用用这个批归一化！！！！！！！！！！！！！！ 自动对一批数据的不同特征进行归一化处理！
        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        # print("sssssssssssssssssssss", self.num_layers)
        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))   # 输入层，mlp有几层，输入x的dim，隐藏层，输出层
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    # 区分中心节点和邻居节点，然后用学习参数self.eps来作为学习参数
    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.
        # 该方法用于在每一层中对节点特征进行聚合，并进行epsilon重新加权。
        # 当前层的节点特征 + 第几层 + 邻居节点的填充列表（仅在邻居聚合类型为"max"时使用） + 邻接矩阵块（仅在邻居聚合类型为"mean"或"sum"时使用）

        if self.neighbor_pooling_type == "max":
            # If max pooling
            # 根据理解，就是最大池化操作：节点向量选择max的进行组合
            pooled = self.maxpool(h, padded_neighbor_list)  # 节点的特征张量 + 邻居节点的填充列表（什么形式呢？？？？？？？？？？？？？？？？？）  没写就没用
        else:
            # If sum or average pooling  求和
            pooled = torch.mm(Adj_block, h)  # 矩阵乘法： Adj_block * h： 相当于对每个中心节点的邻居节点特征进行求和（结果不包括中心节点自身）
            if self.neighbor_pooling_type == "average":
                # If average pooling  求平均
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))  # 计算每个中心节点的邻居节点的度（即邻居节点的数量）
                pooled = pooled/degree  # （平均特性，能体现出连接不同节点的差异）pooled中的第一个元素（即第一行）中各个元素 / degree的第一个元素（标量），以此类推，shape= pooled

        # Reweights the center node representation when aggregating it with its neighbors
        # 偏置项为什么会有layer的属性？每一层都不一样吗？是个要学习的参数吗（用nn.Parameter转成可学习的参数张量，layer-1个变量，这里第几层用哪一个）
        # 初始为0，可以训练
        pooled = pooled + (1 + self.eps[layer])*h  # 通过加上一个偏置项 `(1 + self.eps[layer])*h` 来重新加权中心节点的表示。这个偏置项可以帮助保留中心节点的原始特征。
        pooled_rep = self.mlps[layer](pooled) # 通过该层对应的MLP进行变换，transform, 得到节点的新表征！！！！！！！！repr
        h = self.batch_norms[layer](pooled_rep)  # 对新的特征表示 `pooled_rep` 进行批归一化操作 `self.batch_norms[layer]`

        # non-linearity
        h = F.relu(h)  # 先归一化，然后激活函数？？？？？？？？？？？？？
        return h  # 返回节点特征h的矩阵

    """不区分中心节点 + 其对应的邻居节点"""
    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        """pool = aggregation, 池化就是聚合操作，均值 or max的方式选择节点进行聚合操作"""
        # pooling neighboring nodes and center nodes altogether
        # （adj如果是同个node为0，那么pooled的没有本身的中心节点啊？他是怎么一起aggregate的？？？）他计算adj的时候人为的将自身节点的对角线全置一！！！
        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)  # 没给函数就是没用到
        else:
            # If sum or average pooling
            # print(Adj_block.dtype)
            # print(h.dtype)
            """
            Adj_block = bs*task, bs*task
            h = bs*task,特征向量元素个数12
            理解成： task*task 点乘 task*12，各个节点都进行了聚合操作 = （bs*task，12）
            """
            pooled = torch.mm(Adj_block.double(), h.double())  # ！ 矩阵乘法 = 节点聚合or池化，TODO（按照边权重进行加权和，作为新的节点特征向量）聚合邻居节点的特征进行池化（边有权重） # `torch.mm()`函数期望的是`Double`类型的张量，
            if self.neighbor_pooling_type == "average":
                # If average pooling  # `torch.mm()`函数期望的是`Double`类型的张量
                '''
                BUG：如果我有边的权重>1，此时计算的degree！=边的个数吧？？？
                so: 创建一个一模一样的Adj_block，只是其中的value变成了1，用来计算度
                '''
                # 假设a是原始的稀疏矩阵
                a_indices = Adj_block._indices()
                a_values = Adj_block._values()
                a_shape = Adj_block.shape

                # 创建一个与a具有相同形状的稀疏矩阵b，并将其元素初始化为1
                b_indices = a_indices.clone()
                b_values = torch.ones_like(a_values)
                new_Adj_batch = torch.sparse.FloatTensor(b_indices, b_values, a_shape)  # 仅仅是用来计算degree = 边个数 = 邻居节点个数（包括了自身！）
                # print("new_Adj_batch = ", new_Adj_batch)

                '''原计算度 = 表示该节点的边的个数，需要边权重都=1, 不然下述计算方式不准确）'''
                # BUG：250423-如果我传入的adj是包含自身节点的，那么计算度的时候不就是每个节点都多算了一个边，即自身？（除非我自己的adj没有自身置1？？）
                # degree = torch.mm(Adj_block.double(), torch.ones((Adj_block.shape[0], 1), dtype=torch.double).to(self.device))  # 同上：计算各个中心节点的邻居节点的个数
                degree = torch.mm(new_Adj_batch.double(), torch.ones((new_Adj_batch.shape[0], 1), dtype=torch.double).to(self.device))  # 同上：计算各个中心节点的邻居节点的个数（ =  （bs*task，1）
                # print("+++++++++++++++++++++\n",Adj_block,"\n",degree)
                # print("pooled = ", pooled)
                pooled = pooled/degree  # 平均池化: 对应元素相除，task * 1会被自动广播扩充（第二个维度1变成12）然后逐元素相除
                # print("average pooled = ", pooled)
        # representation of neighboring and center nodes

        pooled_rep = self.mlps[layer](pooled) # 池化特征进行mlp的变换，（但是此时没有各个中心节点自身的特征啊，只有邻居节点的池化！！！？？？）
        h = self.batch_norms[layer](pooled_rep) # 批归一化

        # non-linearity
        h = F.relu(h) # 激活函数
        return h

    def forward(self,
                x,  # 输入特征
                graph_pool,  # 全图的池化矩阵所需的均值表示（稀疏矩阵，元素是1/nodes）
                padded_nei,  # 邻居节点的填充列表（max采用）
                adj):  # 邻接矩阵（稀疏矩阵表示Adj_block = bs*task, bs*task)
        x_concat = x  # concat意思为合并，就是堆叠数组那系列操作，所以输入的都是合并之后的ALL节点特征矩阵？？？？？？？？？
        graph_pool = graph_pool

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = padded_nei
        else:
            Adj_block = adj

        # list of hidden representation at each layer (including input)
        h = x_concat

        # 为什么卷积的层数是self.num_layers-1？？？为何这样子定义
        # 每一层进行循环操作：计算当前层的邻居节点的池化/聚合/表征，每一层中的mlp都是多层，看形参的设定；第一层需要对准输入x的dim，后续都是hidden的dim
        for layer in range(self.num_layers-1):  # max的函数定义你都没写，所以默认没有；self.learn_eps:我看=false
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)  # TODO 实际训练用的这个！！！！！！！！！！！！！！！！！！！

        h_nodes = h.clone()  # 最后输出的h节点特征向量矩阵，包含所有batch的所有节点：TODO （batch*tasks）* hidden
        # Logger.log("Training/ppo/job_actor/Encoder/GraphCNN/forward_h_nodes", f"h_nodes={h_nodes.shape}, graph_pool={graph_pool.shape}, Adj_block={Adj_block.shape}", print_true=1)
        
        # 使用稀疏矩阵乘法操作将图的池化矩阵应用到节点特征上，并返回池化后的特征和每一层的节点特征。
        pooled_h = torch.sparse.mm(graph_pool, h) # 直接求all节点的平均值作为图的池化表征：graph_pool = 1/nodes的batch*（batch*tasks）和（batch*tasks）* hidden 矩阵相乘
        # Logger.log("Training/ppo/job_actor/Encoder/GraphCNN/forward_pooled_h", f"pooled_h={pooled_h.shape}", print_true=1)

        # 由于稀疏矩阵乘法的结果是一个稀疏矩阵，所以输出的结果也是一个稀疏矩阵。输出的结果会根据稀疏矩阵的表示方式进行打印，显示非零元素的索引和值。
        # ，按照sise大小建立一个稀疏矩阵，给出索引的位置（按照size判断），按照顺序填入给出的非零元素的值，其余位置为0
        return pooled_h, h_nodes  # batch* hidden 和（batch*tasks）* hidden


"""
MLP with lienar output
用在GIN里边的MLP！！！！！！！！！！！！！！！！！！！！！！！！！！
"""
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))  # 输入层
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))  # 隐藏层
            self.linears.append(nn.Linear(hidden_dim, output_dim))   # 输出层

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))   # 层数-1个批归一化层

    def forward(self, x):  # mlp都是单x输入
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x.to(torch.float32) # `torch.mm()`函数期望的是`Double`类型的张量, 所以有些变量变成了double=float64,而线性层是float32，需要转回来
            for layer in range(self.num_layers - 1):
                # print(h.dtype) # torch.float64
                # print(self.linears[layer].weight.dtype) # torch.float32
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))  # 输入+隐藏层：全连接 + 批归一化 + 激活函数
            return self.linears[self.num_layers - 1](h)  # ! TODO 最后的输出层，没有用激活函数

def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    # batch_size is the shape of batch
    # for graph pool sparse matrix   整图的池化稀疏矩阵！
    if graph_pool_type == 'average':
        # 使用`torch.full`创建`elem`张量，大小为`(batch_size[0]*n_nodes, 1)`，并填充为值`1 / n_nodes`，表示求平均。然后使用`.view(-1)`将张量重新形状为1维张量。
        elem = torch.full(size=(batch_size * n_nodes, 1), 
                          fill_value=1 / n_nodes,
                          dtype=torch.float32,
                          device=device).view(-1)  # bs*task, view=reshape,转成1维度，按照元素个数
    else:
        elem = torch.full(size=(batch_size * n_nodes, 1),
                          fill_value=1,   # 此时填充为0
                          dtype=torch.float32,
                          device=device).view(-1)
    idx_0 = torch.arange(start=0, end=batch_size, # 使用`torch.arange`创建`idx_0`张量，取值范围从0到`batch_size[0]`，表示结果图池化矩阵的第一维的索引
                         device=device,
                         dtype=torch.long)  #  一维度，从0---bs-1
    # print(idx_0)
    """repeat指定每个维度重复次数，stack默认dim=0在维度堆叠，view=reshape,转成1维度，按照元素个数，squeeze去除张量中所有单维度（即维度大小为1的维度）"""
    
    # 通过将`idx_0`张量重复`n_nodes`次并使用`.t()`进行转置。然后使用`.reshape()`将其重新形状为`(batch_size[0]*n_nodes, 1)`，并使用`squeeze()`去除冗余的维度。
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size*n_nodes, 1)).squeeze()  # task*bs -- bs*task -- (bs*task)*1 -- 1维度，bs*task个元素

    # 使用`torch.arange`创建`idx_1`张量，取值范围从0到`n_nodes*batch_size[0]`，表示结果图池化矩阵的第二维的索引。
    idx_1 = torch.arange(start=0, end=n_nodes*batch_size,  
                         device=device,
                         dtype=torch.long)  # 1维度，0--task*bs-1
    idx = torch.stack((idx_0, idx_1))  # 通过将`idx_0`和`idx_1`张量堆叠在一起，创建`idx`张量(默认dim=0) =  2*（bs*task）

    """
    创建一个稀疏矩阵：不为0的索引 + 对应元素 + size
    1、# size是一个二维张量：batch行，batch*node列
    2、# 元素都是1/node，bs行，每一行只有task个连续元素是1/task，其他（bs-1）*task个元素都是0.稀疏矩阵，为了求所有task节点的均值
    3、# 使用`torch.sparse.FloatTensor`创建`graph_pool`张量，其中`idx`为索引，`elem`为值。张量的大小为`[batch_size[0], n_nodes*batch_size[0]]`，表示结果图池化矩阵的形状。张量还被移动到指定的`device`上。
    
    假设 batch_size = 2，n_nodes = 3，graph_pool_type = 'average'：
    elem：[1/3, 1/3, 1/3, 1/3, 1/3, 1/3]
    idx_0：[0, 0, 0, 1, 1, 1]
    idx_1：[0, 1, 2, 3, 4, 5]
    graph_pool 稀疏矩阵：
    [1/3, 1/3, 1/3,  0,  0,  0]
    [ 0,  0,  0, 1/3, 1/3, 1/3]

    # """
    graph_pool = torch.sparse.FloatTensor(idx, 
                                          elem,  
                                          torch.Size([batch_size,n_nodes*batch_size])
                                          ).to(device)
    '''graph_pools = []
    for i in range(configs.batch_size):
        graph_pools.append(graph_pool)'''

    return graph_pool

def aggr_obs(obs_mb, n_node):
    # obs_mb is [m, n_nodes_each_state, fea_dim], m is number of nodes in batch
    # batch * tasks * tasks  传入之前需要先转成稀疏张量！！！！！！！！！！！！！！
    idxs = obs_mb.coalesce().indices()  # 调用`obs_mb.coalesce()`方法来获取稀疏张量的非零元素的索引和值：索引也是三维的！！！！
    vals = obs_mb.coalesce().values()
    new_idx_row = idxs[1] + idxs[0] * n_node  # 通过将原始索引乘以`n_node`并加上节点在批处理中的偏移量，可以将节点索引映射到批处理中的全局索引。这样做是为了将每个状态中的节点与其他状态中的节点连接起来，形成全局的邻接矩阵。
    new_idx_col = idxs[2] + idxs[0] * n_node  # 就是把原先在自己矩阵里边的行列，都按照第几个矩阵进行扩大了几*16倍（tasks=16）
    idx_mb = torch.stack((new_idx_row, new_idx_col)) # 代码使用`torch.stack()`函数将新的行索引和列索引堆叠成一个张量`idx_mb`，默认dim=0
    # print(idx_mb)
    # print(obs_mb.shape[0])
    adj_batch = torch.sparse.FloatTensor(indices=idx_mb,
                                         values=vals,
                                         size=torch.Size([obs_mb.shape[0] * n_node,
                                                          obs_mb.shape[0] * n_node]),
                                         ).to(obs_mb.device)
    return adj_batch # 总体而言，该函数的作用是将批处理中的观测聚合成稀疏邻接矩阵，用于表示节点之间的连接关系 = （bs*task，bs*task）

class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPActor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers   # 神经网络的层数（包含in+hidden+out）

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)  # 单层网络
        else:
            # Multi-layer model
            self.linear_or_not = False  # 多层网络，标志维false，为了forward计算
            self.linears = torch.nn.ModuleList()  # 可储存多个线性层
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))  # append添加，按照索引可以被访问
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))  # 按照层数添加隐藏层（去掉输入和输出）
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''
        # if configs.use_orthogonal:
        #     print("------Actor MLP：use_orthogonal_init------")  # 网络的正交初始化, 默认gain=1.0；actor的输出层gain=0.01
        #     if num_layers == 1:
        #         orthogonal_init(self.linear)
        #     elif num_layers > 1:
        #         orthogonal_init(self.linears[:-1])  # 除了最后输出层层的gain=1.0！
        #         orthogonal_init(self.linears[-1], gain=0.01)  # 输出层的gain=0.01！

    def forward(self, x):  # 传入的是展平之后的state
        if self.linear_or_not:
            # If linear model
            return self.linear(x)  # 单层网络就直接输出，无激活函数
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))  # 多层网络需要加上激活函数  TODO MLPActor和MLPCritic用的都是tanh，结果都在-1和1之间！！！注意一下网络输出的大小和数量级是否对应，计算aadv和loss的时候！！！！！！
                # h = F.relu(self.linears[layer](h))  # 注意此地用的是F.relu，一样的操作
            # return F.softmax(self.linears[self.num_layers - 1](h), dim=-1)  # 统一在actor网络输出softmax之后的概率
            return self.linears[self.num_layers - 1](h)  # ! TODO 最后一层网络：输出logits，图网络里理解为score！

# 用于global_critic的MLP建立
class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)

