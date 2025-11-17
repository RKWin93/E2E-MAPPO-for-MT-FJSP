import torch.nn as nn
import torch
import torch.nn.functional as F


import torch
from torch import nn
import torch.nn.functional as F

################################
### GAT LAYER DEFINITION ###
################################
"""
1、可以考虑采用多头？现在先为1
2、保证：输入的事所有要聚合的节点特征向量！（不同维度的在输入之前已经unify了！） + 然后进行聚合操作（和adj进行mask，然后输出一个包含所有节点的权重注意力系数，然后加权和即可！）

理解：GAT用来聚合！再用GIN来多MLP-输出一个类似adj的系数矩阵+adj用来屏蔽！（包含自身系数和邻居系数）
本质上：W就是一个无偏的线性层！A也是无偏线性层的转换！只不过这里是手动写出来了！

in_features = out_features = hidden传入的fcl之后的元素个数

concat = False！！！没有对应修改过！合并head！！我这里用的是均值的heads！！
"""
class GATLayer(nn.Module):

    def __init__(self,
                 in_features: int,  # 输入特征向量的维度；如果所有节点的维度不同，先进行一个FCL进行下统一！
                 out_features: int,  # 输出的维度
                 n_heads: int = 1,  # 几个多头，可以为1
                 concat: bool = False,  # 是否对多头就行concate TODO 多头机制需要拆分的，这里不用多头，head=1
                 # dropout: float = 0.4,  # dropout的系数，防止过拟合！
                 dropout: float = 0,  # dropout的系数，防止过拟合！
                 leaky_relu_slope: float = 0.2 # `leaky_relu_slope`参数确定了负输入的斜率大小。解决训练过程中神经元对负输入变得不响应的问题，这被称为“dying ReLU”问题
                 ):
        super(GATLayer, self).__init__()

        self.n_heads = n_heads # Number of attention heads
        self.concat = concat # wether to concatenate the final attention heads
        self.dropout = dropout # Dropout rate

        if concat: # concatenating the attention heads
            self.out_features = out_features # Number of output features per node
            assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
            self.n_hidden = out_features // n_heads
        else: # averaging output over the attention heads (Used in the main paper)
            self.n_hidden = out_features

        # A shared linear transformation, parametrized by a weight matrix W is applied to every node
        # Initialize the weight matrix W
        # self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads))) # TODO 生成empty的可学习参数？还是参考TII的torch.rand = [0, 1)
        self.W = nn.Parameter(torch.rand(size=(in_features, self.n_hidden * n_heads), dtype=torch.float))

        # Initialize the attention weights a
        # self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))
        self.a = nn.Parameter(torch.rand(size=(n_heads, 2 * self.n_hidden, 1), dtype=torch.float))

        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope) # LeakyReLU activation function
        self.softmax = nn.Softmax(dim=1) # softmax activation function to the attention coefficients

        self.reset_parameters() # Reset the parameters


    def reset_parameters(self):

        nn.init.xavier_normal_(self.W)  # 将权重矩阵W的元素从 Xavier 正态分布中采样得到，其中均值为0，标准差根据权重矩阵的大小进行调整，以保持方差相等。
        nn.init.xavier_normal_(self.a)

    def _get_attention_scores(self, h_transformed: torch.Tensor):

        # h_transformed = (n_heads, n_nodes, n_hidden)
        # self.a[:, :self.n_hidden, :] = (n_heads, n_hidden, 1)
        #  = (n_heads, n_nodes, 1)
        source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
        target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])

        # broadcast add
        # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
        """(bs*m, head, 2, 1) --.mT = (bs*m, head, 1, 2)"""
        e = source_scores + target_scores.mT # TODO 我试了，tensor的4维度 * 3维度，=4维度，然后.mT真的只有后两个维度对换！
        return self.leakyrelu(e)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):

        """

        :param h:   传入 bs*m，2，hidden=in_feature
        :param adj_mat:
        :return:
        """
        # n_nodes = h.shape[0]
        n_nodes = h.shape[-2]  # 不管是2维还是3维都能找到传入的nodes的个数！ TODO node = 2 因为同个m只有两个不同的fea向量，视为不同的节点

        # Apply linear transformation to node feature -> W h
        # output shape (n_nodes, n_hidden * n_heads)
        """h= bs*m，2，hidden * w=(hidden, self.n_hidden * n_heads)----(bs*m,2,hidden*head)"""
        h_transformed = torch.matmul(h, self.W) # TODO 相乘的维度变化我test过！nn.matmul相比nn.mm执行矩阵惩罚，对不同维度的输入更灵活
        # `self.dropout`是dropout的概率，表示被丢弃的元素比例。`self.training`是一个布尔值，用于指示当前是否处于训练模式。
        # `self.training`变量通常用于指示当前模型是否处于训练模式。它是在`nn.Module`类中的`train()`和`eval()`方法中自动设置的。
        h_transformed = F.dropout(h_transformed, self.dropout, training=self.training) #  Dropout是一种常用的正则化技术，用于减少过拟合。它在训练过程中随机丢弃一些神经元的输出，以增强模型的鲁棒性和泛化能力。

        # splitting the heads by reshaping the tensor and putting heads dim first
        # output shape (n_heads, n_nodes, n_hidden)
        # 参数1（第一个参数）：表示原始张量中的维度索引1（从0开始计数），将会成为新张量中的第0维度。
        """(bs*m,2,hidden*head)-我多了一维度--(bs*m,2,head,hidden)---再转换（bs*m, head, 2, hidden）"""
        if h_transformed.dim() == 2:  # 针对传入的是 = nodes * in_feature（hidden）的变量
            h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)  # permute维度变换新维度参数（第一，第二，第三），从旧维度（1，0,2）变换来
        else:
            h_transformed = h_transformed.view(-1, n_nodes, self.n_heads, self.n_hidden).permute(0,2,1,3)  # permute维度变换新维度参数（第一，第二，第三），从旧维度（1，0,2）变换来

        # getting the attention scores
        # h_transformed =  (n_heads, n_nodes, n_hidden)
        # a = (n_heads, 2 * self.n_hidden, 1)  取第二个维度的一半
        # output shape (n_heads, n_nodes, n_nodes)
        """（bs*m, head, 2, hidden） * (n_heads, 2 * self.n_hidden / 2, 1) = (bs*m, head, 2, 1)---广播加法=自身+邻居= (bs*m, head, 2, 2) +  2*2可以视为adj矩阵！！"""
        e = self._get_attention_scores(h_transformed)

        # Set the attention score for non-existent edges to -9e15 (MASKING NON-EXISTENT EDGES)
        """e = (bs*m, head, 2, 2) + 定义adj=(bs*m, head, 2, 2)，且ij=1表示j->i! + 默认有向边2->1，自身的位置置1"""
        # connectivity_mask = -9e16 * torch.ones_like(e)
        connectivity_mask = float("-inf") * torch.ones_like(e)  # TODO 这里设置一个小值，可能会导致Nan；还是经典，要mask就用-inf！
        e = torch.where(adj_mat > 0, e, connectivity_mask) # masked attention scores adj = e = (n_heads, n_nodes, n_nodes) 维度应该相同！ 大于0的地方都赋值-inf
        # print("-----test: 注意力分数 = e = \n", e, e.shape)

        # attention coefficients are computed as a softmax over the rows
        # for each column j in the attention score matrix e
        """(bs*m, head, 2, 2)"""
        # TODO Nan原因：因为我的adj的一整行都是0，会mask成-inf；然后这一行全是-inf的softmax之后，就是Nan！！！！！！！！！！！（修改adj = [[1,1],[0,1]]让m_fea2自己聚合自己，虽然我不用这一行！）
        attention = F.softmax(e, dim=-1) # TODO 最后一个维度softmax，就是相当一行中的所有的元素softmax下！维度不变
        attention = F.dropout(attention, self.dropout, training=self.training)
        # print("-----test: softmax之后 = attention = \n", attention, attention.shape)

        # attention = attention * 1000

        # final node embeddings are computed as a weighted average of the features of its neighbors
        # out shape = (n_heads, n_nodes, n_hidden)
        """attention权重 = (bs*m, head, 2, 2) * h_transformed=（bs*m, head, 2, hidden）-----（bs*m, head, 2, hidden）"""
        # TODO 聚合节点特征用错了吗？？？应该和原始特征进行矩阵乘法（原始特征这里指的是传入的h！但是维度不同，少了head，那就先去head！）
        h_prime = torch.matmul(attention, h_transformed) # attention = (n_heads, n_nodes, n_nodes) * h_transformed =  (n_heads, n_nodes, n_hidden)

        """attention权重 = (bs*m, head, 2, 2) * h=（bs*m, 2, hidden）expand（bs*m, head, 2, hidden） -----（bs*m, head, 2, hidden）"""
        # h_repeated = h.unsqueeze(-3).expand(h.shape[0], self.n_heads, h.shape[-2], h.shape[-1])
        # h_prime = torch.matmul(attention, h_repeated) # attention = (n_heads, n_nodes, n_nodes) * h_transformed =  (n_heads, n_nodes, n_hidden)
        # print("-----test: GAT=adj * h之后: h_prime = \n",h_prime, h_prime.shape)

        # concatenating/averaging the attention heads
        # h_prime = shape = (n_heads, n_nodes, n_hidden)
        # output shape (n_nodes, out_features)
        """h_prime = （bs*m, head, 2, hidden）"""
        if self.concat:
            if h_prime.dim() == 3:
                h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features) # TODO 同样是消除heads的维度，并到最后一个维度：当时self.out_features// n_heads被拆分了，现在又乘进去
            else:
                h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(-1, n_nodes, self.out_features)  # 同理，保留bs*m，剩下和上行一致
        else:
            # h_prime = h_prime.mean(dim=0)  # TODO mean的作用是针对head的，消除掉head的多注意力，求均值 = (n_nodes, out_features)
            """h_prime = （bs*m, head, 2, hidden）--（bs*m, 2, hidden）--- 其中2*hidden的第一行是我们想要的，第二行是m_fea2的自身聚合，我们只用第一行的参数 [:,0,:]= （bs*m, hidden）--reshape = （bs, m ,hiddden）"""
            h_prime = h_prime.mean(dim=-3)  # TODO -3不管3维度or4维度张量，mean的时候消除的都是heads！！！

        return h_prime  # out = （bs*m, 2, hidden）

class GAT_multi_heads(nn.Module):

    def __init__(self,
                in_features,
                n_hidden,
                n_heads,
                num_classes,
                concat=False,
                dropout=0.4,
                leaky_relu_slope=0.2):

        super(GAT_multi_heads, self).__init__()

        # Define the Graph Attention layers
        self.gat1 = GATLayer(
        in_features=in_features, out_features=n_hidden, n_heads=n_heads,
        concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope
        )

        self.gat2 = GATLayer(
        in_features=n_hidden, out_features=num_classes, n_heads=1,
        concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope
        )

    def forward(self, input_tensor: torch.Tensor , adj_mat: torch.Tensor):


        # Apply the first Graph Attention layer
        x = self.gat1(input_tensor, adj_mat)
        x = F.elu(x) # Apply ELU activation function to the output of the first layer  TODO 多头网络的时候才使用这个激活函数？

        # Apply the second Graph Attention layer
        x = self.gat2(x, adj_mat)

        return F.softmax(x, dim=1) # Apply softmax activation function

"""ESWA里边写的Attetion"""
# class Attention(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Attention, self).__init__()
#         self.hidden_size = hidden_size
#         self.W1 = nn.Linear(input_size, hidden_size, bias=False)  # TODO 采用weight的意思就是无偏的的，no bias
#         self.W2 = nn.Linear(input_size, hidden_size, bias=False)
#         self.fcl = nn.Linear(hidden_size, 1, bias=False)  # TODO 单层前馈神经网络模型就1层？
#         self.activation = nn.LeakyReLU()
#
#     def forward(self, x1, x2):
#         # (batch_size, max_seq_len, hidden_size)
#         # x1 = self.activation(self.W1(x1))   # TODO 激活函数用什么？
#         x1 = self.W1(x1)   # transform，因为fea的特征向量不一样，所以
#
#         # (batch_size, 1 (unsqueezed), hidden_size)
#         # x2 = self.W2(x2).unsqueeze(1)
#         # x2 = self.activation(self.W2(x2))
#         x2 = self.W2(x2)
#
#         # TODO 我的两个m的fea应该是不同的特征，要concat！
#         x = torch.cat((x1, x2), dim=-1)
#
#         # 1st line of Eq.(3) in the paper
#         # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
#         u_i = self.fcl(torch.tanh(x1 + x2)).squeeze(-1)
#         # u_i = self.fcl(x)  # TODO 不用再softmax了吧，一个值怎么softmax？
#         return u_i




"""=================不同的GAT的实现方式============================"""
#
# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     图注意力层
#     """
#
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.in_features = in_features  # 节点表示向量的输入特征维度
#         self.out_features = out_features  # 节点表示向量的输出特征维度
#         self.dropout = dropout  # dropout参数
#         self.alpha = alpha  # leakyrelu激活的参数
#         self.concat = concat  # 如果为true, 再进行elu激活
#
#         # 定义可训练参数，即论文中的W和a
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
#         self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化
#
#         # 定义leakyrelu激活函数
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, inp, adj):
#         """
#         inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
#         adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
#         """
#         h = torch.mm(inp, self.W)  # [N, out_features]
#         N = h.size()[0]  # N 图的节点数
#
#         # N*1, out*N -> N*N, out 和  N*N, out*1
#         a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
#         # [N, N, 2*out_features]
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
#         # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
#
#         zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
#         attention = torch.where(adj > 0, e, zero_vec)  # [N, N]  >0选择e，不满足选择zero_vec
#         # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
#         # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
#         attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
#         attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
#         h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
#         # 得到由周围节点通过注意力权重进行更新的表示
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
#
# """
# 多头：multi-head机制
# """
# class GAT(nn.Module):
#     def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
#         """Dense version of GAT
#         n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
#         从不同的子空间进行抽取特征。
#         """
#         super(GAT, self).__init__()
#         self.dropout = dropout
#
#         # 定义multi-head的图注意力层
#         self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
#         # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
#         self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
#         x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
#         x = F.elu(self.out_att(x, adj))  # 输出并激活
#         return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定

"""===============================另外一种GAT的写法================================"""


# import torch
# from torch import nn
# import torch.nn.functional as F
#
# ################################
# ### GAT LAYER DEFINITION ###
# ################################
#
# class GraphAttentionLayer(nn.Module):
#
#     def __init__(self, in_features: int, out_features: int,
#     n_heads: int, concat: bool = False, dropout: float = 0.4,
#     leaky_relu_slope: float = 0.2):
#         super(GraphAttentionLayer, self).__init__()
#
#         self.n_heads = n_heads # Number of attention heads
#         self.concat = concat # wether to concatenate the final attention heads
#         self.dropout = dropout # Dropout rate
#
#         if concat: # concatenating the attention heads
#             self.out_features = out_features # Number of output features per node
#             assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
#             self.n_hidden = out_features // n_heads
#         else: # averaging output over the attention heads (Used in the main paper)
#             self.n_hidden = out_features
#
#         # A shared linear transformation, parametrized by a weight matrix W is applied to every node
#         # Initialize the weight matrix W
#         self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))
#
#         # Initialize the attention weights a
#         self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))
#
#         self.leakyrelu = nn.LeakyReLU(leaky_relu_slope) # LeakyReLU activation function
#         self.softmax = nn.Softmax(dim=1) # softmax activation function to the attention coefficients
#
#         self.reset_parameters() # Reset the parameters
#
#
#     def reset_parameters(self):
#
#         nn.init.xavier_normal_(self.W)  # 将权重矩阵W的元素从 Xavier 正态分布中采样得到，其中均值为0，标准差根据权重矩阵的大小进行调整，以保持方差相等。
#         nn.init.xavier_normal_(self.a)
#
#     def _get_attention_scores(self, h_transformed: torch.Tensor):
#
#         source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
#         target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])
#
#         # broadcast add
#         # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
#         e = source_scores + target_scores.mT
#         return self.leakyrelu(e)
#
#     def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
#
#         n_nodes = h.shape[0]
#
#         # Apply linear transformation to node feature -> W h
#         # output shape (n_nodes, n_hidden * n_heads)
#         h_transformed = torch.mm(h, self.W)
#         h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)
#
#         # splitting the heads by reshaping the tensor and putting heads dim first
#         # output shape (n_heads, n_nodes, n_hidden)
#         # 参数1（第一个参数）：表示原始张量中的维度索引1（从0开始计数），将会成为新张量中的第0维度。
#         h_transformed = h_transformed.view(n_nodes, self.n_heads, self.n_hidden).permute(1, 0, 2)  # permute维度变换新维度参数（第一，第二，第三），从旧维度（1，0,2）变换来
#
#         # getting the attention scores
#         # h_transformed =  (n_heads, n_nodes, n_hidden)
#         # a = (n_heads, 2 * self.n_hidden, 1)  取第二个维度的一半
#         # output shape (n_heads, n_nodes, n_nodes)
#         e = self._get_attention_scores(h_transformed)
#
#         # Set the attention score for non-existent edges to -9e15 (MASKING NON-EXISTENT EDGES)
#         connectivity_mask = -9e16 * torch.ones_like(e)
#         e = torch.where(adj_mat > 0, e, connectivity_mask) # masked attention scores
#
#         # attention coefficients are computed as a softmax over the rows
#         # for each column j in the attention score matrix e
#         attention = F.softmax(e, dim=-1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#
#         # final node embeddings are computed as a weighted average of the features of its neighbors
#         h_prime = torch.matmul(attention, h_transformed)
#
#         # concatenating/averaging the attention heads
#         # output shape (n_nodes, out_features)
#         if self.concat:
#             h_prime = h_prime.permute(1, 0, 2).contiguous().view(n_nodes, self.out_features)
#         else:
#             h_prime = h_prime.mean(dim=0)
#
#         return h_prime
#
# class GAT(nn.Module):
#
#     def __init__(self,
#                 in_features,
#                 n_hidden,
#                 n_heads,
#                 num_classes,
#                 concat=False,
#                 dropout=0.4,
#                 leaky_relu_slope=0.2):
#
#         super(GAT, self).__init__()
#
#         # Define the Graph Attention layers
#         self.gat1 = GraphAttentionLayer(
#         in_features=in_features, out_features=n_hidden, n_heads=n_heads,
#         concat=concat, dropout=dropout, leaky_relu_slope=leaky_relu_slope
#         )
#
#         self.gat2 = GraphAttentionLayer(
#         in_features=n_hidden, out_features=num_classes, n_heads=1,
#         concat=False, dropout=dropout, leaky_relu_slope=leaky_relu_slope
#         )
#
#     def forward(self, input_tensor: torch.Tensor , adj_mat: torch.Tensor):
#
#
#         # Apply the first Graph Attention layer
#         x = self.gat1(input_tensor, adj_mat)
#         x = F.elu(x) # Apply ELU activation function to the output of the first layer
#
#         # Apply the second Graph Attention layer
#         x = self.gat2(x, adj_mat)
#
#         return F.softmax(x, dim=1) # Apply softmax activation function