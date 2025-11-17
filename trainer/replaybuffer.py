import torch
import numpy as np
from trainer.train_device import device



"""
会有total_step是因为我每一step都记录进buffer,然后buffer_size= 5 倍大小的episode（记录5轮episode，每轮都有task个step）
state_m = [m+t+e1] + fx  step * batch * channel * j * m （CNN）
state_m = [m+t+e1] + fx  step * batch * （j * m） （MLP）
action_prob:  step * batch * m  （所有的action的概率）
具体的select_action：step * batch
reward: step * batch 
done： step * batch

total_step * env_batch * xxx  缓存的batch + 并行环境的batch + 数据大小
"""
class ReplayBuffer:
    def __init__(self, args):  # args是我定义的全局参数，用parse传入的

        """会有total_step是因为我每一step都记录进buffer,然后buffer_size= 5 倍大小的episode（记录5轮episode，每轮都有task个step）"""
        
        self.total_task = args['n_job'] * args['n_machine'] # 一个episode会遍历所有的task
        self.total_step = args['buffer_size'] * self.total_task   # buffer多大，所以维度就是多大
        
        """
        记录选择operation的经验池：
        1、扩加一个训练时记录的总步数total_step 
        2、其余大小等于传入的变量的原大小
        """
        self.adj = np.zeros((self.total_step, args['env_batch'], self.total_task, self.total_task))
        self.tasks_fea = np.zeros((self.total_step, args['env_batch'] * self.total_task, args['gcn_input_dim'])) # args['gcn_input_dim'] = task_fea的特征向量的个数
        self.candidate = np.zeros((self.total_step, args['env_batch'], args['n_job']))
        self.mask_operation = torch.zeros((self.total_step, args['env_batch'], args['n_job']),dtype=torch.bool)  # mask已经是一个tensor，我给转成bool了

        self.adj_ = np.zeros((self.total_step, args['env_batch'], self.total_task, self.total_task))
        self.tasks_fea_ = np.zeros((self.total_step, args['env_batch'] * self.total_task, args['gcn_input_dim'])) # args['gcn_input_dim'] = task_fea的特征向量的个数
        self.candidate_ = np.zeros((self.total_step, args['env_batch'], args['n_job']))
        self.mask_operation_ = torch.zeros((self.total_step, args['env_batch'], args['n_job']),dtype=torch.bool)

        # TODO: 新增一个machine_mask，因为m的能力出现了Minus！
        self.mask_machine_ = torch.zeros((self.total_step, args['env_batch'], 1, args['n_machine']), dtype=torch.bool)

        self.a_operation = torch.zeros((self.total_step, args['env_batch']),dtype=torch.long) # env_batch个选择的m （张量矩阵）
        self.a_logprob_operation = torch.zeros((self.total_step, args['env_batch']),dtype=torch.float)  # 所选择的具体select_action的对数概率 （张量矩阵）

        self.r_operation = np.zeros((self.total_step, args['env_batch']))

        # todo 新增4个指标r的记录
        self.mk = np.zeros((self.total_step, args['env_batch']))
        self.pt = np.zeros((self.total_step, args['env_batch']))
        self.tt = np.zeros((self.total_step, args['env_batch']))
        self.it = np.zeros((self.total_step, args['env_batch']))

        # self.dw = np.zeros((self.total_step, 1))  # 不存在没有下一个state的情况！
        self.done_operation = np.zeros((self.total_step, args['env_batch']))

        self.machine_fea1 = np.zeros((self.total_step, args['env_batch'], args['n_machine'], 6))  # machine_fea = env_batch * m * 6
        self.machine_fea2 = np.zeros((self.total_step, args['env_batch'], args['n_machine'], 8))  # machine_fea = env_batch * m * 5
        self.machine_fea2_ = np.zeros((self.total_step, args['env_batch'], args['n_machine'], 8))
        # self.h_mch_pooled = torch.zeros((self.total_step, args['env_batch'], args.machine_hidden_dim),
        #                                 dtype=torch.float)  # 张量：m节点的全局池化平均嵌入 = batch * hidden_dim

        self.a = torch.zeros((self.total_step, args['env_batch']), dtype=torch.long)  # env_batch个选择的m （张量矩阵）
        self.a_logprob = torch.zeros((self.total_step, args['env_batch']),dtype=torch.float)  # 所选择的具体select_action的对数概率 （张量矩阵）

        # todo 1108-新增不同bs的随机权重，记录下来，虽然整个Minibatch里边数值都一样
        self.random_weight = np.zeros((self.total_step, args['env_batch'], 3))

        # todo 1113-新增记录job和machine网络运行时输出的v，你在更新时用的记录的state，不更新网络参数，输出的v都一样的！
        # todo 1224-!!!long是长整形，不是长float，注意！
        self.job_v = torch.zeros((self.total_step, args['env_batch'], 2), dtype=torch.float)
        self.machine_v = torch.zeros((self.total_step, args['env_batch'], 2), dtype=torch.float)

        self.job_v_ = torch.zeros((self.total_step, args['env_batch'], 2), dtype=torch.float)
        self.machine_v_ = torch.zeros((self.total_step, args['env_batch'], 2), dtype=torch.float)

        self.count_operation = 0
        self.count_operation_ = 0  # todo 1113-只针对记录v_才使用！

    # update之后就会把count=0.这样子就是重新重头开始记录，相当于清空
    def store_operation(self, adj, fea, candidate, mask, a_o, a_o_logprob, r,
                        adj_, fea_, candidate_, mask_,
                        mch_fea1, mch_fea2, mch_fea2_, a_m, a_m_logprob, dw, done, mask_machine_,
                        mk, pt, tt, it, rw, j_v, m_v):  # 除了action相关是张量，其他的都是数组！（多此一举，可以直接初始化一个张量矩阵！）
        self.adj[self.count_operation] = adj
        self.tasks_fea[self.count_operation] = fea
        self.candidate[self.count_operation] = candidate
        self.mask_operation[self.count_operation] = mask

        self.a_operation[self.count_operation] = a_o
        self.a_logprob_operation[self.count_operation] = a_o_logprob

        self.r_operation[self.count_operation] = r

        # todo 新增4个指标r的记录
        self.mk[self.count_operation] = mk
        self.pt[self.count_operation] = pt
        self.tt[self.count_operation] = tt
        self.it[self.count_operation] = it


        self.adj_[self.count_operation] = adj_
        self.tasks_fea_[self.count_operation] = fea_
        self.candidate_[self.count_operation] = candidate_
        self.mask_operation_[self.count_operation] = mask_
        self.mask_machine_[self.count_operation] = mask_machine_


        # self.dw[self.count_operation] = dw
        self.done_operation[self.count_operation] = done

        self.machine_fea1[self.count_operation] = mch_fea1
        self.machine_fea2[self.count_operation] = mch_fea2
        self.a[self.count_operation] = a_m
        self.a_logprob[self.count_operation] = a_m_logprob
        self.machine_fea2_[self.count_operation] = mch_fea2_

        # todo 1108-新增不同bs的随机权重，记录下来，虽然整个Minibatch里边数值都一样
        self.random_weight[self.count_operation] = rw

        # todo 1113-新增记录job和machine网络运行时输出的v，你在更新时用的记录的state，不更新网络参数，输出的v都一样的！
        self.job_v[self.count_operation] = j_v
        self.machine_v[self.count_operation] = m_v

        self.count_operation += 1  # 记录缓存次数的变量在此！！！

        # self.h_mch_pooled[self.count] = h_mch_pooled

    """
    专门写的用来记录job和machine网络的下一时刻的v_
    1、牵扯到全图特征的交叉使用
    2、牵扯到m_fea1_的问题，搞起来好麻烦啊
    """
    def store_v_next(self, j_v_, m_v_):
        self.job_v_[self.count_operation_] = j_v_
        self.machine_v_[self.count_operation_] = m_v_

        self.count_operation_ += 1  # 记录的是v_的次数，记录缓存次数的变量在此！！！

    def numpy_to_tensor_operation(self):
        adj = torch.tensor(self.adj, dtype=torch.float).to(device)
        tasks_fea = torch.tensor(self.tasks_fea, dtype=torch.float).to(device)
        candidate = torch.tensor(self.candidate, dtype=torch.long).to(device)  # candidate的
        mask_operation = self.mask_operation.to(device)   # 已经是张量tensor了！

        adj_ = torch.tensor(self.adj_, dtype=torch.float).to(device)
        tasks_fea_ = torch.tensor(self.tasks_fea_, dtype=torch.float).to(device)
        candidate_ = torch.tensor(self.candidate_, dtype=torch.float).to(device)
        mask_operation_ = self.mask_operation_.to(device)  # 已经是张量tensor了！
        mask_machine_ = self.mask_machine_.to(device)  # 已经是张量tensor了！

        """
        本身选择action的时候都是with torch.no_grad(): 不带梯度的, 所以用.clone().detach()???
        Trick代码选择action时：return a.numpy()[0], a_logprob.numpy()[0]  
        # 返回action和a_logprob的时候都转成numpy，然后只取其中的第一个数值，根本没有梯度的
        """
        # a = torch.tensor(self.a, dtype=torch.long).to(device) # In discrete action space, 'a' needs to be torch.long 长整形
        # a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(device)
        a_operation = self.a_operation.to(device)   # 初始化的时候已经是tensor了！而且数据类型也制定好了!
        a_logprob_operation = self.a_logprob_operation.to(device)  # 已经是张量tensor了！

        r_operation = torch.tensor(self.r_operation, dtype=torch.float).to(device)

        # todo 新增4个指标r的记录
        mk = torch.tensor(self.mk, dtype=torch.float).to(device)
        pt = torch.tensor(self.pt, dtype=torch.float).to(device)
        tt = torch.tensor(self.tt, dtype=torch.float).to(device)
        it = torch.tensor(self.it, dtype=torch.float).to(device)

        # todo 1108-新增不同bs的随机权重，记录下来，虽然整个Minibatch里边数值都一样
        rw = torch.tensor(self.random_weight, dtype=torch.float).to(device)


        # dw = torch.tensor(self.dw, dtype=torch.float).to(device)
        done_operation = torch.tensor(self.done_operation, dtype=torch.float).to(device)

        """
        本身选择action的时候都是with torch.no_grad(): 不带梯度的, 所以用.clone().detach()???
        Trick代码选择action时：return a.numpy()[0], a_logprob.numpy()[0]  
        # 返回action和a_logprob的时候都转成numpy，然后只取其中的第一个数值，根本没有梯度的
        """
        # a = torch.tensor(self.a, dtype=torch.long).to(device) # In discrete action space, 'a' needs to be torch.long 长整形
        # a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(device)
        a = self.a.to(device)  # 初始化的时候已经是tensor了！而且数据类型也制定好了!
        a_logprob = self.a_logprob.to(device)

        machine_fea1 = torch.tensor(self.machine_fea1, dtype=torch.float).to(device)
        machine_fea2 = torch.tensor(self.machine_fea2, dtype=torch.float).to(device)
        machine_fea2_ = torch.tensor(self.machine_fea2_, dtype=torch.float).to(device)
        # dw = torch.tensor(self.dw, dtype=torch.float).to(device)
        # h_mch_pooled = self.h_mch_pooled.to(device)  # 已经是tensor了

        # todo 1113-新增记录job和machine网络运行时输出的v，你在更新时用的记录的state，不更新网络参数，输出的v都一样的！
        job_v = self.job_v.to(device)  # 已经是张量tensor了！
        machine_v = self.machine_v.to(device)  # 已经是张量tensor了！
        job_v_ = self.job_v_.to(device)  # 已经是张量tensor了！
        machine_v_ = self.machine_v_.to(device)  # 已经是张量tensor了！


        return adj, tasks_fea, candidate, mask_operation, a_operation, a_logprob_operation,\
               adj_, tasks_fea_, candidate_, mask_operation_, r_operation, done_operation, \
               machine_fea2, a, a_logprob, machine_fea2_, mask_machine_, \
               mk, pt, tt, it, machine_fea1, rw, job_v, machine_v, job_v_, machine_v_



class esa_ReplayBuffer_JointActor_GAT:
    def __init__(self, args):  # args是我定义的全局参数，用parse传入的

        """"""
        self.total_task = args['n_job'] * args['n_machine'] # 一个episode会遍历所有的task
        self.total_step = args['buffer_size'] * self.total_task   # buffer多大，所以维度就是多大
        """
        记录选择operation的经验池：
        1、扩加一个总共的个数 = train_batch的大小
        2、其余大小等于传入的变量的原大小
        """
        self.adj = np.zeros((self.total_step, args['env_batch'], self.total_task, self.total_task))
        self.tasks_fea = np.zeros((self.total_step, args['env_batch'] * self.total_task, args['gcn_input_dim'])) # args['gcn_input_dim'] = task_fea的特征向量的个数
        self.candidate = np.zeros((self.total_step, args['env_batch'], args['n_job']))
        self.mask_operation = torch.zeros((self.total_step, args['env_batch'], args['n_job']),dtype=torch.bool)  # mask已经是一个tensor，我给转成bool了

        self.adj_ = np.zeros((self.total_step, args['env_batch'], self.total_task, self.total_task))
        self.tasks_fea_ = np.zeros((self.total_step, args['env_batch'] * self.total_task, args['gcn_input_dim'])) # args['gcn_input_dim'] = task_fea的特征向量的个数
        self.candidate_ = np.zeros((self.total_step, args['env_batch'], args['n_job']))
        self.mask_operation_ = torch.zeros((self.total_step, args['env_batch'], args['n_job']),dtype=torch.bool)

        # TODO: 新增一个machine_mask，因为m的能力出现了Minus！
        self.mask_machine_ = torch.zeros((self.total_step, args['env_batch'], 1, args.n_machine), dtype=torch.bool)

        self.a_operation = torch.zeros((self.total_step, args['env_batch']),dtype=torch.long) # env_batch个选择的m （张量矩阵）
        self.a_logprob_operation = torch.zeros((self.total_step, args['env_batch']),dtype=torch.float)  # 所选择的具体select_action的对数概率 （张量矩阵）

        self.r_operation = np.zeros((self.total_step, args['env_batch']))

        # todo 新增4个指标r的记录
        self.mk = np.zeros((self.total_step, args['env_batch']))
        self.pt = np.zeros((self.total_step, args['env_batch']))
        self.tt = np.zeros((self.total_step, args['env_batch']))
        self.it = np.zeros((self.total_step, args['env_batch']))

        # self.dw = np.zeros((self.total_step, 1))  # 不存在没有下一个state的情况！
        self.done_operation = np.zeros((self.total_step, args['env_batch']))

        self.machine_fea1 = np.zeros((self.total_step, args['env_batch'], args.n_machine, 6))  # machine_fea = env_batch * m * 6
        self.machine_fea2 = np.zeros((self.total_step, args['env_batch'], args.n_machine, 8))  # machine_fea = env_batch * m * 5
        self.machine_fea2_ = np.zeros((self.total_step, args['env_batch'], args.n_machine, 8))
        # self.h_mch_pooled = torch.zeros((self.total_step, args['env_batch'], args.machine_hidden_dim),
        #                                 dtype=torch.float)  # 张量：m节点的全局池化平均嵌入 = batch * hidden_dim

        self.a = torch.zeros((self.total_step, args['env_batch']), dtype=torch.long)  # env_batch个选择的m （张量矩阵）
        self.a_logprob = torch.zeros((self.total_step, args['env_batch']),dtype=torch.float)  # 所选择的具体select_action的对数概率 （张量矩阵）

        # todo 1108-新增不同bs的随机权重，记录下来，虽然整个Minibatch里边数值都一样
        self.random_weight = np.zeros((self.total_step, args['env_batch'], 3))

        # todo 1113-新增记录job和machine网络运行时输出的v，你在更新时用的记录的state，不更新网络参数，输出的v都一样的！
        self.job_v = torch.zeros((self.total_step, args['env_batch'], 4), dtype=torch.float) #todo 1223-ESA里边的job网络的critic直接视为全局，所以有4个变量！！！
        self.machine_v = torch.zeros((self.total_step, args['env_batch'], 2), dtype=torch.float)

        self.job_v_ = torch.zeros((self.total_step, args['env_batch'], 4), dtype=torch.float) #todo 1223-ESA里边的job网络的critic直接视为全局，所以有4个变量！！！
        self.machine_v_ = torch.zeros((self.total_step, args['env_batch'], 2), dtype=torch.float)

        self.count_operation = 0
        self.count_operation_ = 0  # todo 1113-只针对记录v_才使用！

    # update之后就会把count=0.这样子就是重新重头开始记录，相当于清空
    def store_operation(self, adj, fea, candidate, mask, a_o, a_o_logprob, r,
                        adj_, fea_, candidate_, mask_,
                        mch_fea1, mch_fea2, mch_fea2_, a_m, a_m_logprob, dw, done, mask_machine_,
                        mk, pt, tt, it, rw, j_v, m_v):  # 除了action相关是张量，其他的都是数组！（多此一举，可以直接初始化一个张量矩阵！）
        self.adj[self.count_operation] = adj
        self.tasks_fea[self.count_operation] = fea
        self.candidate[self.count_operation] = candidate
        self.mask_operation[self.count_operation] = mask

        self.a_operation[self.count_operation] = a_o
        self.a_logprob_operation[self.count_operation] = a_o_logprob

        self.r_operation[self.count_operation] = r

        # todo 新增4个指标r的记录
        self.mk[self.count_operation] = mk
        self.pt[self.count_operation] = pt
        self.tt[self.count_operation] = tt
        self.it[self.count_operation] = it


        self.adj_[self.count_operation] = adj_
        self.tasks_fea_[self.count_operation] = fea_
        self.candidate_[self.count_operation] = candidate_
        self.mask_operation_[self.count_operation] = mask_
        self.mask_machine_[self.count_operation] = mask_machine_


        # self.dw[self.count_operation] = dw
        self.done_operation[self.count_operation] = done

        self.machine_fea1[self.count_operation] = mch_fea1
        self.machine_fea2[self.count_operation] = mch_fea2
        self.a[self.count_operation] = a_m
        self.a_logprob[self.count_operation] = a_m_logprob
        self.machine_fea2_[self.count_operation] = mch_fea2_

        # todo 1108-新增不同bs的随机权重，记录下来，虽然整个Minibatch里边数值都一样
        self.random_weight[self.count_operation] = rw

        # todo 1113-新增记录job和machine网络运行时输出的v，你在更新时用的记录的state，不更新网络参数，输出的v都一样的！
        self.job_v[self.count_operation] = j_v
        self.machine_v[self.count_operation] = m_v

        self.count_operation += 1  # 记录缓存次数的变量在此！！！

        # self.h_mch_pooled[self.count] = h_mch_pooled

    """
    专门写的用来记录job和machine网络的下一时刻的v_
    1、牵扯到全图特征的交叉使用
    2、牵扯到m_fea1_的问题，搞起来好麻烦啊
    """
    def store_v_next(self, j_v_, m_v_):
        self.job_v_[self.count_operation_] = j_v_
        self.machine_v_[self.count_operation_] = m_v_

        self.count_operation_ += 1  # 记录缓存次数的变量在此！！！

    def numpy_to_tensor_operation(self):
        adj = torch.tensor(self.adj, dtype=torch.float).to(device)
        tasks_fea = torch.tensor(self.tasks_fea, dtype=torch.float).to(device)
        candidate = torch.tensor(self.candidate, dtype=torch.long).to(device)  # candidate的
        mask_operation = self.mask_operation.to(device)   # 已经是张量tensor了！

        adj_ = torch.tensor(self.adj_, dtype=torch.float).to(device)
        tasks_fea_ = torch.tensor(self.tasks_fea_, dtype=torch.float).to(device)
        candidate_ = torch.tensor(self.candidate_, dtype=torch.float).to(device)
        mask_operation_ = self.mask_operation_.to(device)  # 已经是张量tensor了！
        mask_machine_ = self.mask_machine_.to(device)  # 已经是张量tensor了！

        """
        本身选择action的时候都是with torch.no_grad(): 不带梯度的, 所以用.clone().detach()???
        Trick代码选择action时：return a.numpy()[0], a_logprob.numpy()[0]  
        # 返回action和a_logprob的时候都转成numpy，然后只取其中的第一个数值，根本没有梯度的
        """
        # a = torch.tensor(self.a, dtype=torch.long).to(device) # In discrete action space, 'a' needs to be torch.long 长整形
        # a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(device)
        a_operation = self.a_operation.to(device)   # 初始化的时候已经是tensor了！而且数据类型也制定好了!
        a_logprob_operation = self.a_logprob_operation.to(device)  # 已经是张量tensor了！

        r_operation = torch.tensor(self.r_operation, dtype=torch.float).to(device)

        # todo 新增4个指标r的记录
        mk = torch.tensor(self.mk, dtype=torch.float).to(device)
        pt = torch.tensor(self.pt, dtype=torch.float).to(device)
        tt = torch.tensor(self.tt, dtype=torch.float).to(device)
        it = torch.tensor(self.it, dtype=torch.float).to(device)

        # todo 1108-新增不同bs的随机权重，记录下来，虽然整个Minibatch里边数值都一样
        rw = torch.tensor(self.random_weight, dtype=torch.float).to(device)


        # dw = torch.tensor(self.dw, dtype=torch.float).to(device)
        done_operation = torch.tensor(self.done_operation, dtype=torch.float).to(device)

        """
        本身选择action的时候都是with torch.no_grad(): 不带梯度的, 所以用.clone().detach()???
        Trick代码选择action时：return a.numpy()[0], a_logprob.numpy()[0]  
        # 返回action和a_logprob的时候都转成numpy，然后只取其中的第一个数值，根本没有梯度的
        """
        # a = torch.tensor(self.a, dtype=torch.long).to(device) # In discrete action space, 'a' needs to be torch.long 长整形
        # a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(device)
        a = self.a.to(device)  # 初始化的时候已经是tensor了！而且数据类型也制定好了!
        a_logprob = self.a_logprob.to(device)

        machine_fea1 = torch.tensor(self.machine_fea1, dtype=torch.float).to(device)
        machine_fea2 = torch.tensor(self.machine_fea2, dtype=torch.float).to(device)
        machine_fea2_ = torch.tensor(self.machine_fea2_, dtype=torch.float).to(device)
        # dw = torch.tensor(self.dw, dtype=torch.float).to(device)
        # h_mch_pooled = self.h_mch_pooled.to(device)  # 已经是tensor了

        # todo 1113-新增记录job和machine网络运行时输出的v，你在更新时用的记录的state，不更新网络参数，输出的v都一样的！
        job_v = self.job_v.to(device)  # 已经是张量tensor了！
        machine_v = self.machine_v.to(device)  # 已经是张量tensor了！
        job_v_ = self.job_v_.to(device)  # 已经是张量tensor了！
        machine_v_ = self.machine_v_.to(device)  # 已经是张量tensor了！


        return adj, tasks_fea, candidate, mask_operation, a_operation, a_logprob_operation,\
               adj_, tasks_fea_, candidate_, mask_operation_, r_operation, done_operation, \
               machine_fea2, a, a_logprob, machine_fea2_, mask_machine_, \
               mk, pt, tt, it, machine_fea1, rw, job_v, machine_v, job_v_, machine_v_



"""
我就知道：tensor转tensor会报错！！！但是，先别管，能跑就行！
/remote-home/iot_wangrongkai/RUN/MO-FJSP-DRL/replaybuffer.py:40: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  a = torch.tensor(self.a, dtype=torch.long).to(device) # In discrete action space, 'a' needs to be torch.long
/remote-home/iot_wangrongkai/RUN/MO-FJSP-DRL/replaybuffer.py:41: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(device)
"""