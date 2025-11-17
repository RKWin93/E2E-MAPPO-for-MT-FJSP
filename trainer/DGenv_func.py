"""
1、一个node如果有很多个入边，说明有很多个先后约束
2、这么多先后约束，我们只能选结束的最晚的时间（=上一步的ft+对应的运输t），也就是当前节点最大的开始时间（因为有约束存在啊！所以必须选最大）
"""
import numpy as np


"""def find_max_st_in_inedge(G, taskid, edge_init_weight):
    inedge_list = list(G.in_edges(taskid)) # 找到所有入边
    max_list = []
    for i in range(len(inedge_list)):  # 遍历所有入边！
        prev_nodeid, _ = list(G.in_edges(taskid))[i] #第一个入边的节点id
        edge_weight = edge_init_weight[G.nodes[prev_nodeid]['machine'], G.nodes[taskid]['machine']]  #计算入边节点到当前节点的运输时间t

        # 修改BUG，防止src节点的m_id=-1查表导致错误，应该是边权重为0，但是查表得到了，担心（-1，x）的情形吧
        # 当前节点task_id不会出现虚拟的sink节点的，总共就几个子任务就运行几遍，所以不用担心（x，-1）的情形
        if G.nodes[prev_nodeid]['machine'] < 0:
            edge_weight = 0  # 此时没有运输时间！！！

        max_val = G.nodes[prev_nodeid]['finish_time'] + edge_weight  #入边节点的完工时间+运输时间 = 当前taskid节点的开始时间
        max_list.append(max_val)
    max_st_currentTask = max(max_list)  #必定要选最大值，因为是有先后约束的，不能选小的先开始的，约束不满足！

    return max_st_currentTask"""

"""
1、本质：寻找当前节点的所有的入边（都是先后约束），最晚结束的入边 = 当前节点的最早开始时间 = 当前节点的最晚的到达时间
2、node图 + 当前节点 + 传输t能力矩阵 + job个数 + machine个数 + 传输时间次数和累加的统计 + 用来记录已经查到过的平行边的传输时间t
3、需要判断：1、前置节点不是src源节点，（-1,1）会倒着查表；2、同一个job的子任务之间才存在运输时间
4、machine_num 严格等于 每一个job的子任务数量（正方形矩阵）
5、energy_transport 为了统计有几个传输时间，累加的传输时间是多少

BUG：
# 修改BUG，防止src节点的m_id=-1查表导致错误，应该是边权重为0，但是查表得到了，担心（-1，x）的情形吧
# 当前节点task_id不会出现虚拟的sink节点的，总共就几个子任务就运行几遍，所以不用担心（x，-1）的情形

6、避免被反复查运输t（平行边）的最简单方法：画完图，每个平行边和其出发节点的weight和duration之间的差值之和就行（或者画平行边的时候，再记录，次数 = jobN*（machineN-1））！！！
7、一个是有结果之后的查询，一个是在调度过程中的统计（前者验证后者，保证调度没问题）
"""

"""
根据当前节点（`taskid`）的入边和其他条件找到最晚到达时间
1、`G.in_edges('D')`的返回值将是`[('B', 'D'), ('C', 'D')]`。其中，每个元组表示一条边，第一个元素是起始节点，第二个元素是终点节点。
2、肯定选最大值作为当前节点的st，因为入边都是先后顺序的约束
"""
def find_max_arrivaTime_for_currentNode(G, taskid, edge_init_weight, configs):
    inedge_list = list(G.in_edges(taskid))  # 找到所有入边, 返回list中存有n个元组（起始节点，taskID节点）
    max_list = []
    for i in range(len(inedge_list)):  # 遍历所有入边！
        prev_nodeid, _ = inedge_list[i]  # 入边的节点id
        transport_t = edge_init_weight[G.nodes[prev_nodeid]['machine'], G.nodes[taskid]['machine']]  # 计算入边节点到当前节点的运输时间t

        if G.nodes[prev_nodeid]['machine'] < 0:  # 排除src的节点，m_id=-1
            transport_t = 0  # 此时没有运输时间！！！

        same, same_job_id = find_2nodeID_in_same_job(prev_nodeid, taskid, configs.n_job, configs.n_machine)  # 判断输入的两个节点是否位于同一个job,和是哪一个job

        if not same:  # 不是同一个job中
            transport_t = 0

        max_val = G.nodes[prev_nodeid]['finish_time'] + transport_t  # 入边节点的完工时间+运输时间 = 当前taskid节点的开始时间
        max_list.append(max_val)
        # print(exist_list)
    st_currentTask = max(max_list)  # 必定要选最大值，因为是有先后约束的，不能选小的先开始的，约束不满足！

    return st_currentTask


"""
# 查找两个节点是否位于同一个job中，笨方法：
1、按照行j查找，查j次，两个节点分别记录在哪一行
2、同行，返回true和job的行数，否则不同行
"""
def find_2nodeID_in_same_job(prev_nodeid, taskid, nj, nm) -> (bool, int):
    n_job = nj
    n_machine = nm
    total_task = n_job * n_machine
    task_list = np.arange(1, (total_task + 1)).reshape(n_job, -1)  # -1不指定维度  j*m的taskid的矩阵
    # print("task_list", task_list)

    job_current = 0
    job_prev = 0  # 虽然初始化了，for之后就会重新赋值了

    same_job_id = 0

    for i in range(task_list.shape[0]):
        if taskid in task_list[i]:
            job_current = i
        if prev_nodeid in task_list[i]:
            job_prev = i

    if job_current == job_prev:
        same_job = True  # 位于同一个job，有运输t
        same_job_id = job_current  # 是第几个job的平行边
    else:
        same_job = False

    return same_job, same_job_id


"""
# 查表node1到node2之间的运输时间t，并判断是否有效（同一个job）
# energy_transport用来记录运输了几次 + exist_list用来记录已经查到过的平行边的传输时间t

1、先通过task的id找到都是选定的哪些m，然后查表m之间是否有transT
2、还要判断这些taskid是否是同一个m的，不然也没有transT
"""
def find_transportT(G, prev_node1, curr_node2, edge_init_weight, configs):
    # print(G.nodes[node1]['machine'])
    # print(G.nodes[node2]['machine'])  # 为什么node的machine的属性会变成浮点数？？？？？？？
    """"""
    """
    因为现在初始化存在m_id=-1小于0的task节点，所以这些节点没必要判断运输时间（__schedule之前都会先更新当前task节点的属性信息：m_id + color + dur！！！！）
    """
    if G.nodes[curr_node2]['machine'] >= 0:  # 防止初始化load_instance的时候无效判断transT
        transport_t = edge_init_weight[G.nodes[prev_node1]['machine'], G.nodes[curr_node2]['machine']]  # 查表分配的m——id得要新增的运输时间
    else:
        transport_t = 0

    same, same_job_id = find_2nodeID_in_same_job(prev_node1, curr_node2, configs.n_job, configs.n_machine)  # 判断输入的两个节点是否位于同一个job，和是哪一个job

    if G.nodes[prev_node1]['machine'] < 0:  # 排除src的节点，m_id=-1
        transport_t = 0  # 此时没有运输时间！！！

    if not same:
        transport_t = 0

    return transport_t


"""
1、需要反馈参数（y，width，left） 对应 node[m_id+1, ft-st, st]：直接从machien_route里边获取
2、可以基于当前的machien_route来输出实时的idle指标 + 最终的指标

machine_route里边的task是有顺序的，代表当前m被选取后的task的先后执行顺序
    machien_routes = {0: array([], dtype=int64), 1: array([3, 4, 1, 2])}  字典形式：存有同一个m中的task的加工顺序！

总体理解：
1、每个节点的开始-上一个节点的结束
2、首节点就是开始-0
3、不断累加其中的空白！
4、其中必然会包含transT的时间，有的时候是别的task的dur，但是说实际的，有这时间必须要等待的（无非就是最后不算it？）
"""
def calculate_idle_t_for_each_machine(G, machineRoute, p2_ins):
    blank = 0
    sum_idle = 0
    count = 0  # 画图的纵坐标
    # for m_id in machineRoute.keys():  # key的名字
    #     count += 1
    # for route in machineRoute.values():  # 字典的value的列表
    for m_index, route in machineRoute.items():  # 字典的key和value的列表  todo 1113-新增空闲功率的存在！
        # 防止某些设备列表中只有一个任务
        if len(route) == 1:
            blank = G.nodes[route[0]]['start_time'] - 0  # 空闲时间就是其开始时间和0的偏差
            blank = blank * p2_ins[0][m_index]   # 空闲的时间 * 对应m的空闲功率！
            sum_idle = sum_idle + blank
        elif len(route) > 1:  # 防止在每一步运行的时候，出现空集的操作
            # 每个列表的首个任务，计算其空闲时间
            blank = G.nodes[route[0]]['start_time'] - 0
            blank = blank * p2_ins[0][m_index]  # 空闲的时间 * 对应m的空闲功率！
            sum_idle = sum_idle + blank

            for i in range(len(route) - 1):  # 每个value列表中的元素个数 - 1
                blank = G.nodes[route[i + 1]]['start_time'] - G.nodes[route[i]]['finish_time']  # 遍历每个列表中的元素：后一个st开始 - 前一个ft结束
                blank = blank * p2_ins[0][m_index]  # 空闲的时间 * 对应m的空闲功率！
                sum_idle = sum_idle + blank  # 遍历玩所有元素，求和

    """if_print(1, "根据machine_route计算的：空闲时间{}".format(sum_idle))"""

    return sum_idle

