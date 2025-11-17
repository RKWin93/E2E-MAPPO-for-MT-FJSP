import copy

import gym
import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt

from collections import OrderedDict
from typing import List, Union, Dict, Callable

from DGenv_func import find_transportT, find_max_arrivaTime_for_currentNode
from graph_jsp_env.disjunctive_graph_jsp_visualizer import DisjunctiveGraphJspVisualizer
from graph_jsp_env.disjunctive_graph_logger import log
# from parameters import configs
from wrk_function import find_latest_arrivaTime_for_currentNode, make_sure_transportT_work, \
     calculate_idle_t_for_each_machine

# todo 1224-为了实现多进程运行程序，同时可以修改参数，在每一个使用configs的文件增加下边语句： from parameters import parser + configs = parser.parse_args()
from parameters import parser
configs = parser.parse_args()


class DisjunctiveGraphJspEnv(gym.Env):
    """
    Custom Environment for the Job Shop Problem (jsp) that follows gym interface.

    This environment is inspired by the

        `The disjunctive graph machine representation of the job shop scheduling problem`

        by Jacek Błażewicz 2000

            https://www.sciencedirect.com/science/article/pii/S0377221799004865

    and

        `Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning`

        by Zhang, Cong, et al. 2020

            https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf

            https://github.com/zcaicaros/L2D

    This environment does not explicitly include disjunctive edges, like specified by Jacek Błażewicz,
    only conjunctive edges. Additional information is saved in the edges and nodes, such that one could construct
    the disjunctive edges, so the is no loss in information.
    Moreover, this environment does not implement the graph matrix datastructure by Jacek Błażewicz, since in provides
    no benefits in chosen the reinforcement learning stetting (for more details have a look at the
    master thesis).

    This environment is more similar to the Zhang, Cong, et al. implementation.
    Zhang, Cong, et al. seems to store exclusively time-information exclusively inside nodes
    (see Figure 2: Example of state transition) and no additional information inside the edges (like weights in the
    representation of Jacek Błażewicz).
    However, I had a rough time in understanding the code of Zhang, Cong, et al. 2020, so I might be wrong about that.

    The DisjunctiveGraphJssEnv uses the `networkx` library for graph structure and graph visualization.
    It is highly configurable and offers a lot of rendering options.

这个环境不显式地包括析取边，就像Jacek bzhaewicz指定的那样，只包括合取边。附加的信息保存在边和节点中，这样就可以构造析取边，因此信息不会丢失。
此外，这个环境没有实现Jacek bzhaewicz的图矩阵数据结构，因为它在选择强化学习方法时没有任何好处(更多细节请参阅硕士论文)。

这种环境更类似于Zhang, Cong等人的实现。Zhang, Cong等人似乎只在节点内部存储时间信息(参见图2:状态转换示例)，而在边缘内部不存储额外信息(如Jacek bzhaewicz表示中的权重)。
然而，我在理解Zhang, Cong, et al. 2020的代码时遇到了困难，所以我可能是错的。

DisjunctiveGraphJssEnv使用networkx库进行图形结构和图形可视化。它是高度可配置的，并提供了许多呈现选项。

    """


    # 是否用了强化学习？代码里边还是只是一个训练好的模型。Zhang用了GNN
    # 可以基于别人的开原模型，拼一下进行二次的创作，有的加一两行改进都能发论文；开源就是这个好处
    # 画图的操作没有看！

    #  DisjunctiveGraphJspEnv  新建一个类！
    metadata = {'render.modes': ['human', 'rgb_array', 'console']}

    def __init__(self,    #我盲猜，形参名字：类型 = 初始值
                 jps_instance: np.ndarray = None, *,   #jsp实例，自定义的矩阵，几个job：哪些设备，时间是多少？  j * m

                 # parameters for reward
                 reward_function='nasuta',    #反馈函数自定义为作者的名字了nasuta
                 custom_reward_function: Callable = None,  #Python中能被调用（called）的东西就是callable，例如function
                 reward_function_parameters: Dict = None,  #Dict字典，无序的、可变的序列，它的元素以“键值对（key-value）”的形式存储。

                 # parameters for observation
                 #flat() 方法会按照一个可指定的深度递归遍历数组，并将所有元素与遍历到的子数组中的元素合并为一个新数组返回。
                 normalize_observation_space: bool = True,  #观测空间，为真，value都是0-1之间
                 flat_observation_space: bool = True,  #观测空间，为真，扁平化：遍历数组，并组合成新数组返回，否则是矩阵
                 dtype: str = "float32",   #观测空间的数据类型

                 # parameters for actions
                 action_mode: str = "task",    #默认动作是子任务，也可以job or node
                 env_transform: str = None,   #
                 perform_left_shift_if_possible: bool = True,  #基于最大完工时间，一个任务（step method期间）加在2个任务之间不改变这两个的开始和结束时间，那就true加进去；否则排在后边；插空进去，时间更短

                 # parameters for rendering
                 c_map: str = "rainbow",   #渲染画图的matplotlib colormap
                 dummy_task_color="tab:gray",   #表示源和目的节点dummy node的颜色
                 default_visualisations: List[str] = None,  #渲染时候的可视化，["gantt_window", "gantt_console", "graph_window", "graph_console"]
                 visualizer_kwargs: dict = None, #额外的可视化参数
                 verbose: int = 0,   #决定是否有信息打印在console，0无1重要2全部

                 # wrk 传入信息
                 ability_tr_mm,   # 初始边的权重，即m到m的运输t的表格 m * m
                 ability_e_tm # 已知的对应被选择m的能耗  j * m
                 ):
        """

        :param jps_instance:                    a jsp instance as numpy array

        :param scaling_divisor:                 lower-bound of the jsp or some other scaling number for the reward.
                                                Only has an effect when `:param scale_reward` is `True`.
                                                If `None` is specified and `:param scale_reward` is `True` a naive
                                                lower-bound will be calculated automatically.

                                                If `scaling_divisor` is equal to the optimal makespan, then the (sparse)
                                                reward will be always smaller or equal to -1.

        :param scale_reward:                    `:param scaling_divisor` is only applied if set to `True`

        :param normalize_observation_space:     If set to `True` all values in the observation space will be between
                                                0.0 and 1.0.
                                                This includes an one-hot-encoding of the task-to-machine mapping.
                                                See `DisjunctiveGraphJssEnv._state_array`

        :param flat_observation_space:          If set to `True` the observation space will be flat. Otherwise, a matrix
                                                The exact size depends on the jsp size.

        :param dtype:                           the dtype for the observation space. Must follow numpy notation.

        :param action_mode:                     'task' or 'job'. 'task' is default. Specifies weather the
                                                `action`-argument of the `DisjunctiveGraphJssEnv.step`-method
                                                corresponds to a job or an task (or node in the graph representation)

                                                Note:

                                                    task actions and node_ids are shifted by 1.
                                                    So action = 0 corresponds to the node/task 1.


        :param perform_left_shift_if_possible:  if the specified task in the `DisjunctiveGraphJssEnv.step`-method can
                                                fit between two other task without changing their start- and finishing-
                                                times, the task will be scheduled between them if set to `True`.
                                                Otherwise, it will be appended at the end.
                                                Performing a left shift is never a downside in therms of the makespan.

                                                如果指定的任务在' DisjunctiveGraphJssEnv.;step ' -method可以放置在其他两个任务之间而不改变它们的开始和结束时间，如果设置为' True '，任务将在它们之间调度。
                                                否则，它将被附加在末尾。从最大完工时间的角度来看，执行左移从来都不是一个缺点。

        :param c_map:                           the name of a matplotlib colormap for visualization.
                                                Default is `rainbow`.

        :param dummy_task_color:                the color that shall be used for the dummy tasks (source and sink task),
                                                introduced in the graph representation.
                                                Can be any string that is supported by `networkx`.

        :param default_visualisations:          the visualizations that will be shown by default when calling `render`
                                                Can be any subset of
                                                ["gantt_window", "gantt_console", "graph_window", "graph_console"]
                                                as a list of strings.

                                                    Note:
                                                    "gantt_window" is computationally expensive operation.

        :param visualizer_kwargs:               additional keyword arguments for
                                                `jss_graph_env.DisjunctiveGraphJspVisualizer`

        :param verbose:                         0 = no information printed console,
                                                1 = 'important' printed to console,
                                                2 = all information printed to console,
        """
        # Note: None-fields will be populated in the 'load_instance' method  没有字段将在'load_instance'方法中填充，都是传进来的？
        # load_instance方法中的自定义变量
        self.size = None
        self.n_jobs = None
        self.n_machines = None
        self.total_tasks_without_dummies = None
        self.total_tasks = None
        self.src_task = None
        self.sink_task = None
        self.longest_processing_time = None
        self.observation_space_shape = None
        self.scaling_divisor = None
        self.machine_colors = None
        self.G = None
        self.machine_routes = None

        # 共有三个地方： 初始化init建立为None，load_instance最初的值为0，step结束记录当前的值，reward函数里用来计算
        self.makespan_previous_step = None
        self.total_e1_previous_step = None
        # self.e2_previous_step = None
        self.trans_t_previous_step = None
        self.idle_t_previous_step = None

        """
        1、wrk 加上self可以作为全局变量，__init__初始化中会用一次
        2、只要有有向边，说明生产的时候是都会走一遍的，所以有有向边，就有传输t，就要累加！
        3、add_edge增加新边的时候，最开始的平行边的时候：加一个边，加一个次数，累加下时间
        """
        # 记得清0
        # 全局变量，记录某些随step变化的值
        self.energy_transport = [0, 0]  # 用来记录运输设备的能耗，这里我直接累加运输设备的总的时间！(运了几次，累加时间)：同一个设备上边的不算次数！！！
        self.exist_list = []  #用来记录已经查到过的平行边的传输时间t,防止多查

        """
        记录当前 ”至今为止“ 的指标变量：
        1、makespan会专门的传入到self.get_reward中，不用初始变量
        2、上述self.energy_transport和self.exist_list已经废弃不用
        3、加工能耗：self.total_e1_this_step
        4、空闲时间：self.idle_t_this_step
        5、运输时间：self.trans_t_this_step
        """
        self.total_e1_this_step = 0 #用来记录至今为止选了的设备的能耗和
        self.idle_t_this_step = 0 #用来记录当前的idle时间总和（machine_route里边已经分配的m的空闲时间）
        self.trans_t_this_step = 0 #用来记录至今为止的运输transT时间

        self.reward_list = [0, 0, 0, 0, 0]  #对应记录当前的4个reward+总cost

        # 共有2次：init里边初始化为0，reward函数中进行累加
        self.reward_t = 0 # 传输时间的累计误差
        self.reward_e1 = 0 # 加工能耗的累计误差
        self.reward_e2 = 0 # 运输能耗的累计误差
        self.reward_idle_t = 0 # 空闲时间的累计误差

        # 记录已经被选择的task，记得清0
        self.selected_action = []
        # 记录每一步选择action后新增的空闲时间idle_t
        self.it_s = []  # 后边在load_instance中变为全0列表

        # reward function settings 加上自己的reward_function == 'wrk'
        if reward_function not in ['nasuta', 'zhang', 'graph-tassel', 'samsonov', 'zero', 'custom', 'wrk']:        #判断反馈函数有无
            raise ValueError(f"only 'nasuta', 'zhang', 'graph-tassel', 'samsonov', 'zero', 'custom' "
                             f"are valid arguments for 'reward_function'. {reward_function} is not.")
        if reward_function == 'custom' and custom_reward_function is None:
            raise ValueError(f"if 'reward_function' is 'custom', 'custom_reward_function' must be specified.")

        self.reward_function = reward_function       #reward_function='nasuta',  形参，是传进来的变量；付给init函数中的操作
        self.custom_reward_function = custom_reward_function

        # default reward function params  确定self.reward_function_parameters是什么值
        if reward_function_parameters is None:    #形参，是个字典
            if reward_function == 'nasuta':
                self.reward_function_parameters = {
                    'scaling_divisor': 1.0      #scaling_divisor缩放因子，放的什么?我感觉是折扣率
                }
            elif reward_function == 'zhang':
                self.reward_function_parameters = {}
            elif reward_function == 'samsonov':
                self.reward_function_parameters = {
                    'gamma': 1.025,
                    't_opt': None,
                }
            elif reward_function == 'graph-tassel':
                self.reward_function_parameters = {
                }
            elif reward_function == 'zero':
                self.reward_function_parameters = {}
            elif reward_function == 'custom':
                self.reward_function_parameters = {}
            elif reward_function == 'wrk':              #增加自己的reward_function == 'wrk':
                self.reward_function_parameters = {
                    # 放在外边定义了，这里先注释！
                    # 'scaling_divisor': 20.0  # scaling_divisor缩放因子
                }
            else:
                raise ValueError('something went wrong. This error should not be called.')
        else:
            self.reward_function_parameters = reward_function_parameters

        # observation settings   从形参复制到类中的变量，观测空间
        self.normalize_observation_space = normalize_observation_space  #观测空间是否归一化
        self.flat_observation_space = flat_observation_space #观测空间是否扁平化
        self.dtype = dtype  #观测空间数据类型

        # action setting   从形参复制到类中的变量，动作空间
        self.perform_left_shift_if_possible = perform_left_shift_if_possible  #是否插空，在两个节点之间：不影响开始和结束时间
        if action_mode not in ['task', 'job']:
            raise ValueError(f"only 'task' and 'job' are valid arguments for 'action_mode'. {action_mode} is not.")
        self.action_mode = action_mode    #按照zhang的论文，动作就是每一个task，每次选一个o来在schedule表上边分配，从而更新析取图

        if env_transform not in [None, 'mask']:   #env_transform 环境转移？是析取图的方向的改变？没有给出定义！！！
            raise ValueError(f"only `None` and 'mask' are valid arguments for 'action_mode'. {action_mode} is not.")
        self.env_transform = env_transform

        # rendering settings 从形参复制到类中的变量，渲染
        self.c_map = c_map    # matplotlib 画图的配色
        if default_visualisations is None:  #default_visualisations 要显示的东西是什么：甘特控制台 + 甘特窗口 + 图控制台 + 图控制窗口（我run之后console和弹出的界面，4个）
            self.default_visualisations = ["gantt_console", "gantt_window", "graph_console", "graph_window"]
        else:
            self.default_visualisations = default_visualisations
        if visualizer_kwargs is None:  #额外可视化参数
            visualizer_kwargs = {}
        self.visualizer = DisjunctiveGraphJspVisualizer(**visualizer_kwargs)

        # values for dummy tasks nedded for the graph structure  画图所需的虚拟节点，source和sink的节点。要改成needed吧，hhh
        self.dummy_task_machine = -1  #设备：？
        self.dummy_task_job = -1  #job：？
        self.dummy_task_color = dummy_task_color  #颜色

        self.verbose = verbose  #“啰嗦”，是否显示信息

        """
        1、边的权重：
        2、传输时间t的矩阵 = 传入的形参ability_tr_mm
        3、传入的实例：选择的m + 对应的处理时间
        
        不用清0，因为每次都会初始化这个环境env
        m + t = self.jsp_instance # 2 * j * m
        e1 = self.initial_energy # j * m
        e2/transT =  self.edge_initial_weight  # m * m
        """
        self.edge_initial_weight = ability_tr_mm  # 边的权重的信息，存到self.edge_initial_weight这个变量中，以后这个类中的所有方法都可以直接调用！！！！！！！！！！！！！！！！！！！！！！！！！！
        self.initial_energy = ability_e_tm # m对应的功率，用e表示
        self.jsp_instance = jps_instance # jps_instance是实例化Class的时候传进来的

        if jps_instance is not None:
            self.load_instance(jsp_instance=jps_instance)   #jsp_instance是load_instance方法的形参，jps_instance是init的形参，我们传进来的np矩阵，job和processing time
            # self.load_machine_instance(jsp_instance=jps_instance)   #jsp_instance是load_instance方法的形参，jps_instance是init的形参，我们传进来的np矩阵，job和processing time

    ''' eg.
    jsp = np.array([   #array创建时需要几个维度要用[ ]括起来
            [
                [0, 1, 2, 3],  # job 0
                [0, 2, 1, 3],  # job 1
                [1, 3, 0, 2]   # job 2
            ],
            [
                [11, 3, 3, 12],  # task durations of job 0
                [5, 16, 7, 4],  # task durations of job 1
                [4, 1, 8, 5]   # task durations of job 2
            ]

        ])
    '''
    # 建立Disjunctive Graph里边的节点和边
    def load_instance(self, jsp_instance: np.ndarray, *, reward_function_parameters: Dict = None) -> None:    #python里边同样名字变量不会报错，果然方便了好多！
        """
        This loads a jsp instance, sets up the corresponding graph and sets the attributes accordingly.
        这将加载一个jsp实例，设置相应的图形并相应地设置属性。
        加载实例，根据jsp实例，确定动作和观测空间的基本属性，

        :param jsp_instance:                a jsp instance as numpy array
        :param reward_function_parameters:  if specified, the reward functions params will be updated.  若具体，会更新reward参数；字典，缩放因子不知道是啥？？？

        :return:                            None
        """
        _, n_jobs, n_machines = jsp_instance.shape   #获取几个job，几个machine，简单理解几行几列（不全对，算的是维度）,不关心有有几个大维度，我又新加上了边权重（设备与设备之间的传输时间）
        self.size = (n_jobs, n_machines)   #自定义size = 上步获取到的信息
        self.n_jobs = n_jobs  #job个数
        self.n_machines = n_machines  #machine个数
        self.total_tasks_without_dummies = n_jobs * n_machines   #不算source和sink，有几个点  12个
        self.total_tasks = n_jobs * n_machines + 2  # 2 dummy tasks: start, finish  #全图几个点，算上source和sink
        self.src_task = 0   #source节点的编号
        self.sink_task = self.total_tasks - 1  #sink节点的编号

        self.longest_processing_time = jsp_instance[1].max()   #找第二个矩阵（处理时间）中的最大值

        self.it_s = [0] * self.total_tasks_without_dummies  # init之前没有self.total_tasks_without_dummies变量；init之后，马上把这个初始化

        # print("ssssssssssssssssssssssssssssss", self.edge_initial_weight)

        if self.action_mode == 'task':    #按照子任务来走action
            self.action_space = gym.spaces.Discrete(self.total_tasks_without_dummies)   #创建一个离散的n维空间：动作空间有多大（子任务总个数），是离散的，12个节点
        else:  # action mode 'job'
            self.action_space = gym.spaces.Discrete(self.n_jobs)

        # 生成状态空间的大小
        if self.normalize_observation_space:    #观测空间space的设定，是否正规范化（归一化，方便学习）：shape理解成维度，12行方括号，每个方括号内17个元素
            self.observation_space_shape = (self.total_tasks_without_dummies,      #归一化，eg (12,12+4+1)  why加上设备数?
                                            self.total_tasks_without_dummies + self.n_machines + 1)
        else:
            self.observation_space_shape = (self.total_tasks_without_dummies, self.total_tasks_without_dummies + 2)  #不归一，eg (12,12+2)，2：开始和结束？  why?

        if self.flat_observation_space:  #观测空间扁平化
            a, b = self.observation_space_shape  #维度读取：行+元素
            self.observation_space_shape = (a * b,) #一维的，所以shape维度是相乘，平铺开

        #在自定义观测空间：上边定义了观测空间的维度，这里定义数据范围
        if self.env_transform is None:    #环境转移标志，是一个str字符串格式 ：默认是None
            self.observation_space = gym.spaces.Box(   #连续，观测空间
                low=0.0,
                high=1.0 if self.normalize_observation_space else jsp_instance.max(),  #这里表示我需要状态空间的归一化，低0高1
                shape=self.observation_space_shape,  #维度
                dtype=self.dtype  #数据类型
            )
        elif self.env_transform == 'mask':
            self.observation_space = gym.spaces.Dict({   #观测空间变成字典了：动作标记（low0，high1，12维度，int）+观测（low0，high1，（a，b）的二位维度，数据类型）
                "action_mask": gym.spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.int32),  #eg，action_space = 12个节点
                "observations": gym.spaces.Box(
                    low=0.0,
                    high=1.0 if self.normalize_observation_space else jsp_instance.max(),
                    shape=self.observation_space_shape,
                    dtype=self.dtype)
            })
        else:
            raise NotImplementedError(f"'{self.env_transform}' is not supported.")

        # 下边都是画图相关的
        # generate colors for machines
        c_map = plt.cm.get_cmap(self.c_map)  # select the desired cmap   返回c_map对象，将0-1数值映射成颜色
        arr = np.linspace(0, 1, n_machines, dtype=self.dtype)  # create a list with numbers from 0 to 1 with n items 这个数组的作用是为每个机器分配一个数值，这个数值的范围是0到1之间的连续值
        self.machine_colors = {m_id: c_map(val) for m_id, val in enumerate(arr)}   #使用字典推导式为每个机器分配一个颜色。遍历`arr`数组中的每个元素，将其作为参数传递给颜色

        # jsp representation as directed graph
        self.G = nx.DiGraph()  #G是一个有向图
        # 'routes' of the machines. indicates in which order a machine processes tasks   机器的“路线”。指示机器处理任务的顺序
        self.machine_routes = {m_id: np.array([], dtype=int) for m_id in range(n_machines)}  #创建字典，给每个设备记录其最终的工序路径

        """
        更具体地说，这一行代码创建了一个名为self.machine_routes的实例变量，
        它是一个Python字典，有n_machines个键，每个键都是数字0到n_machines-1，对应着一个机器的ID。
        对于每个键，其对应的值是一个空的1维numpy数组，其中数据类型为整型（dtype=int）。
        这个数组最初为空，但随着算法的执行，它会用于存储表示每个机器工作路径的整数数组。
        
        """

        #
        # setting up the graph
        # G是一个有向图
        #

        # src node  源节点的属性
        self.G.add_node(
            self.src_task,                    #源节点
            pos=(-2, int(-n_jobs * 0.5)),     #节点在绘图中的位置，这里的pos属性是一个元组，表示节点的x和y坐标
            duration=0,                        #节点处理时间
            machine=self.dummy_task_machine,   #源节点机器id：-1
            scheduled=True,                    # 是否被调度：初始肯定被调度了
            color=self.dummy_task_color,       #源节点颜色
            job=self.dummy_task_job,           #源节点job的id：-1
            start_time=0,
            finish_time=0
        )

        """
        jsp2 = np.array([   #array创建时需要几个维度要用[ ]括起来，下边的shape=2,3,4
            [
                [0, 1],  # job 0
                [0, 1],  # job 1
            ],
            [
                [1, 2],  # task durations of job 0
                [3, 4],  # task durations of job 1
            ]
        
        ])
        
        这段代码是为了将作业车间调度问题实例转化为一张表示任务和机器之间依赖关系的有向图。

        首先，它通过迭代jsp_instance的行和列完成了以下操作：
        
        1. 创建了一个唯一的task_id，表示要添加到图中的每个任务的ID。
        2. 确定了每个任务所在的机器ID，并将其存储在变量m_id中。
        3. 确定了每个任务所需的处理时间，并将其存储在变量dur中。
        
        接下来，对于每个任务，该代码向有向图G中添加一个节点，该节点包含以下属性：
        
        1. task_id：任务的唯一标识符。
        2. pos：任务在绘图中的位置，这里表示x和y坐标，其中x为j，y为-i。
        3. color：任务所在的机器的颜色，从预定义的colors列表中选择。
        4. duration：任务所需的处理时间。
        5. scheduled：任务是否已经被调度，这里设置为False。
        6. machine：任务所在的机器的ID。
        7. job：任务所属的作业的ID。
        8. start_time：任务的开始时间，初始设置为None。
        9. finish_time：任务的完成时间，初始设置为None。
        
        然后，该代码根据任务的顺序，向有向图中添加从任务1到任务2的有向边。当该任务是作业中的第一项任务时，会由源任务节点连接到该任务。当该任务是作业中的最后一个任务时，该任务会连接到上一任务，否则它连接到上一个任务并表示其成功的完成。依据这些边的权重可以计算得到调度问题的最终时间。
        
        简而言之，此代码块创建了一个有向图，其中节点代表作业车间调度问题中的任务，边表示任务之间的限制关系。
        
        
        注意：
        1、边缘的权重都是由其出发的那个节点的duration时间决定的，我又新加了运输时间（按照m到m的转移运输t矩阵查表）
        2、我更新了每一个节点的st开始时间 = 上一个节点的结束时间 + 上一个节点m到这个节点m的运输t（查表），上述和的最大的那一个（因为有先后的顺序，只能是最大的）
        3、st更新之后，节点结束时间ft = st+持续时间duration，也会不断更新，是个迭代的过程（ft和运输t无关了，运输是要给下一个节点的，你还没确定，不用加到ft上的）
        4、_schdule_task/_insert_index_0/append_at_end 是我主要更新的函数，加上了新的约束（运输t，即边权重），这样子的话，reward也会改变，即是我建模的cost
        
        5、增加了边权重，导致输出的state发生了改变（析取图状态就是边权重），所以归一化不能用设备最大的持续时间来除，会有》1产生（现在应该用调度完成之后的最大的边权重，也只能是跑完一个episode之后才能归一，训练之前）
        6、选择了设备，也新增了能耗的reward，但是很直接，是常量，而且选定m之后就是固定的！不对，还有运输设备的能耗啊！！！！！！
        7、能耗也需要加入到state的矩阵中，最后一列！！！
        8、gantt是按照st和ft画的，所以不会那么紧密，除非是同一个设备之间
        """
        # add nodes in grid format
        task_id = 0
        machine_order = jsp_instance[0].astype(int)    #m的id最好都是int类型，防止出错！每个job中：设备的已知工序,分好了设备，都是从0开始的
        processing_times = jsp_instance[1]

        for i in range(n_jobs):
            for j in range(n_machines):
                task_id += 1  # start from task id 1, 0 is dummy starting task
                m_id = machine_order[i, j]  # machine id    遍历job列表中的m的id

                dur = processing_times[i, j]  # duration of the task   遍历对应的处理时间,用来储存到node的“duratuin”键值，然后传递给edge作为权重；同时也是开始和结束时间的判断条件（改这里！）

                self.G.add_node(    # 因为是循环，所以会每一个task都建立好了，从1开始的，0是源节点
                    task_id,
                    pos=(j, -i),
                    color=self.machine_colors[m_id],
                    duration=dur,         # 这里是我们设定的node的处理时间啊，他画图画在了有向边上！self.G.nodes的信息在这里被读取了！
                    scheduled=False,
                    machine=m_id,
                    job=i,
                    start_time=None,    # start_time理论上应该一直在更新，根据不同的调度顺序，运输时间也不一样啊
                    finish_time=None
                )

                # 添加边的权重：加上我的传输时间权重
                # 都是添加的上一个节点的权重，理解为平行的边
                if j == 0:  # first task in a job    job列表中的第一个task的index，就直接添加边了啊！！！
                    self.G.add_edge(
                        self.src_task, task_id,      #源节点和当前的job的第一个的节点连接：有向的,[self.src_task]['duration']=0
                        job_edge=True,
                        weight=self.G.nodes[self.src_task]['duration'],   #从图 G 中读取节点 src_task 的 'duration' 属性的值，并将其赋值给变量 weight
                        nweight=-self.G.nodes[self.src_task]['duration']  # duration = 0 ，src源节点
                    )
                elif j == n_machines - 1:  # last task of a job   job列表中的最后一个task的index，就直接添加边了啊！！！
                    # 判断这两个节点之间的传输时间是否有效：是否是同一个job内；不是返回0
                    # transport_t, self.energy_transport, self.exist_list = make_sure_transportT_work(self.G, (task_id - 1), task_id, self.edge_initial_weight, self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)
                    transport_t = find_transportT(self.G, (task_id - 1), task_id, self.edge_initial_weight)
                    # print("load instance末位 self.energy_transport",self.energy_transport,j,self.exist_list)

                    self.G.add_edge(
                        task_id - 1, task_id,   #job的最后一个节点和其上一个节点进行连接：有向，上一个---最后一个，填的是上一个节点的duration，确实没错
                        job_edge=True,
                        # weight=self.G.nodes[task_id - 1]['duration'],
                        # nweight=-self.G.nodes[task_id - 1]['duration']
                        # wrk 新的平行边的权重!!
                        weight=self.G.nodes[task_id - 1]['duration'] + transport_t,
                        nweight=-(self.G.nodes[task_id - 1]['duration'] + transport_t)
                    )
                else:
                    # 判断这两个节点之间的传输时间是否有效：是否是同一个job内；不是返回0
                    # transport_t, self.energy_transport, self.exist_list = make_sure_transportT_work(self.G, (task_id - 1), task_id, self.edge_initial_weight, self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)
                    transport_t = find_transportT(self.G, (task_id - 1), task_id, self.edge_initial_weight)
                    # print("load instance中 self.energy_transport", self.energy_transport, j,self.exist_list)

                    self.G.add_edge(    #其他情况：上一个节点连接当前节点，有向，因为按照1234.。。来的，相当于按照子任务工序约束直接建好了一个初始有向图
                        task_id - 1, task_id,
                        job_edge=True,
                        # weight=self.G.nodes[task_id - 1]['duration'],
                        # nweight=-self.G.nodes[task_id - 1]['duration']
                        # wrk 新的平行边的权重!!
                        weight=self.G.nodes[task_id - 1]['duration'] + transport_t,
                        nweight=-(self.G.nodes[task_id - 1]['duration'] + transport_t)
                    )
        # add sink task at the end to avoid permutation in the adj matrix.   在最后添加下沉任务，避免在形容词矩阵中排列。
        # the rows and cols correspond to the order the nodes were added not the index/value of the node. 行和cols对应于节点添加的顺序，而不是节点的索引/值。
        self.G.add_node(   #添加最后的节点
            self.sink_task,
            pos=(n_machines + 1, int(-n_jobs * 0.5)),
            color=self.dummy_task_color,
            duration=0,
            machine=self.dummy_task_machine,  # m_id =-1
            job=self.dummy_task_job,
            scheduled=True,
            start_time=None,
            finish_time=None
        )
        # add edges from last tasks in job to sink  添加每个job的最后一个task的边到结束节点
        for task_id in range(n_machines, self.total_tasks, n_machines):   #起始任务编号分别为 n_machines, 2*n_machines, 3*n_machine, ... 直至 self.total_tasks
            self.G.add_edge(
                task_id, self.sink_task,
                job_edge=True,
                weight=self.G.nodes[task_id]['duration']    #最后一个task到sink节点的边的权重就是，最后一个子任务的处理时间（sink节点是虚拟节点啊！）
            )

        """
        接收一个有向无环图 self.G 作为输入，返回图的最长路径长度。其实现基于动态规划，时间复杂度为 $O(V+E)$，其中 $V$ 和 $E$ 分别为图中节点数和边数。
        代码将计算得到的最长路径长度赋值给变量 initial_makespan，这相当于以 DAG 执行所有任务的最小总时间。
        因此，如果后续的任务调度算法可以找到一个总时间更短的执行方案，就可以将其作为更优的解。如果算法不能改进初始 makespan，则初始 makespan 将作为一个上界，算法的目标是找到一个解，使其 makespan 越接近初始 makespan 越好。
        """
        initial_makespan = nx.dag_longest_path_length(self.G)   #自带函数，(基于动态规划实现的！最初的，目的是找个比他更短，获证更接近的！！不会管node上的权重，只看边权重)
        self.makespan_previous_step = initial_makespan   # 这里表示建好的只有平行边的DG图中的makespan

        """
        在step运行之前：
        load_instance里边进行：第一个reward初始变量的值
        都设定为0：因为还没调度怎么能有reward呢
        """
        if self.reward_function == 'wrk':
            initial_makespan = 0
            self.makespan_previous_step = initial_makespan
            self.total_e1_previous_step = 0
            # self.e2_previous_step = 0
            self.trans_t_previous_step = 0
            self.idle_t_previous_step = 0

        """
        这里记录了reward的缩放因子，不能删掉！！！
        """
        if reward_function_parameters is not None:
            if self.verbose > 1:
                log.info(f"updating reward_function_parameters from '{self.reward_function_parameters}' "
                         f"to '{reward_function_parameters}'")
            self.reward_function_parameters = reward_function_parameters  # reward的缩放因子的赋值！！！！

    def step(self, action: int) -> (np.ndarray, float, bool, dict):   #环境中的action具体怎么走？-> (np.ndarray, float, bool, dict)描述函数返回的数据类型。
        """
        perform an action on the environment. Not valid actions will have no effect.

        :param action: an action：是task的index值
        :return: state, reward, done-flag, info-dict    返回状态+奖励+完成与否+信息字典
        """

        # 记录被选择的action = taskID - 1
        self.selected_action.append(action)

        info = {                 #是一个字典
            'action': action
        }

        """
        # add arc in graph    选完一个task之后，会有有向边的产生！
        # 调用_schedule_task一次，开始调度
        """
        if self.action_mode == 'task':
            task_id = action + 1  #task_id代表了当前的子任务，是按照123按顺序来的；action是从0开始的，这里是将action+1转变成DG图上的节点的id！！！！！！！

            if self.verbose > 1:
                log.info(f"handling action={action} (Task {task_id})")
            info = {
                **info,
                **self._schedule_task(task_id=task_id)    #在这里，调用_schedule_task方法：return：更新边的方向 + 更新当前节点的开始和结束时间！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            }
        # 下边是git提供的直接选择job的方式：和我思路一样，job转task的id，然后选完job不能选了之后报错！！！！！！！！！！！！
        else:  # case for self.action_mode == 'job'
            # 返回值为一个bool数组，它的长度与self.action_space中的元素数量相等，其中索引为True的元素表示相关操作是有效的，False则表示该操作是无效的。
            task_mask = self.valid_action_mask(action_mode='task')
            # 任务操作掩码task_mask划分为元素数量相等的n_jobs个子掩码,然后通过索引action选择与当前作业相关的子掩码
            job_mask = np.array_split(task_mask, self.n_jobs)[action]

            if True not in job_mask:  # 说明当前job选完了，不能选了，然后呢？有给很小的R吗？没有，下次又选了这个job，继续报错？没啥用啊！
                if self.verbose > 0:  # 一个job里边全是False，说明不能选。然后就给log，会一直不调度
                    log.info(f"job {action} is already completely scheduled. Ignoring it.")
                info["valid_action"] = False  # False表明被调度了，该task不是有效的！
            else:
                # np.argmax(job_mask)会返回给定掩码job_mask中最大元素的索引
                task_id = 1 + action * self.n_machines + np.argmax(job_mask)  # [F,F,T,T],返回索引2，这个job的第三个task可选
                if self.verbose > 1:
                    log.info(f"handling job={action} (Task {task_id})")
                info = {
                    **info,
                    **self._schedule_task(task_id=task_id)
                }
        """
        array_split
        将一个阵列分裂为多个子阵列。
        >>> import numpy as np
        >>> np.array_split(np.arange(8.0), 3)
        [array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.])]
        >>> np.array_split(np.arange(9), 4)
        [array([0, 1, 2]), array([3, 4]), array([5, 6]), array([7, 8])]
        请参考“split”文档。这些函数之间的唯一区别是' ' array_split ' ' '允许' ' indices_or_sections ' '是一个整数，
        不相等地分割轴。对于一个长度为l的数组，它应该被分割成n个部分，它返回l % n个子数组，大小为l//n + 1，其余大小为l//n。
        """

        """
        具体来说，self.machine_routes 是一个 Python 字典，其键是机器的 ID，其值是列表类型，包含了该机器的任务处理路径。列表中的每个元素表示在该机器上处理的任务 ID。因此，self.machine_routes.items() 返回一个对 (机器 ID, 任务处理路径) 键值对进行迭代的迭代器。
        列表推导式 [len(route) for m_id, route in self.machine_routes.items()] 遍历每个机器的路径列表，并将每个路径的长度（即路径包含的任务数）提取出来，形成新的列表。min() 函数接受这个列表作为输入，并返回其中的最小值，即所有路径长度中的最小值。 
        将这个最小路径长度赋值给变量 min_length。因此，这行代码计算出了所有机器路径集合中最短路径的长度，可以用于后续的任务调度算法中。
        """
        # check if done （not use by wrk）!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # min_length = min([len(route) for m_id, route in self.machine_routes.items()])  #machine_routes记录了每个m的子任务顺序（分配好的），route是子任务列表，求所有m的最短的列表的长度

        """
        有BUG：这里默认所有的任一m在每一job中都出现：machine_route里边的最短路径 == n_job数量
        如果任一m不是在每一job中都出现的话，那么done的判断条件是有问题的；machine_route里边的最短路径，永远不会等于n_job数量，本身就少！（改成：machine_route的所有m的路径之和 == 总的任务数量，这个倒是可以的）
        """
        # check if done by wrk!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        total_length = sum([len(route) for m_id, route in self.machine_routes.items()])  # machine_routes记录了每个m的子任务顺序（分配好的），route是子任务列表，求所有m的列表的长度的总和
        # print("已选m的工序集合的total_length: ", total_length)
        done = total_length == self.total_tasks_without_dummies   # 所有m的被调度的工序列表的长度和 == 总共的工序任务数！分配完成了！（也可能会有bug，但是现阶段是这样子的）

        # 这个done是所有action都搞定的时候，true；还是每一个action结束了，都会变成true？？？？（直观感受是前者，一个episod结束，全部分配完）
        # done = min_length == self.n_jobs   # done  判断条件竟然是job的个数，判断是否完成当前job和machine的schedule甘特图！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！(有bug，我先注释掉了！！！)

        """
        更新下reward相关的变量
        1、 每一步的step之后都会反馈一个最长路径，因为有图有边权重，所以会自动返回最长路径
        2、上述已经运行了一次_schedule_task了，更新下当前被选m消耗的能耗
        """
        # makespan相关的
        # makespan = nx.dag_longest_path_length(self.G)    #计算最大完工时间！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！这里输出
        # print("当前step的：已分配的m的makespan：", makespan)

        """
        一、计算至今为止所选action中的最大的finish time：
        这种计算最长路径的方式太稀疏！
        改成：每一步都计算已分配的m的finish time的最大值！
        """
        makespan = self.max_finish_time_in_machineRoute(self.G, self.machine_routes)
        """print("New：每一步返回的最大的结束时间", makespan)"""

        # 加工能耗相关的
        """当前动作转成task_id"""
        task_id = action + 1 # 当前的任务编号
        # 暂时：self.initial_energy已经等于是p*t了
        e1_arr = self.jsp_instance[1] * self.initial_energy  # 对应设备的持续时间 * 对应的能耗， 用时从里边取
        # print("sssssssssssssssssss",e1_arr)

        row_index = task_id // self.n_machines #取整
        col_index = task_id % self.n_machines   #取余
        if col_index == 0:  #防止会超限
            row_index -= 1
            col_index = 0  #按列倒着取，下边有-1，所以这里写0,；-1是最后一列

        # e1_current = e1_arr[row_index][(col_index - 1)]  #选择了当前设备产生的能耗
        # self.total_e1_this_step = self.total_e1_this_step + e1_current  # 不断累加至今为止的选择了m的能耗和

        """
        二、计算当前action对应的p*t，然后累加
        1、传入的：self.initial_energy已经等于是p*t了
        2、将task_id转成对应j*m矩阵的行列索引inddex
        3、对照index获得当前task_id的具体的加工能耗p*t
        """
        # 修改为直接输出e1，不乘以t了
        e1_current = self.initial_energy[row_index][(col_index - 1)]  # 选择了当前设备产生的能耗
        self.total_e1_this_step = self.total_e1_this_step + e1_current  # 不断累加至今为止的选择了m的能耗和

        # print("当前step的：已分配的m的所有能耗之和：", self.total_e1_this_step)

        # 运输时间已经内嵌在_schedule_task函数了，不用在step里边更新了
        # print("当前step的：已分配的m的所有运输时间之和（reward要乘以运输m的功率）：", self.energy_transport[1])

        """
        三、获取新选择action之后的总空闲时间t：
        1、就是从self.machine_routes中求和空白
        2、可能会包含：运输时间t + 某个task的dur加工时间
        """
        self.idle_t_this_step = calculate_idle_t_for_each_machine(self.G, self.machine_routes) #基于已选的m的空闲时间总和，是累加的
        # print("当前step的：已分配的m的所有空闲时间之和：", self.idle_t_this_step)
        """print("计算空闲时间时的machine_route = ", self.machine_routes)"""

        """
        四、计算当前action选择之后“有效的”运输时间：
        1、首位的task都没有运输时间
        2、新增被选task_id之后，其同job的上一个task的对应的m到当前m的运输时间变得有效
        3、累加有效的运输时间作为当前this_step的state
        
        注意：实际上的运输时间t都是固有的平行边才有，所以我们要做的就是找当前task，及其同job的上一个task，查表得transT
        self.edge_initial_weight = m * m 运输时间能力t
        action: an action：是task的index值（从0开始）
        machien_routes = {0: array([], dtype=int64), 1: array([3, 4, 1, 2])}  字典形式：存有同一个m中的taskid的加工顺序！（从1开始的吧？）
        """
        if action % self.n_machines == 0: # action从0开始，表明是每个job的首位，不会有运输
            new_avail_transT = 0
        else: # 既然选到了中间位置的task，其前置task必定被选，不然不满足先后工序！！！
            new_avail_transT = find_transportT(self.G, action, (action+1), self.edge_initial_weight) # action+1=当前的task_id，所以action=上一个的task_id
        self.trans_t_this_step += new_avail_transT

        """
        先更新当前的idle的数值，然后再get_reward计算prev-curr
        1、获取下一时刻的状态！！！
        2、更新对应的reward！！！！
        """
        state, ft_s, it_s, adj_wrk, tasks_fea = self._state_array()  # 状态获取的函数，看看怎么定义的？怎么和图网络结合呢？？？？？？？？？？？？？？？？？？？？？？
        reward, r_t, r_idle, r_e1, r_e2 = self.get_reward(  # reward获取的函数，可以把自己约束加进去！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            state=state,
            done=done,  # 传进去这些形参，作为输入
            info=info,
            makespan_this_step=makespan  # 前边刚计算了当前步骤下的makespan
        )

        """
        这里是记录上一次的reward相关的变量
        最后一次，也会在done之前记录当前的结果；done之后会被reset，reset有load_instance，里边有重新把previous_step=0的操作。（不必担心）
        
        注意：只要不reset，done之后：那么这些prev值代表的就是当前全局的没有权重、没有累加、不用相减的真实的cost！！！！！！！！！！！
        """
        self.makespan_previous_step = makespan  # 上次的makespan
        self.total_e1_previous_step = self.total_e1_this_step  # 上次的加工能耗之和（已分配设备的）
        # self.e2_previous_step = self.energy_transport[1]  # 至今为止的运输时间t （暂时没有乘以运输设备的e）
        self.trans_t_previous_step = self.trans_t_this_step  # 至今为止的运输时间t （暂时没有乘以运输设备的e）
        self.idle_t_previous_step = self.idle_t_this_step  # 上次记录当前的idle时间（machine_route里边已经分配的m的空闲时间）

        # 如果结束了，在最后进行变量的清0
        if done:
            try:
                # by construction a cycle should never happen    不能有毕环的建立啊，这里是检验看看网络有无错误
                # add cycle check just to be sure
                cycle = nx.find_cycle(self.G, source=self.src_task)
                log.critical(f"CYCLE DETECTED cycle: {cycle}")
                raise RuntimeError(f"CYCLE DETECTED cycle: {cycle}")
            except nx.exception.NetworkXNoCycle:
                pass
            info["makespan"] = makespan     #给info这个字典添加元素：最大完工时间
            info["gantt_df"] = self.network_as_dataframe()
            if self.verbose > 0:
                log.info(f"makespan: {makespan}")   #这里是输出makespan的结果！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

            # 全局变量，记录某些随step变化的值
            self.energy_transport = [0, 0]  # 用来记录运输设备的能耗，这里我直接累加运输设备的总的时间！(运了几次，累加时间)：同一个设备上边的不算次数！！！
            self.exist_list = []  # 用来记录已经查到过的平行边的传输时间t,防止多查
            self.total_e1_this_step = 0  # 用来记录至今为止选了的设备的能耗和
            self.idle_t_this_step = 0  # 用来记录当前的idle时间（machine_route里边已经分配的m的空闲时间）
            self.trans_t_this_step = 0 # 记录至今为止的运输时间，清0

            self.reward_list = [0, 0, 0, 0, 0]  # 对应记录当前的4个reward+总cost

            # 共有2次：init里边初始化为0，reward函数中进行累加；这里清0
            self.reward_t = 0  # 传输时间的累计误差
            self.reward_e1 = 0  # 加工能耗的累计误差
            self.reward_e2 = 0  # 运输能耗的累计误差
            self.reward_idle_t = 0  # 空闲时间的累计误差

            # 记录被选择的action
            self.selected_action = []
            self.it_s = [0] * self.total_tasks_without_dummies # 清0之前先存在某个变量，不影响

        return state, reward, done, info, r_t, r_idle, r_e1, r_e2, ft_s, it_s, adj_wrk, tasks_fea     #每一步都会返回（state + reward + done + info），

    # 针对不同论文的reward的定义，返回不同的reward，可以输出看下用的什么reward_function
    # 输入的是：self自己 + 当前状态 + 是否完成 + 额外信息（字典） + 当前的makespan
    # 每一步都返回reward吗？？？？不是的，step每一次的action都会反馈done，没有全部action搞定，done = false
    def get_reward(self, state: np.ndarray, done: bool, info: Dict, makespan_this_step: float):
        info['reward_function'] = self.reward_function   #info的字典，reward_function是一个字符串'nasuta'，然后会 self.reward_function_parameters赋值缩放因子为1，折扣率？
        reward_function_parameters = self.reward_function_parameters  #重新从init里边把他读了过来，放到method的内部变量里，用来逻辑操作

        if self.reward_function == 'nasuta':
            if not done:  # = if （not false） = if true：没全跑完action，一直反馈0；
                return 0.0                  #没有结束析取图（schedule表格没有画完），没有反馈，反馈一直为0
            else:                           #最后一次，跑完了，给一个负的makespan
                return - makespan_this_step / reward_function_parameters['scaling_divisor']    #缩放因子scaling_divisor我记得是1，奖励就是-makespan！！！！！！！！！！！！！！！！！！！！ 没啥牵扯到的开始时间，就无脑计算边权重最大；前提step那里新增了边

        elif self.reward_function == 'zhang':
            return 1.0 * self.makespan_previous_step - makespan_this_step  # 1.0 to convert to float   上次的makespan-这次的makespan（选了新的子任务之后的makes的差值）= 必定负数，累计reward，迭代到最后=常数-makespan（和nasuta一样的）

        elif self.reward_function == 'graph-tassel':
            # this implementation is not equal to the orginal tassel implementation, because in the disjunctive graph
            # approach a timestep does not correspond to a step in time but rather a step in the scheduling process.
            # However this code uses tassels idea to use the machine utilization or the scheduled area in the gant
            # chart.   这里考虑了设备的利用率，是另一种reward
            max_finish_time = 0.0
            total_filled_area = 0.0
            at_least_one_scheduled_node = False
            for m, m_route in self.machine_routes.items():
                if len(m_route):
                    m_ft = self.G.nodes[m_route[-1]]["finish_time"]
                    if m_ft >= max_finish_time:
                        max_finish_time = m_ft
                    m_filled_area = sum([self.G.nodes[task]["duration"] for task in m_route])
                    total_filled_area += m_filled_area
                    at_least_one_scheduled_node = True
                else:
                    pass
            if not at_least_one_scheduled_node:  # needed to avoid division through zero after invalid first action
                return 0.0
            total_gantt_area = max_finish_time * self.n_machines
            machine_utilization = total_filled_area / total_gantt_area  # always between 0 and 1
            return machine_utilization

        elif self.reward_function == 'samsonov':
            if not done:
                return 0.0
            else:
                gamma = reward_function_parameters['gamma']
                if reward_function_parameters['t_opt'] is None:
                    raise ValueError(f"'t_opt' must be provided inside 'reward_function_parameters' for the samsonov "
                                     f"reward function.")
                t_opt = reward_function_parameters['t_opt']
                return 1000 * gamma ** t_opt / gamma ** makespan_this_step

        elif self.reward_function == 'zero':
            return 0.0

        elif self.reward_function == 'custom':
            return self.custom_reward_function(
                state,  # numpy array
                done,  # boolean done flag
                info,  # info dict
                self.G,  # networkX directed graph
                makespan_this_step,  # longest path in G before the current timestep action
                self.makespan_previous_step,  # longest path in G before the current timestep action
                **reward_function_parameters  # custom reward function parameters
            )
        elif self.reward_function == 'wrk':
            return self.wrk_reward_function(done, makespan_this_step)  #返回这个函数所返回的值！！！！！！！！！！！！！！！！！！！！！！（我自定义的reward函数）

    """
    load_instances函数:
        initial_makespan = nx.dag_longest_path_length(self.G)   #这里好像是用的现成的函数来找最长的路径啊！！！！！！！！！！！！！！！(基于动态规划实现的！最初的，目的是找个比他更短，获证更接近的！！不会管node上的权重，只看边权重)
        self.makespan_previous_step = initial_makespan
    
    1、原先定义的最初的makespan是建好的只有平行边的DG图的makespan,不为0；我已经修改了，reward_function = ’wrk‘，最初改为 0
    2、该函数是在step函数最后，_schedule_task_之后的，更新下reward
    """
    def wrk_reward_function(self, done: bool, makespan_this_step: float):

        """
        makespan也有问题：
        1、因为是现有的平行边，所以makespan一直都存在
        2、但是实际上，刚开始调度，都还没有走到sink节点呢，makespan就会有了，而且这一段时间还不会变
        3、所以，干脆最后调度完了，才有makespan吧

        :param done:
        :param makespan_this_step:
        :return:
        """
        # 带self的都是全局变量
        """最大完工时间"""
        r_t = 1.0 * self.makespan_previous_step - makespan_this_step # include传输时间了
        self.reward_t = self.reward_t + r_t    #累加Reward = initial - maskpan_this_step,  类似于直接取负数，值=makespan_this_step
        self.reward_list[0] = self.reward_t   # 记录每一step的时候的状态对应的reward的负值（因为上两行，累加=0）

        # self.total_e1_previous_step和self.total_e1_this_step
        # 加工设备的总能耗：我在step里边更新了，已经是：持续时间 * 对应的功率（用e表示的）
        """加工能耗"""
        r_e1 = 1.0 * self.total_e1_previous_step - self.total_e1_this_step # 这里存的都是当前的分配好设备的能耗之和
        self.reward_e1 = self.reward_e1 + r_e1  #累加能耗，类似于直接取负数，值=total_e1_this_step
        self.reward_list[1] = self.reward_e1

        # self.e2_previous_step写的是e2，实际记录的是上次的运输时间
        # r_e2 = 1.0 * self.e2_previous_step * 1.0 - self.energy_transport[1] * 1.0 #运输设备的能耗 = 运输时间 * 消耗功率
        """运输时间"""
        r_e2 = 1.0 * self.trans_t_previous_step - self.trans_t_this_step # 这里存的都是当前的分配好设备的能耗之和 #运输设备的能耗 = 运输时间 * 消耗功率
        self.reward_e2 = self.reward_e2 + r_e2  #累加，类似于直接取负数，值=energy_transport[1]
        self.reward_list[2] = self.reward_e2

        """空闲时间"""
        r_idle = 1.0 * self.idle_t_previous_step - self.idle_t_this_step
        self.reward_idle_t = self.reward_idle_t + r_idle  #累加，类似于直接取负数，值=idle_t_this_step
        self.reward_list[3] = self.reward_idle_t

        """
        运输时间有点问题：
        1、只要是load_instance就会有运输时间的总和了（因为是平行边啊）
        2、所以不是按照调度了哪些节点，这些节点之间会产生运输时间的！
        3、所以会一开始就是固定值，不会变
        """
        # print("每一个step里边计算当前的各个实时reward（上一次-当前）：makespan: {} 加工能耗：{} 运输能耗：{} 空闲时间：{} ".format(r_t,r_e1,r_e2,r_idle))
        # print("每一个step里边计算当前的累加reward：makespan: {} 加工能耗：{} 运输能耗：{} 空闲时间：{} ".format(self.reward_t, self.reward_e1, self.reward_e2, self.reward_idle_t))

        # w1 = 0.5
        # w2 = 0.2
        # w3 = 0.2
        # w4 = 0.1
        """w1 = 0.5 # makespan   这个结果是挺好的！！！！！
        w4 = 0.5  # 空闲时间"""
        # w2 = 0  # 加工能耗
        # w3 = 0  # 运输能耗
        # w1 = 0.9  # makespan
        w4 = 0.1 # 空闲时间
        w1 = 0.9  # 试一下就是真实的reward，不加权重
        # w4 = configs.weight_ec  # 空闲时间
        # w1 = configs.weight_mk  # 试一下就是真实的reward，不加权重
        w2 = 0
        w3 = 0

        """# 这里返回的就是每一个cost的真实值(负数！)了！用这个，没有done之前，都得是0，可以试试归一化？
        total_cost = w1 * self.reward_t + w2 * (self.reward_e1 +self.reward_e2) + w3 * self.reward_idle_t  # 累加=f(0)-f(t),初始为0，必然为负
        self.reward_list[4] = total_cost"""

        # 每一步返回的是prev-current。反正Gt有一个是累加，我也做了处理
        # step_total_r = w1 * r_t + w2 * r_e1 + w3 * r_e2 + w4 * r_idle  # 每一步输出的rewark = 上一次 - 当前
        # 空闲时间的功率：min=1
        """step_total_r = w1 * r_t  + w4 * r_idle * 1 + 0 + 0  # 每一步输出的rewark = 上一次 - 当前"""
        step_total_r = w1 * r_t + w2 * r_e1 + w3 * r_e2 + w4 * r_idle  # 每一步输出的rewark = 上一次 - 当前

        """
        新思路：
        1、DGenv反馈的直接是4个指标，然后用的是3个权重！
        2、会覆盖上述，不用就还是以前的
        """
        w_mk = configs.weight_mk
        w_ec = configs.weight_ec
        w_tt = configs.weight_transT
        if configs.use_2critic_net:
            step_total_r = w_mk * r_t + w_ec * (r_e1 + r_idle*1) + w_tt * r_e2   # 每一步输出的rewark = 上一次 - 当前

        """w5 = 0.5 
        w6 = 0.5

        # 只关注makespan + idle （因为能耗 + 运输时间 在m选的actor里边定义了）
        step_total_r_new = w5 * r_t + w6 * r_idle  # 每一步输出的rewark = 上一次 - 当前"""

        if not done:
            # return 0.0            # 先每一步不返回,看看最终结果是什么
            # return total_cost / 1
            """
            在循环中会对16step之后的r求和=-真实的cost（不考虑权重和放缩）
            """
            return step_total_r / self.reward_function_parameters['scaling_divisor'], r_t , r_idle, r_e1, r_e2
            # return 0/ self.reward_function_parameters['scaling_divisor'], r_t , r_idle, r_e1, r_e2
            # return step_total_r_new / self.reward_function_parameters['scaling_divisor'], r_t , r_idle
        else:  #
            # return total_cost / 1  # 缩放因子scaling_divisor我记得是1
            return step_total_r / self.reward_function_parameters['scaling_divisor'], r_t , r_idle, r_e1, r_e2  # 缩放因子scaling_divisor我记得是1，累计奖励才是-makespan！！！
            # return total_cost / self.reward_function_parameters['scaling_divisor'], r_t , r_idle, r_e1, r_e2  # 缩放因子scaling_divisor我记得是1，累计奖励才是-makespan！！！
            # return step_total_r_new / self.reward_function_parameters['scaling_divisor'], r_t , r_idle  # 缩放因子scaling_divisor我记得是1，累计奖励才是-makespan！！！

    def max_finish_time_in_machineRoute(self, G, machineRoute):
        max = 0
        for value in machineRoute.values():  # k key + v value
            # print(value)
            for i in range(len(value)):
                if max <= G.nodes[value[i]]['finish_time']:   # 寻找machine_route里边所有点的最大的finish time
                    max = G.nodes[value[i]]['finish_time']
        # print(max)
        return max

    def reset(self):
        """
        resets the environment and returns the initial state.

        重置环境，并返回最初的状态：没有分配任何一个oeration

        :return: initial state as numpy array.
        """
        # remove machine edges/routes  删除设备边和路线（可能是makespan的那个粗线）
        machine_edges = [(from_, to_) for from_, to_, data_dict in self.G.edges(data=True) if not data_dict["job_edge"]]
        self.G.remove_edges_from(machine_edges)

        # reset machine routes dict
        self.machine_routes = {m_id: np.array([]) for m_id in range(self.n_machines)}

        # remove scheduled flags, reset start_time and finish_time
        for i in range(1, self.total_tasks_without_dummies + 1):
            node = self.G.nodes[i]
            node["scheduled"] = False
            node["start_time"] = None,
            node["finish_time"] = None

        """
        重新加载instance
        不重新画点的话，会不能连续运行！！！（血的教训！）
        
        self.energy_transport会在self.load_instance中被执行，然后下一语句有清0
        """
        self.load_instance(jsp_instance=self.jsp_instance)   # 重置环境的时候，这些固有的点还是要画上去！！！！！！！！！

        """
        Env：如果连续运行，这些都需要请0！！！！！！！！！！！！！！！！！！！
        全局变量，记录某些随step变化的值
        """
        self.energy_transport = [0, 0]  # 用来记录运输设备的能耗，这里我直接累加运输设备的总的时间！(运了几次，累加时间)：同一个设备上边的不算次数！！！
        self.exist_list = []  # 用来记录已经查到过的平行边的传输时间t,防止多查
        self.total_e1_this_step = 0  # 用来记录至今为止选了的设备的能耗和
        self.idle_t_this_step = 0  # 用来记录当前的idle时间（machine_route里边已经分配的m的空闲时间）
        self.trans_t_this_step = 0 # 记录至今为止的运输时间t，清0

        self.reward_list = [0, 0, 0, 0, 0]  # 对应记录当前的4个reward+总cost

        # 共有2次：init里边初始化为0，reward函数中进行累加
        self.reward_t = 0  # 传输时间的累计误差
        self.reward_e1 = 0  # 加工能耗的累计误差
        self.reward_e2 = 0  # 运输能耗的累计误差
        self.reward_idle_t = 0  # 空闲时间的累计误差

        """
        self.selected_action不清0，到时候读取node["finish time"/"start time"]都会产生None，初始值
        """
        # 记录已经被选择的task，记得清0
        self.selected_action = []
        # 记录每一步选择action后新增的空闲时间idle_t
        self.it_s = [0] * self.total_tasks_without_dummies # 清0之前先存在某个变量，不影响  # 后边在load_instance中变为全0列表

        return self._state_array() # done时会清0，所以返回值都是0

    # 渲染就是和可视化的操作有点关系了，暂时我用不到，可以直接关闭
    def render(self, mode="human", show: List[str] = None, **render_kwargs) -> Union[None, np.ndarray]:  #将可变数量的参数 或 键值对参数 *args、**kwargs传递给函数，作为形参，作为输入
        """
        renders the enviorment.

        :param mode:            valid options: "human", "rgb_array", "console"   选择渲染的模式

                                "human" (default)

                                    render the visualisation specified in :param show:
                                    If :param show:  is `None` `DisjunctiveGraphJssEnv.default_visualisations` will be
                                    used.

                                "rgb_array"

                                    returns rgb-arrays of the 'window' visualisation specified in
                                    `DisjunctiveGraphJssEnv.default_visualisations`

                                "console"

                                    prints the 'console' visualisations specified in
                                    `DisjunctiveGraphJssEnv.default_visualisations` to the console

        :param show:            subset of the available visualisations   可用可视化的选择，列表数据类型，就是我们能看到的动画环境
                                ["gantt_window", "gantt_console", "graph_window", "graph_console"]
                                as list of strings.

        :param render_kwargs:   additional keword arguments for the
                                `jss_graph_env.DisjunctiveGraphJspVisualizer.render_rgb_array`-method.

        :return:                numpy array if mode="rgb_array" else `None`
        """
        df = None
        colors = None

        if mode not in ["human", "rgb_array", "console"]:
            raise ValueError(f"mode '{mode}' is not defined. allowed modes are: 'human' and 'rgb_array'.")

        if show is None:
            if mode == "rgb_array":
                show = [s for s in self.default_visualisations if "window" in s]
            elif mode == "console":
                show = [s for s in self.default_visualisations if "console" in s]
            else:
                show = self.default_visualisations   # show要显示的是什么就是你之前初始化Env的self.default_visualisations设定值

        if "gantt_console" in show or "gantt_window" in show:
            df = self.network_as_dataframe()
            colors = {f"Machine {m_id}": (r, g, b) for m_id, (r, g, b, a) in self.machine_colors.items()}

        """
        每一步都有render，所以就会在下边进行画图，除非是你不想要有图案输出
        那就注释完下边的代码：应该不会有影响。networkx是在Env.G图里边：node和edge的信息用来进行任务的调度，输出最终方案和reward
        
        不画图了！！！！！！！
        
        """
        if "graph_console" in show:
            self.visualizer.graph_console(self.G, shape=self.size, colors=colors)
        if "gantt_console" in show:
            self.visualizer.gantt_chart_console(df=df, colors=colors)

        #===================上述画图，只要结果：这里注释，就加在reset的第一行==========================================

        if "graph_window" in show:
            if "gantt_window" in show:
                if mode == "human":
                    self.visualizer.render_graph_and_gant_in_window(G=self.G, df=df, colors=colors, **render_kwargs)
                elif mode == "rgb_array":
                    return self.visualizer.gantt_and_graph_vis_as_rgb_array(G=self.G, df=df, colors=colors)
            else:
                if mode == "human":
                    self.visualizer.render_graph_in_window(G=self.G, **render_kwargs)
                elif mode == "rgb_array":
                    return self.visualizer.graph_rgb_array(G=self.G)

        elif "gantt_window" in show:
            if mode == "human":
                self.visualizer.render_gantt_in_window(df=df, colors=colors, **render_kwargs)
            elif mode == "rgb_array":
                return self.visualizer.gantt_chart_rgb_array(df=df, colors=colors)


    '''
    _schedule_task返回的字典信息，包含了每一个节点的开始+结束+节点ID+是否有效动作+插入2任务之间+插入标志位
     return {
            "start_time": st,
            "finish_time": ft,
            "node_id": task_id,
            "valid_action": True,
            "scheduling_method": 'left_shift',
            "left_shift": 1,
            }
            
    self.G.add_node(    # 因为是循环，所以会每一个task都建立好了，从1开始的，0是源节点
                    task_id,
                    pos=(j, -i),
                    color=self.machine_colors[m_id],
                    duration=dur,         # 这里是我们设定的node的处理时间啊，他画图画在了有向边上
                    scheduled=False,
                    machine=m_id,
                    job=i,
                    start_time=None,
                    finish_time=None
                )
    '''

    """
    可以认为：
    1、前置节点有1个，入边只有1个，那就是最开始的子任务的约束
    2、如果有多个入边，说明该节点被scheduled调度了，那就不会再继续该节点了。所以默认的只有一个入边，没被调度，作为前置工序
    3、没被调度：咋办，看下边
    4、当前节点的结束时间ft我没法修改加权重，因为传给谁我不知道；（so，我只能如果用到ft结束时间之类的要比较，我要加上传输时间）：node的信息，ft的部分是错的，但是用起来比较的时候没有错！！！！！！！！！！！！！！！！！！！

    5/ finish_time就是按照定义：ft=开始+结束，不包括运输时间；这是正确的！！！开始时间才会受到运输时间的影响！！！！！！！！！！！！！！！！！！！！！！！（上一次的finish_time + 运输过来的时间 = 作为当前任务的开始时间st！！！）

    这段代码中的 `return` 语句的作用是向生成该方法的上一级代码返回一个 dict 对象。该 dict 对象包含了两个关键字参数，即 `"valid_action"` 和 `"node_id"`。
    其中，`"valid_action"` 表示该操作是否是有效的。如果前一个任务节点 `prev_job_node` 还没有被调度，那么该操作将会导致图中存在环路（cycle），因此会将 `"valid_action"` 设置为 False，表示这个操作是无效的。   
    如果该操作是无效的，那么在调用该方法的主代码块中，该方法应该返回这个无效操作的信息。如果 `self.verbose` 属性的值大于 1，那么输出一条 log 记录，
    描述前一个任务节点还没有被调度，然后在 `return` 语句中返回一个 dict，并将 `valid_action` 设置为 False，表示这个操作是无效的。   
    如果该操作是有效的（即前一个任务节点已经被调度），则该方法将继续执行，插入新任务节点，并将相关信息保存在返回的 dict 对象中，以用于后续的处理和调用。       
    """
    def _schedule_task(self, task_id: int) -> dict:
        """
        schedules a task/node in the graph representation if the task can be scheduled.
        选取子任务，在甘特图schedule中进行dispatch

        This adding one or multiple corresponding edges (multiple when performing a left shift) and updating the
        information stored in the nodes.
        添加一条边（选了方向），或者多条边（当前子任务加入插空2个任务之间，不会影响开始时间）
        更新node里边的信息：即该node的开始时间！！！

        :param task_id:     the task or node that shall be scheduled.   当前是哪个节点在进行调度安排
        :return:            a dict with additional information for the `DisjunctiveGraphJssEnv.step`-method. 为step方法返回一个字典，储存额外的信息

        """
        node = self.G.nodes[task_id]  # 获取当前的选择的action，即+1转换成DG上对应的那一个task
        duration = node["duration"]  # 当前节点的处理时间 （node字典的duration属性）

        if node["scheduled"]:  # 判断是否被调度了，
            if self.verbose > 0:
                log.info(f"task {task_id} is already scheduled. ignoring it.")
            return {
                "valid_action": False,  # 被调度了，就false表示不能再选了！！！
                "node_id": task_id,
            }

        m_id = node["machine"]  # 该task也就是node的分配的machine id是什么

        # 很常见的在有向无环图中查找当前节点的前驱节点的做法（我方法里没用这个了）


        # print("sdsdsdsdsdsdsdsdsdsd", list(self.G.in_edges(task_id)))
        prev_task_in_job_id, _ = list(self.G.in_edges(task_id))[0]  # 找当前task_id节点的所有入边，只用第一条入边，取出去节点id（why只用第一条入边？？？因为没调度之前，只有其前置子任务才会指向这个节点！）
        prev_job_node = self.G.nodes[prev_task_in_job_id]  # 按照节点id，获取上一个节点的信息，是个字典哦： 前置工序！！！！（当前节点没调度，且同一m上有调度好的节点，必为前置工序！！！！）

        if not prev_job_node["scheduled"]:  # 没调度，输出没调度
            if self.verbose > 1:
                log.info(f"the previous task (T{prev_task_in_job_id}) in the job is not scheduled jet. "
                         f"Not scheduling task T{task_id} to avoid cycles in the graph.")
            return {
                "valid_action": False,  # 这里的return啥意思，不是有效的动作选择（我随机产生已经避免了这种情况了，）
                "node_id": task_id,
            }


        len_m_routes = len(self.machine_routes[m_id])  # 输出当前task/node的分配的machine的已经选择了的路径的长度，为0：该m还没分配任何一个task
        if len_m_routes:  # 当前节点的m已经有分配了task,考虑插首or插中间；否则直接填入insert_at_index_0，因为是None

            if self.perform_left_shift_if_possible:  # 确认要采用left-shift来进一步减少动作的可能组合，插空方法，不会排得很长；否则直接加到序列队尾append_at_end

                # 返回task_id的最早的开始时间（其前置的最晚的到达时间= 前置的结束时间+ 运输时间（有/无）），函数中遍历了
                # j_lower_bound_st, self.energy_transport, self.exist_list = find_latest_arrivaTime_for_currentNode(self.G, task_id, self.edge_initial_weight, self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)
                j_lower_bound_st = find_max_arrivaTime_for_currentNode(self.G, task_id, self.edge_initial_weight)
                j_lower_bound_ft = j_lower_bound_st + duration # 当前开始（前置的最晚到达）+ 当前持续 = 当前节点的结束时间

                # check if task can be scheduled between src and first task
                # 判断能否插入到首位之前
                m_first = self.machine_routes[m_id][0]  # m_id：当前节点分配的machine，该machine的首位node
                # m_first_st, self.energy_transport, self.exist_list = find_latest_arrivaTime_for_currentNode(self.G, m_first, self.edge_initial_weight, self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)
                m_first_st = find_max_arrivaTime_for_currentNode(self.G, m_first, self.edge_initial_weight)

                #当前节点的完工时间ft 小于等于 同一个m上的第一个节点的开始时间st时：可以插入到第一节点之前开始
                if j_lower_bound_ft <= m_first_st:
                    # schedule task as first node on machine
                    # self.render(show=["gantt_console"])
                    """ 
                    在 字典machine_routes的m_id设备的array中，第一位插入task_id，基于prev_job_node的结束时间，更新node的开始/结束时间/被调度
                    # 当前节点task_id，用的m_id的设备，信息在node里边，prev_job_node是其前置工序！
                    # 当前节点放在队列首位，然后按照其前置工序更新其开始和结束时间；原先的第一个节点后移了，整体后移（因为是插入）；更新当前节点的node字典信息；返回当前节点的node信息
                    # 里边也会更新当前节点的信息
                    """
                    info = self._insert_at_index_0(task_id=task_id, node=node, m_id=m_id)

                    # 更新新边：task_id---m_first
                    # 查找两个节点ID之间是否存在运输时间（同一个job？）
                    # transport_t, self.energy_transport, self.exist_list = make_sure_transportT_work(self.G, task_id, m_first, self.edge_initial_weight, self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)
                    transport_t = find_transportT(self.G, task_id, m_first, self.edge_initial_weight)

                    # 更新新边：当前节点，和，同一个m的第一个节点的
                    self.G.add_edge(  # task_id移到当前设备的任务加工顺序的首位，与下一位的m_first连接边
                        task_id, m_first,
                        job_edge=False,  # 判断是否是原先最开始画的平行边（表示同一个job的子任务先后顺序！！）
                        # weight=duration
                        # wrk 增加初始权重
                        weight=duration + transport_t  # duration是当前节点处理时间，现在移到前边了，edge的权重就是他的处理时间+运输（如果有的话）
                    )
                    # self.render(show=["gantt_console"])
                    return info
                elif len_m_routes == 1:  # 当前节点的完工时间ft 大于等于 同一个m上的第一个节点的开始时间st时：没必要插空，且当前m只有一个节点
                    return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
                    # return self._append_at_the_end(task_id=task_id, node=node, m_id=m_id)

                # check if task can be scheduled between two tasks
                # 这种插空的方式会不会导致缺少了一种可行解，虽然是有利于时间最少的，因为还有能耗呀！！！！！！！！！！！！！！！！！！！！！！！！！！！！（不同可行解会影响等待能耗吗？？？？干活能耗选定m是可算出来的，与不同可行解无关！！！！）
                # m_id 当前节点用的设备的索引
                """
                1、被调度好的节点，且又是在machine_route里边的，不会只有一个入边和一个出边（会有多个的入边，那就用新写的函数find_max_st_in_inedge找最长的结束时间，即可以开始当前任务的时间）
                2、所以可以按照入边的节点，查询到所属的machine，然后查表得运输t
                """
                for i, (m_prev, m_next) in enumerate(zip(self.machine_routes[m_id], self.machine_routes[m_id][1:])):  # 遍历machine是同一个m_id的列表，枚举每一对相邻的约束任务节点:返回是节点id
                    m_temp_prev_ft = self.G.nodes[m_prev]["finish_time"]
                    m_temp_next_st = self.G.nodes[m_next]["start_time"]  # 后一个节点的开始时间是不对的，因为有运输时间的；结束时间

                    # wrk 有节点的前后，就有运输t
                    # 找m_next节点的所有入边，里边的最长结束时间（ft + 运输t），即m_first的最大开始时间
                    # m_next_st, self.energy_transport, self.exist_list = find_latest_arrivaTime_for_currentNode(self.G, m_next, self.edge_initial_weight, self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)
                    m_next_st = find_max_arrivaTime_for_currentNode(self.G, m_next, self.edge_initial_weight)

                    if j_lower_bound_ft > m_next_st:    # 当前节点的结束时间 大于 集合中的后一个节点的开始时间：不能插空
                        continue                        # Python跳过当前循环的剩余语句，然后继续进行下一轮循环
                    m_gap = m_next_st - m_temp_prev_ft
                    if m_gap < duration:                # 两个节点之间的时间空隙 小于 当前节点的持续时间
                        continue                        # 跳过循环，下一轮

                    # at this point the task can fit in between two already scheduled task
                    # self.render(show=["gantt_console"])

                    # remove the edge from m_temp_prev to m_temp_next
                    replaced_edge_data = self.G.get_edge_data(m_prev, m_next)  # 获取m_prev和m_next之间的边的属性

                    """ 
                    更新插在中间的节点的node信息：
                    1、遍历当前task_id的所有前置节点，找其最晚的到达时间（当前节点的最早开始时间）
                    2、插入后，前一个节点prev，送到当前task_id，可能的运输时间
                    """
                    # task_id_st_forPrev, self.energy_transport, self.exist_list = find_latest_arrivaTime_for_currentNode(self.G, task_id, self.edge_initial_weight,self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)
                    task_id_st_forPrev = find_max_arrivaTime_for_currentNode(self.G, task_id, self.edge_initial_weight)
                    transport_t = find_transportT(self.G, m_prev, task_id, self.edge_initial_weight)
                    # transport_t, self.energy_transport, self.exist_list = make_sure_transportT_work(self.G, m_prev, task_id, self.edge_initial_weight,self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)

                    st = max(task_id_st_forPrev, (self.G.nodes[m_prev]["finish_time"] + transport_t))  # 入边的节点中的最大的结束时间（+了运输时间），因为有先后约束的
                    ft = st + duration  # ft结束时间是正常的，不包括运输的
                    node["start_time"] = st
                    node["finish_time"] = ft
                    node["scheduled"] = True

                    """
                    更新新边： 2个
                    from previous task to the task to schedule
                    原先是m_prev到m_next的边，现在中间插入了task_id：前一个有向边是原先m_prev到m_next的边的权重，后边是当前节点的权重；当然，都会加上运输t；（边权重=发出节点的duration时间）
                    """
                    self.G.add_edge(
                        m_prev, task_id,
                        job_edge=replaced_edge_data['job_edge'],  # 判断是否是原先最开始画的平行边，表示同一个job的子任务先后顺序！！
                        # weight=replaced_edge_data['weight']
                        # wrk
                        weight=replaced_edge_data['weight'] + transport_t  # 加上运输时间
                    )
                    # from the task to schedule to the next 查找当前task_id到next节点的可能运输时间
                    # transport_t, self.energy_transport, self.exist_list = make_sure_transportT_work(self.G, task_id, m_next, self.edge_initial_weight,self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)
                    transport_t = find_transportT(self.G, task_id, m_next, self.edge_initial_weight)

                    self.G.add_edge(
                        task_id, m_next,
                        job_edge=False,
                        # weight=duration
                        # wrk
                        weight=duration + transport_t  # 当前节点的持续 + 加上查表得的运输时间t
                    )
                    # remove replaced edge
                    self.G.remove_edge(m_prev, m_next)  # 删除之前的边
                    # insert task at the corresponding place in the machine routes list 更新下machine_routes，因为有插空
                    self.machine_routes[m_id] = np.insert(self.machine_routes[m_id], i + 1, task_id)  # 在_schedule_task这里会更新所有machine的已经被选择的路径！！！！

                    if self.verbose > 1:
                        log.info(f"scheduled task {task_id} on machine {m_id} between task {m_prev:.0f} "
                                 f"and task {m_next:.0f}")
                    # self.render(show=["gantt_console"])
                    return {  # 返回节点（子任务的）的开始和结束时间（甘特图就看子任务先后顺序，和同一个m上的先后顺序，先后用时间来表示：gantt上边的抽象时间）
                        "start_time": st,
                        "finish_time": ft,
                        "node_id": task_id,
                        "valid_action": True,
                        "scheduling_method": 'left_shift',
                        "left_shift": 1,
                    }
                else:
                    return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
                    # return self._append_at_the_end(task_id=task_id, node=node, m_id=m_id)

            else:
                return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
                # return self._append_at_the_end(task_id=task_id, node=node, m_id=m_id)

        else:  # 当前节点的machin_route为None，这个当前节点直接加入序列中
            return self._insert_at_index_0(task_id=task_id, node=node, m_id=m_id)

    # 牵扯了我如果选了某个子任务，它的无向边怎么变化？都是画图相关
    # 那个left_shift会不会无形之中减少了某一类可行解？？？？（不只看时间的）
    def _append_at_the_end(self, task_id: int, node: dict, prev_job_node: dict, m_id: int) -> dict:
    # def _append_at_the_end(self, task_id: int, node: dict, m_id: int) -> dict:
        """
        inserts a task at the end (last element) in the `DisjunctiveGraphJssEnv.machine_routes`-dictionary.
        在' DisjunctiveGraphJssEnv.machine_routes '字典的最后一个元素插入一个任务。

        :param task_id:             the id oth the task with in graph representation.
        :param node:                the corresponding node in the graph (self.G).
        :param prev_job_node:       the node the is connected to :param node: via a job_edge (job_edge=True).
        :param m_id:                the id of the machine that corresponds to :param task_id:.
        :return:                    a dict with additional information for the `DisjunctiveGraphJssEnv.step`-method.

        这段代码实现的是在已分配机器 route 列表的末尾（即最后一个元素）添加任务节点的 _append_at_the_end 方法
        """
        prev_m_task = self.machine_routes[m_id][-1]  # 当前m_id设备的最后一个子任务的id
        prev_m_node = self.G.nodes[prev_m_task]  # 最后一个子任务的node信息

        """
        更新新边： 序列最后一个node --- 当前节点（因为是加在最后了呀）
        1、最后一个节点： prev_m_task
        2、更新machine_route的列表
        """
        # transport_t, self.energy_transport, self.exist_list = make_sure_transportT_work(self.G, prev_m_task, task_id, self.edge_initial_weight, self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)
        transport_t = find_transportT(self.G, prev_m_task, task_id, self.edge_initial_weight)

        self.G.add_edge(
            prev_m_task, task_id,  # 添加原先最后一个子任务prev_m_task，到，当前任务task_id的，边(权重都是上一个节点的权重)
            job_edge=False,
            # weight=prev_m_node['duration']
            # wrk 修改权重
            weight=prev_m_node['duration'] + transport_t  # 查表增加初始权重
        )
        self.machine_routes[m_id] = np.append(self.machine_routes[m_id], task_id)  # 将任务ID task_id 插入到当前任务用的m的序列中

        """
        更新当前节点的信息：
        1、前置任务：分配好的所有前置节点 vs 列表最后一个任务，都和当前节点比较
        2、 所以比较，谁的结束时间最长，当前就从哪里开始！！！
        3、考虑加上传输时间了，都是上一个（有两个，一个前置工序，一个同m的最后一个任务）传到当前的运输时间
        """
        # task_id_st_forPrev, self.energy_transport, self.exist_list = find_latest_arrivaTime_for_currentNode(self.G, task_id, self.edge_initial_weight,self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)
        task_id_st_forPrev = find_max_arrivaTime_for_currentNode(self.G, task_id, self.edge_initial_weight)
        transport_t = find_transportT(self.G, prev_m_task, task_id, self.edge_initial_weight)
        # transport_t, self.energy_transport, self.exist_list = make_sure_transportT_work(self.G, prev_m_task, task_id, self.edge_initial_weight, self.n_jobs,self.n_machines, self.energy_transport, self.exist_list)

        st = max(task_id_st_forPrev, (prev_m_node["finish_time"] + transport_t))  # prev_job_node前置的任务，同m的列表的最后一个任务,谁大就排在谁后边！（约束摆着呢）
        ft = st + node["duration"]  # 结束 = 开始 +持续；当前不知道传给谁，所以不用加传输时间
        node["start_time"] = st
        node["finish_time"] = ft
        node["scheduled"] = True
        # return additional info
        return {
            "start_time": st,
            "finish_time": ft,
            "node_id": task_id,
            "valid_action": True,
            "scheduling_method": '_append_at_the_end',
            "left_shift": 0,
        }

    def _insert_at_index_0(self, task_id: int, node: dict, m_id: int) -> dict:
        """
        inserts a task at index 0 (first element) in the `DisjunctiveGraphJssEnv.machine_routes`-dictionary.

        :param task_id:             the id oth the task with in graph representation.
        :param node:                the corresponding node in the graph (self.G).
        :param prev_job_node:       the node the is connected to :param node: via a job_edge (job_edge=True).    job_edge理解为子任务约束的边吗，固有存在的
        :param m_id:                the id of the machine that corresponds to :param task_id:.
        :return:                    a dict with additional information for the `DisjunctiveGraphJssEnv.step`-method.
        """
        self.machine_routes[m_id] = np.insert(self.machine_routes[m_id], 0, task_id)  # 在 字典machine_routes的m_id设备的array中，第一位插入task_id

        """ 
        更新当前节点:
        1、当前节点的开始时间（前置的最晚的到达时间）
        2、当前的开始时间 = （上一个节点的结束时间 + 前置节点运输消耗的时间）max
        self.edge_initial_weight = m * m  运输能力表！
        """
        # st, self.energy_transport, self.exist_list = find_latest_arrivaTime_for_currentNode(self.G, task_id, self.edge_initial_weight, self.n_jobs, self.n_machines, self.energy_transport, self.exist_list)
        st = find_max_arrivaTime_for_currentNode(self.G, task_id, self.edge_initial_weight)
        ft = st + node["duration"]  # 结束时间 = 开始时间 + 当前节点的持续时间，不用知道传给谁，反正任务已经在此节点结束了！！！
        node["start_time"] = st  # 更新node的各种信息
        node["finish_time"] = ft
        node["scheduled"] = True
        # return additional info
        return {  # 并返回信息
            "start_time": st,
            "finish_time": ft,
            "node_id": task_id,
            "valid_action": True,
            "scheduling_method": '_insert_at_index_0',
            "left_shift": 0,
        }

    """
    1、需要实时更新ft和st的状态：因为每一次调度之后，上一时刻的完工时间就变了
        1/1 或者不是每一步都更新的，直接初始化的时候就固定了
    self.jsp_instance = 被选择的m + 对应的加工时间
    self.edge_initial_weight = m * m的传输时间
    
    current_ft:  当前时刻的ft矩阵
    if_schedule： 01表示是否被调度的task
    估计值：（不考虑耦合）st1 + pt1 + trasnT1 = st2  | ft1 + transt1 + pt2 = ft2
    """
    def estiamte_ft_st_eachStep(self, current_ft, current_st, if_schedule):
        # np.array： j * m 加工时间 + m * m 运输时间
        machine_j_m = copy.deepcopy(self.jsp_instance[0]).astype(int)  # machine保持整数！就2个元素，第一个是选择的m，第二个是对应的加工时间
        processT_j_m = copy.deepcopy(self.jsp_instance[-1])  # 就2个元素，第一个是选择的m，第二个是对应的加工时间
        transT_m_m = copy.deepcopy(self.edge_initial_weight) # m和m之间的运输时间

        # print("ssssssssssssssss:",current_ft,if_schedule)
        begin_ft = current_ft * if_schedule # 保留真实值，重新计算其他未调度的task的估计值  (此为有0的新state！！！！)
        update_ft = begin_ft # 实际更新的ft矩阵
        # print("begin_ft = ", begin_ft)

        begin_st = current_st * if_schedule  # 保留真实值，重新计算其他未调度的task的估计值  (此为有0的新state！！！！)
        update_st = begin_st  # 实际更新的st矩阵
        # print("begin_st = ", begin_st)

        # 傻瓜式遍历更新
        for row in range(current_ft.shape[0]):  # 矩阵的行,job数量
            for col in range(current_ft.shape[1]): # 矩阵的列，machine数量
                if begin_ft[row][col] == 0:
                    if col != 0:  # 防止是第一列，index会出错
                        # print(row,col)
                        # print(update_ft[row][col-1])
                        # print(machine_j_m[row][col-1])
                        # print(machine_j_m[row][col])
                        # print(transT_m_m[machine_j_m[row][col-1]][machine_j_m[row][col]])
                        # print(processT_j_m[row][col])
                        update_ft[row][col] = update_ft[row][col-1] + transT_m_m[machine_j_m[row][col-1]][machine_j_m[row][col]] + processT_j_m[row][col]# 运输t都是同job（同一行）中上一个m到当前m
                    else:
                        update_ft[row][col] = 0 + 0 + processT_j_m[row][col] # 首列，直接就是加工时间！
                # st的初始值，第一列有0存在，但不一定全是0
                if if_schedule[row][col] == 0: # 说明还没有被调度，所以需要进行估计
                    if col == 0:
                        update_st[row][col] = 0 # 说明此时第一列的首个task，还没被调度，开始时间预估为0
                    else:
                        update_st[row][col] = update_st[row][col-1] + processT_j_m[row][col-1] + transT_m_m[machine_j_m[row][col-1]][machine_j_m[row][col]] # 运输t都是同job（同一行）中上一个m到当前m
        # 遍历，重新更新当前step且还没调度的task的ft和st
        return update_ft, update_st

    def _state_array(self) -> (np.ndarray,np.ndarray,np.ndarray):
        """
        returns the state of the environment as numpy array.

        将环境的状态返回为numpy数组，看看到底是什么样子的？

        是一个Box的形式

        :return: the state of the environment as numpy array.
        """

        """
        去掉开始和结束两个虚假节点，返回的是节点的邻居矩阵：
        1、其中adj元素 = 边属性 = 加工t+运输t（if有）：需要去掉加工时间t，这里是为了计算mk方便，但是聚合时不应该使用
        2、需要进行转置：我记录的只有入度的节点才会聚合（至于上一状态有关，后续未分配关系不大）
        """
        adj = nx.to_numpy_array(self.G)[1:-1, 1:-1].astype(dtype=int)  # remove dummy tasks
        # print("--test: adj = \n", adj, adj.shape)

        adj_wrk = copy.deepcopy(adj) # 复制一个新的adj用来处理，不改变原adj

        """
        每次新增边（即选了task），都会实时更新adj（此时边都是判断过：dur+transT（if）），加空闲时间：
        1、在 “指定位置”（新增的边） 增加空闲时间，边权重变成：dur+transT（if）+ idleT（if）
        2、然后再统一减去加工时间dur，如果为0，那就置1；不为0，就是transT（if）+ idleT（if）
        3、DG图中的我就不改了，加不加it时间对于最终的mk没有影响（最大的完工时间），只是说同m的新增边的边权重不是真实的，少了空闲时间，只有上一个node的加工时间！！！
        4、遍历machien_route（step实时更新）, 给已经调度的task的新增边，在adj上边增加空闲时间    
            machien_routes = {0: array([], dtype=int64), 1: array([3, 4, 1, 2])}  字典形式：存有同一个m中的task的加工顺序！(记录task_id，从1开始的)
        5、route中只有一个task的，不会新增边！不用考虑。新边：route中上一个id i---下一个id j，空闲计算，然后加到此 “指定位置”：【i-1】【j-1】 （task id从1开始的）
        6、最终结果：有边没有trans和it=1，其余有边=transT（if）+it（if）！！！
        
         (只有同一个m的task之间才会有新增的边，这些边上我在调度时会计算dur+transT，现在把已经调度的空闲t也加上！！！！！！！只计算st-ft的，前边首个task留白的不用计算)
        """
        for route in self.machine_routes.values():  # value的列表
            # 设备列表中》一个任务，才有新边；
            if len(route) > 1:  # {0：[2,5,7]}
                for i in range(len(route) - 1):  # 每个value列表中的元素个数 - 1, 防止超出索引
                    blank = self.G.nodes[route[i + 1]]['start_time'] - self.G.nodes[route[i]]['finish_time']  # 遍历每个列表中的元素：后一个st开始 - 前一个ft结束
                    adj_wrk[route[i]-1][route[i + 1]-1] += blank  # taskid - 1变成adj的索引，当前值+新增的it空闲时间

        # 遍历邻接矩阵中的每个元素: 进行去加工时间t的操作，只剩运输时间+新增边的空闲时间（注意：若运输时间为0，则变成普通权重1）
        for i in range(adj_wrk.shape[0]):
            for j in range(adj_wrk.shape[1]):
                # 检查元素是否为非零值
                if adj_wrk[i, j] != 0:
                    # 获取对应节点的 dur 属性值，并减去该值
                    node_dur = self.G.nodes[i + 1].get('duration', 0)  # task从1开始，获取节点的 dur 属性值，没有默认为 0
                    adj_wrk[i, j] -= node_dur
                    # 防止没有运输时间，导致邻居信息变为0
                    if adj_wrk[i,j] == 0:
                        adj_wrk[i,j] = 1
        # print("--test: adj_wrk = \n", adj_wrk, adj_wrk.shape)

        # 单位矩阵
        identity_matrix = np.eye(adj_wrk.shape[0])
        # 将原始矩阵与单位矩阵相加
        adj_wrk = adj_wrk + identity_matrix
        # print("--test: adj_wrk = \n", adj_wrk, adj_wrk.shape)
        # 转置结果矩阵
        adj_wrk = adj_wrk.T
        # print("--test: adj_wrk = \n", adj_wrk, adj_wrk.shape)

        task_to_machine_mapping = np.zeros(shape=(self.total_tasks_without_dummies, 1), dtype=int)
        task_to_duration_mapping = np.zeros(shape=(self.total_tasks_without_dummies, 1), dtype=self.dtype)
        for task_id, data in self.G.nodes(data=True):
            if task_id == self.src_task or task_id == self.sink_task:
                continue
            else:
                # index shift because of the removed dummy tasks
                task_to_machine_mapping[task_id - 1] = data["machine"]
                task_to_duration_mapping[task_id - 1] = data["duration"]

        if self.normalize_observation_space:
            # one hot encoding for task to machine mapping   task是one-hot编码的，和machine是映射的；就是task-machine的那个表格，也是m的分布
            task_to_machine_mapping = task_to_machine_mapping.astype(int).ravel()
            n_values = np.max(task_to_machine_mapping) + 1
            task_to_machine_mapping = np.eye(n_values)[task_to_machine_mapping]

            """
            normalize归一化：
            1、注释掉了：原先是除以最大的加工时间
            """
            # 现在是边的权重也加上了，所以最好是不要再有归一化了（变得不准了，会有》1数），所以，先注释！！！！！！！！！！！
            # adj = adj / self.longest_processing_time  # note: adj matrix contains weights  adj matrix里边包含的边边的权重，作为析取图的状态
            # task_to_duration_mapping = task_to_duration_mapping / self.longest_processing_time

            # merge arrays
            res = np.concatenate((adj, task_to_machine_mapping, task_to_duration_mapping), axis=1, dtype=self.dtype)
            """
            wrk
            更新状态的输出：边权重信息 + 单列被选择信息(掩码信息，state里边暂时不需要了，先注释!!!)
            上述res还没有扁平化（归一化被我注释了！）
            """
            out_s = res[:, 0:self.total_tasks_without_dummies]      #上边代码整形，这边只选取析取图状态相关的（task*task）
            task_s = [0] * self.total_tasks_without_dummies # 记录被选择了的task
            for i in self.selected_action:  # selected_action = action[0-15]正好对应task[1-16]：[0,1,2,3,4,xxxx]都是action
                task_s[i] = 1
            out_s = np.column_stack((out_s, task_s)) #添加到task*task矩阵的最后一列
            res = out_s # 输出正确

            # 输出被选择的task的完工时间
            # idle时间需要全局初始化，和reset清0
            # 先计算idle当前，再更新state，最后再把这次的复制到prev变量中，所以这里的计算：当前是新的-上一次是旧的
            ft_s = [0] * self.total_tasks_without_dummies # 全0列表
            for i in self.selected_action:   # selected_action = action[0-15]正好对应task[1-16]：[0,1,2,3,4,xxxx]都是action不断累加进来的，0-16个数
                ft_s[i] = self.G.nodes[i+1]['finish_time'] # node图中记录的都是task_id
            if self.selected_action:  # 不是空集的时候
                self.it_s[self.selected_action[-1]] = self.idle_t_this_step - self.idle_t_previous_step  # 每一步，当前action的新增的it记录在对应位置
            ft_s = np.array(ft_s)
            self.it_s = np.array(self.it_s)

            # print(res)

            """
            每一个step都会更新一下state
            1、先判断是否被调度
            2、输出已调度的状态 
            3、更新未调度的预估的状态
            最终，作为当前的状态进行输出                                                                   
            """
            if_schedule_lst = [0] * self.total_tasks_without_dummies # 全0列表
            ft_lst = [0.0] * self.total_tasks_without_dummies # 全0列表
            st_lst = [0.0] * self.total_tasks_without_dummies # 全0列表
            mOrder_lst = [copy.deepcopy(self.n_machines)] * self.total_tasks_without_dummies # 全max machine数量的列表
            # 每一步都重头开始遍历，确定当前step的状态
            # 下列是当前的已调度的完工时间ft和开始时间st + 已经选择的m的id + 是否被调度（未选的元素统一为0！！）
            for i_a in self.selected_action: # selected_action = action[0-15]正好对应task[1-16]：[0,1,2,3,4,xxxx]都是action不断累加进来的，0-16个数
                if_schedule_lst[i_a] = 1  # 对应task被调度，被选了
                ft_lst[i_a] = self.G.nodes[i_a + 1]['finish_time'] # node图中记录的都是task_id, 从1开始的
                st_lst[i_a] = self.G.nodes[i_a + 1]['start_time'] # node图中记录的都是task_id, 从1开始的
                mOrder_lst[i_a] = self.jsp_instance[0][i_a // self.n_machines][i_a % self.n_machines] # jsp_instance中首位是选择的m（np.array），action的id转换成矩阵的行和列
            # np.array->(self, current_ft, current_st, if_schedule), 输出ft和st的矩阵

            # print("ssssssssss：selected_action = ", self.selected_action, ft_lst)
            # 按照已选的m的加工时间和运输时间，预估每一个节点的每一步的完工时间ft和开始时间st（每个元素都在预估，防止0出现）
            ft_s_array, st_s_array = self.estiamte_ft_st_eachStep(current_ft=np.array(ft_lst).reshape(self.n_jobs,self.n_machines),
                                                                  current_st=np.array(st_lst).reshape(self.n_jobs,self.n_machines),
                                                                  if_schedule=np.array(if_schedule_lst).reshape(self.n_jobs,self.n_machines))
            mOrder_s_array = np.array(mOrder_lst).reshape(self.n_jobs,self.n_machines)

            # print("---Operation_new_states: ft = {},{} + st = {},{} + mOrder = {}, {}".format(ft_s_array,ft_s_array.shape,st_s_array,st_s_array.shape,mOrder_s_array,mOrder_s_array.shape))

            """
            基于上述信息组合成：tasks * 4 的各个task节点的特征向量
            1、特征向量： 完工时间（预估完工时间=上一次完工+加工+运输，m已确定） + 是否被调度 + 对应的能耗e1=p*t（固定已知） + 已分配的m的id（固定已知）
            2、每一step，都要更新特征向量的！（此def只会被调用一次）
            3、要记录所有task节点的特征向量：task，vector
            """
            tasks_fea = []
            ft_s_estimated = ft_s_array.flatten()  # 将实时更新的预估ft二维矩阵，转成1维
            m_id_chosen = self.jsp_instance[0].flatten()  # 将选好的m的id矩阵，转成一维，展平
            e1_chosen = self.initial_energy.flatten() # 将选好的加工能耗e1 = p*t矩阵，转成一维
            for i in range(self.total_tasks_without_dummies):  # 遍历所有节点
                task_fea = []
                task_fea.append(ft_s_estimated[i])  # 对应节点的ft + 预估ft
                task_fea.append(if_schedule_lst[i])  # 对应节点是否被调度
                task_fea.append(e1_chosen[i]) # 对应选m定好的能耗e1
                task_fea.append(m_id_chosen[i]) # 每个task对应的m，从0开始，是否要+1，从1开始？？？？

                # 记录每一个task节点
                tasks_fea.append(copy.deepcopy(task_fea))  # 4维度的特征向量
            tasks_fea = np.array(tasks_fea) # list转成二维矩阵！
            # print("--test: tasks_fea = ", tasks_fea, tasks_fea.shape)

            """

            Example:

            normalize_observation_space = True
            (flat_observation_space = False)

            jsp: (numpy array)
            [
                # jobs order on machine
                [
                    [1, 2, 0],      # job 0
                    [0, 2, 1]       # job 1
                ],
                # task durations within a job
                [
                    [17, 12, 19],   # task durations of job 0
                    [8, 6, 2]       # task durations of job 1
                ]

            ]

            total number of tasks: 6 (2 * 3)

            scaling/normalisation:   缩放和标准化

                longest_processing_time = 19 (third task of the first job)

            initial observation:

            ┏━━━━━━━━━┳━━━━━━━━┯━━━━━━━━┯━━━━━━┯━━━━━━━━┳━━━━━━━━━━━┯━━━━━━━━━━━┯━━━━━━━━━━━┳━━━━━━━━━━┓
            ┃         ┃ task_1 │ task_2 │ ...  │ task_6 ┃ machine_0 │ machine_1 │ machine_2 ┃ duration ┃
            ┣━━━━━━━━━╋━━━━━━━━┿━━━━━━━━┿━━━━━━┿━━━━━━━━╋━━━━━━━━━━━┿━━━━━━━━━━━┿━━━━━━━━━━━╋━━━━━━━━━━┫
            ┃ task_1  ┃  0.    │ 17/19  │  ... │  0.    ┃    0.     │    0.     │    0.     ┃    17/19 ┃
            ┠─────────╂────────┼────────┼──────┼────────╂───────────┼───────────┼───────────╂──────────┨
            ┃ task_2  ┃  0.    │   0.   │  ... │  0.    ┃    0.     │    0.     │    1.     ┃    12/19 ┃
            ┠─────────╂────────┼────────┼──────┼────────╂───────────┼───────────┼───────────╂──────────┨
            ┠ ...     ┃  ...   │   ...  │  ... │  ...   ┃       ... │       ... │       ... ┃      ... ┃
            ┠─────────╂────────┼────────┼──────┼────────╂───────────┼───────────┼───────────╂──────────┨
            ┃ task_6  ┃  0.    │   0.   │  ... │  0.    ┃    0.     │    1.     │    0.     ┃     2/19 ┃
            ┗━━━━━━━━━┻━━━━━━━━┷━━━━━━━━┷━━━━━━┷━━━━━━━━┻━━━━━━━━━━━┷━━━━━━━━━━━┷━━━━━━━━━━━┻━━━━━━━━━━┛

            or:

            [
                [0.        , 0.89473684,     ..., 0.        , 0.        , 1.        ,0.        , 0.89473684],
                [0.        , 0.        ,     ..., 0.        , 0.        , 0.        ,1.        , 0.63157895],
                ...
                [0.        , 0.        ,     ..., 0.        , 0.        , 1.        ,0.        , 0.10526316]
            ]
            """
        else:
            """
            Example:

            normalize_observation_space = False
            (flat_observation_space = False)

            jsp: (numpy array)
            [
                # jobs order on machine
                [
                    [1, 2, 0],      # job 0
                    [0, 2, 1]       # job 1
                ],
                # task durations within a job
                [
                    [17, 12, 19],   # task durations of job 0
                    [8, 6, 2]       # task durations of job 1
                ]

            ]

            total number of tasks: 6 (2 * 3)

            initial observation:

            ┏━━━━━━━━┳━━━━━━━━┯━━━━━━━━┯━━━━━┯━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
            ┃        ┃ task_1 │ task_2 │ ... │ task_6  ┃ machine ┃ duration ┃
            ┣━━━━━━━━╋━━━━━━━━┿━━━━━━━━┿━━━━━┿━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━━┫
            ┃ task_1 ┃     0. │    17. │ ... │      0. ┃      1. ┃      17. ┃
            ┠────────╂────────┼────────┼─────┼─────────╂─────────╂──────────┨
            ┃ task_2 ┃     0. │     0. │ ... │      0. ┃      2. ┃      12. ┃
            ┠────────╂────────┼────────┼─────┼─────────╂─────────╂──────────┨
            ┃ ...    ┃    ... │    ... │ ... │     ... ┃     ... ┃       .. ┃
            ┠────────╂────────┼────────┼─────┼─────────╂─────────╂──────────┨
            ┃ task_6 ┃     0. │     0. │ ... │      0. ┃      1. ┃       2. ┃
            ┗━━━━━━━━┻━━━━━━━━┷━━━━━━━━┷━━━━━┷━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━━━┛

            or

            [
                [ 0., 17.,  ...,  0.,  1., 17.],
                [ 0.,  0.,  ...,  0.,  2., 12.],
                ...
                [ 0.,  0.,  ...,  0.,  1.,  2.]
            ]
            """
            res = np.concatenate((adj, task_to_machine_mapping, task_to_duration_mapping), axis=1, dtype=self.dtype)

        if self.flat_observation_space:
            # falter observation
            res = np.ravel(res).astype(self.dtype)     #ravel函数的功能是将原数组拉伸成为一维数组：从左到右，从上到下依次展开

        if self.env_transform == 'mask':
            res = OrderedDict({
                "action_mask": np.array(self.valid_action_mask()).astype(np.int32),
                "observations": res
            })

        return res, ft_s, self.it_s, adj_wrk, tasks_fea

    def network_as_dataframe(self) -> pd.DataFrame:
        """
        returns the current state of the environment in a format that is supported by Plotly gant charts.
        (https://plotly.com/python/gantt/)

        :return: the current state as pandas dataframe
        """
        return pd.DataFrame([
            {
                'Task': f'Job {data["job"]}',
                'Start': data["start_time"],    #从这里看，真的是按照每个任务的开始和结束时间来画图的！
                'Finish': data["finish_time"],
                'Resource': f'Machine {data["machine"]}'
            }
            for task_id, data in self.G.nodes(data=True)
            if data["job"] != -1 and data["finish_time"] is not None
        ])

    def valid_action_mask(self, action_mode: str = None) -> List[bool]:
        """
        returs that indicates which action in the action space is valid (or will have an effect on the environment) and
        which one is not.

        :param action_mode:     Specifies weather the `action`-argument of the `DisjunctiveGraphJssEnv.step`-method
                                corresponds to a job or a task (or node in the graph representation)

        :return:                list of boolean in the same shape as the action-space.
        """
        if action_mode is None:
            action_mode = self.action_mode

        if action_mode == 'task':
            mask = [False] * self.total_tasks_without_dummies
            for task_id in range(1, self.total_tasks_without_dummies + 1):
                node = self.G.nodes[task_id]

                if node["scheduled"]:
                    continue

                prev_task_in_job_id, _ = list(self.G.in_edges(task_id))[0]
                prev_job_node = self.G.nodes[prev_task_in_job_id]

                if not prev_job_node["scheduled"]:
                    continue

                mask[task_id - 1] = True

            if True not in mask:
                if self.verbose >= 1:
                    log.warning("no action options remaining")
                if not self.env_transform == 'mask':
                    raise RuntimeError("something went wrong")  # todo: remove error?
            return mask
        elif action_mode == 'job':
            task_mask = self.valid_action_mask(action_mode='task')
            masks_per_job = np.array_split(task_mask, self.n_jobs)
            return [True in job_mask for job_mask in masks_per_job]
        else:
            raise ValueError(f"only 'task' and 'job' are valid arguments for 'action_mode'. {action_mode} is not.")

    """
    新的load_instance
    为了m选择的时候输出状态
    """
    #
    # # 建立Disjunctive Graph里边的节点和边
    # def load_machine_instance(self, jsp_instance: np.ndarray, *, reward_function_parameters: Dict = None) -> None:  # python里边同样名字变量不会报错，果然方便了好多！
    #
    #     _, n_jobs, n_machines = jsp_instance.shape  # 获取几个job，几个machine，简单理解几行几列（不全对，算的是维度）,不关心有有几个大维度，我又新加上了边权重（设备与设备之间的传输时间）
    #     self.size = (n_jobs, n_machines)  # 自定义size = 上步获取到的信息
    #     self.n_jobs = n_jobs  # job个数
    #     self.n_machines = n_machines  # machine个数
    #     self.total_tasks_without_dummies = n_jobs * n_machines  # 不算source和sink，有几个点  12个
    #     self.total_tasks = n_jobs * n_machines + 2  # 2 dummy tasks: start, finish  #全图几个点，算上source和sink
    #     self.src_task = 0  # source节点的编号
    #     self.sink_task = self.total_tasks - 1  # sink节点的编号
    #
    #     self.longest_processing_time = jsp_instance[1].max()  # 找第二个矩阵（处理时间）中的最大值
    #
    #     # print("ssssssssssssssssssssssssssssss", self.edge_initial_weight)
    #
    #     if self.action_mode == 'task':  # 按照子任务来走action
    #         self.action_space = gym.spaces.Discrete(
    #             self.total_tasks_without_dummies)  # 创建一个离散的n维空间：动作空间有多大（子任务总个数），是离散的，12个节点
    #     else:  # action mode 'job'
    #         self.action_space = gym.spaces.Discrete(self.n_jobs)
    #
    #     # 生成状态空间的大小
    #     if self.normalize_observation_space:  # 观测空间space的设定，是否正规范化（归一化，方便学习）：shape理解成维度，12行方括号，每个方括号内17个元素
    #         self.observation_space_shape = (self.total_tasks_without_dummies,  # 归一化，eg (12,12+4+1)  why加上设备数?
    #                                         self.total_tasks_without_dummies + self.n_machines + 1)
    #     else:
    #         self.observation_space_shape = (
    #             self.total_tasks_without_dummies,
    #             self.total_tasks_without_dummies + 2)  # 不归一，eg (12,12+2)，2：开始和结束？  why?
    #
    #     if self.flat_observation_space:  # 观测空间扁平化
    #         a, b = self.observation_space_shape  # 维度读取：行+元素
    #         self.observation_space_shape = (a * b,)  # 一维的，所以shape维度是相乘，平铺开
    #
    #     # 在自定义观测空间：上边定义了观测空间的维度，这里定义数据范围
    #     if self.env_transform is None:  # 环境转移标志，是一个str字符串格式 ：默认是None
    #         self.observation_space = gym.spaces.Box(  # 连续，观测空间
    #             low=0.0,
    #             high=1.0 if self.normalize_observation_space else jsp_instance.max(),  # 这里表示我需要状态空间的归一化，低0高1
    #             shape=self.observation_space_shape,  # 维度
    #             dtype=self.dtype  # 数据类型
    #         )
    #     elif self.env_transform == 'mask':
    #         self.observation_space = gym.spaces.Dict(
    #             {  # 观测空间变成字典了：动作标记（low0，high1，12维度，int）+观测（low0，high1，（a，b）的二位维度，数据类型）
    #                 "action_mask": gym.spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.int32),
    #                 # eg，action_space = 12个节点
    #                 "observations": gym.spaces.Box(
    #                     low=0.0,
    #                     high=1.0 if self.normalize_observation_space else jsp_instance.max(),
    #                     shape=self.observation_space_shape,
    #                     dtype=self.dtype)
    #             })
    #     else:
    #         raise NotImplementedError(f"'{self.env_transform}' is not supported.")
    #
    #     # 下边都是画图相关的
    #     # generate colors for machines
    #     c_map = plt.cm.get_cmap(self.c_map)  # select the desired cmap   返回c_map对象，将0-1数值映射成颜色
    #     arr = np.linspace(0, 1, n_machines, dtype=self.dtype)  # create a list with numbers from 0 to 1 with n items 这个数组的作用是为每个机器分配一个数值，这个数值的范围是0到1之间的连续值
    #     self.machine_colors = {m_id: c_map(val) for m_id, val in
    #                            enumerate(arr)}  # 使用字典推导式为每个机器分配一个颜色。遍历`arr`数组中的每个元素，将其作为参数传递给颜色
    #
    #     # jsp representation as directed graph
    #     self.G = nx.DiGraph()  # G是一个有向图
    #     # 'routes' of the machines. indicates in which order a machine processes tasks   机器的“路线”。指示机器处理任务的顺序
    #     self.machine_routes = {m_id: np.array([], dtype=int) for m_id in range(n_machines)}  # 创建字典，给每个设备记录其最终的工序路径
    #
    #     # src node  源节点的属性
    #     self.G.add_node(
    #         self.src_task,  # 源节点
    #         pos=(-2, int(-n_jobs * 0.5)),  # 节点在绘图中的位置，这里的pos属性是一个元组，表示节点的x和y坐标
    #         duration=0,  # 节点处理时间
    #         machine=self.dummy_task_machine,  # 源节点机器id：-1
    #         scheduled=True,  # 是否被调度：初始肯定被调度了
    #         color=self.dummy_task_color,  # 源节点颜色
    #         job=self.dummy_task_job,  # 源节点job的id：-1
    #         start_time=0,
    #         finish_time=0
    #     )
    #
    #     # add nodes in grid format
    #     task_id = 0
    #     # machine_order = jsp_instance[0]  # 每个job中：设备的已知工序,分好了设备，都是从0开始的
    #     machine_order = np.array([
    #         [-1, -1, -1, -1],
    #         [-1, -1, -1, -1],
    #         [-1, -1, -1, -1],
    #         [-1, -1, -1, -1]])
    #     # processing_times = jsp_instance[1]
    #     processing_times = np.array([
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0]])
    #
    #     for i in range(n_jobs):
    #         for j in range(n_machines):
    #             task_id += 1  # start from task id 1, 0 is dummy starting task
    #             m_id = machine_order[i, j]  # machine id    遍历job列表中的m的id
    #
    #             dur = processing_times[i, j]  # duration of the task
    #
    #             self.G.add_node(  # 因为是循环，所以会每一个task都建立好了，从1开始的，0是源节点
    #                 task_id,
    #                 pos=(j, -i),
    #                 color=self.machine_colors[jsp_instance[0][i, j]],
    #                 duration=dur,  # 这里是我们设定的node的处理时间啊，他画图画在了有向边上
    #                 scheduled=False,
    #                 machine=m_id,
    #                 job=i,
    #                 start_time=None,  # start_time理论上应该一直在更新，根据不同的调度顺序，运输时间也不一样啊
    #                 finish_time=None
    #             )
    #             # 添加边的权重：加上我的传输时间权重
    #             # 都是添加的上一个节点的权重，理解为平行的边
    #             if j == 0:  # first task in a job    job列表中的第一个task的index，就直接添加边了啊！！！
    #                 self.G.add_edge(
    #                     self.src_task, task_id,  # 源节点和当前的job的第一个的节点连接：有向的,[self.src_task]['duration']=0
    #                     job_edge=True,
    #                     weight=self.G.nodes[self.src_task]['duration'],
    #                     nweight=-self.G.nodes[self.src_task]['duration']  # duration = 0 ，src源节点
    #                 )
    #             elif j == n_machines - 1:  # last task of a job   job列表中的最后一个task的index，就直接添加边了啊！！！
    #                 self.G.add_edge(
    #                     task_id - 1, task_id,  # job的最后一个节点和其上一个节点进行连接：有向，上一个---最后一个，填的是上一个节点的duration，确实没错
    #                     job_edge=True,
    #                     weight=self.G.nodes[task_id - 1]['duration'],
    #                     nweight=-self.G.nodes[task_id - 1]['duration']
    #                 )
    #             else:
    #                 self.G.add_edge(  # 其他情况：上一个节点连接当前节点，有向，因为按照1234.。。来的，相当于按照子任务工序约束直接建好了一个初始有向图
    #                     task_id - 1, task_id,
    #                     job_edge=True,
    #                     weight=self.G.nodes[task_id - 1]['duration'],
    #                     nweight=-self.G.nodes[task_id - 1]['duration']
    #                 )
    #     # add sink task at the end to avoid permutation in the adj matrix.   在最后添加下沉任务，避免在形容词矩阵中排列。
    #     # the rows and cols correspond to the order the nodes were added not the index/value of the node. 行和cols对应于节点添加的顺序，而不是节点的索引/值。
    #     self.G.add_node(  # 添加最后的节点
    #         self.sink_task,
    #         pos=(n_machines + 1, int(-n_jobs * 0.5)),
    #         color=self.dummy_task_color,
    #         duration=0,
    #         machine=self.dummy_task_machine,  # m_id =-1
    #         job=self.dummy_task_job,
    #         scheduled=True,
    #         start_time=None,
    #         finish_time=None
    #     )
    #     # add edges from last tasks in job to sink  添加每个job的最后一个task的边到结束节点
    #     # 起始任务编号分别为 n_machines, 2*n_machines, 3*n_machine, ... 直至 self.total_tasks
    #     for task_id in range(n_machines, self.total_tasks, n_machines):
    #         self.G.add_edge(
    #             task_id, self.sink_task,
    #             job_edge=True,
    #             weight=self.G.nodes[task_id]['duration']  # 最后一个task到sink节点的边的权重就是，最后一个子任务的处理时间（sink节点是虚拟节点啊！）
    #         )
    #
    #     """
    #     在step运行之前：
    #     load_instance里边进行：第一个reward初始变量的值
    #     都设定为0：因为还没调度怎么能有reward呢
    #     """
    #     if self.reward_function == 'wrk':
    #         initial_makespan = 0
    #         self.makespan_previous_step = initial_makespan
    #         self.total_e1_previous_step = 0
    #         self.e2_previous_step = 0
    #         self.idle_t_previous_step = 0
    #
    #     if reward_function_parameters is not None:
    #         self.reward_function_parameters = reward_function_parameters  #
    #
    #
    # """
    # 新的step
    # 用于m的选择
    # 1、每次按顺序选择一个m
    # 2、更新节点信息和边信息
    # 3、查询更新：所选的总t，总e1，运输时间e2
    # 4、计算空闲率
    # """
    # def machine_step(self, action: int) -> (np.ndarray, float, bool, dict):   #环境中的action具体怎么走？-> (np.ndarray, float, bool, dict)描述函数返回的数据类型。
    #     """
    #     perform an action on the environment. Not valid actions will have no effect.
    #
    #     :param action: an action
    #     :return: state, reward, done-flag, info-dict    返回状态+奖励+完成与否+信息字典
    #     """
    #
    #     # action = machine的索引：0，1，2，3
    #
    #     init_m_order = np.array([
    #         [-1, -1, -1, -1],
    #         [-1, -1, -1, -1],
    #         [-1, -1, -1, -1],
    #         [-1, -1, -1, -1]])
    #     init_p_times = np.array([
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0]])
    #
    #     index = 0
    #     m_order = [-1] * (self.n_jobs * self.n_machines)
    #     m_order[index] = action   # machine_order的列表形式
    #
    #     node = self.G.nodes[index + 1]  # 当前node
    #     index += 1
    #
    #     # duration = node["duration"]  # 当前节点的处理时间 （node字典的duration属性）
    #
    #     node["machine"] = action
    #     node["duration"] = self.
    #
    #     for i in range(self.n_jobs):
    #         for j in range(self.n_machines):
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #     # 记录被选择的action = taskID - 1
    #     self.selected_action.append(action)
    #
    #     info = {                 #是一个字典
    #         'action': action
    #     }
    #
    #     # add arc in graph    选完一个task之后，会有有向边的产生！
    #     # 调用_schedule_task一次，开始调度
    #     if self.action_mode == 'task':
    #         task_id = action + 1  #task_id代表了当前的子任务，是按照123按顺序来的；action是从0开始的，这里是将action+1转变成DG图上的节点的id！！！！！！！
    #         info = {
    #             **info,
    #         }
    #
    #
    #
    #
    #
    #
    #
    #     # add nodes in grid format
    #     task_id = 0
    #     # machine_order = jsp_instance[0]  # 每个job中：设备的已知工序,分好了设备，都是从0开始的
    #     machine_order = np.array([
    #         [-1, -1, -1, -1],
    #         [-1, -1, -1, -1],
    #         [-1, -1, -1, -1],
    #         [-1, -1, -1, -1]])
    #     # processing_times = jsp_instance[1]
    #     processing_times = np.array([
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0],
    #         [0, 0, 0, 0]])
    #
    #     for i in range(n_jobs):
    #         for j in range(n_machines):
    #             task_id += 1  # start from task id 1, 0 is dummy starting task
    #             m_id = machine_order[i, j]  # machine id    遍历job列表中的m的id
    #
    #             dur = processing_times[i, j]  # duration of the task
    #
    #             self.G.add_node(  # 因为是循环，所以会每一个task都建立好了，从1开始的，0是源节点
    #                 task_id,
    #                 pos=(j, -i),
    #                 color=self.machine_colors[jsp_instance[0][i, j]],
    #                 duration=dur,  # 这里是我们设定的node的处理时间啊，他画图画在了有向边上
    #                 scheduled=False,
    #                 machine=m_id,
    #                 job=i,
    #                 start_time=None,  # start_time理论上应该一直在更新，根据不同的调度顺序，运输时间也不一样啊
    #                 finish_time=None
    #             )
    #             # 添加边的权重：加上我的传输时间权重
    #             # 都是添加的上一个节点的权重，理解为平行的边
    #             if j == 0:  # first task in a job    job列表中的第一个task的index，就直接添加边了啊！！！
    #                 self.G.add_edge(
    #                     self.src_task, task_id,  # 源节点和当前的job的第一个的节点连接：有向的,[self.src_task]['duration']=0
    #                     job_edge=True,
    #                     weight=self.G.nodes[self.src_task]['duration'],
    #                     nweight=-self.G.nodes[self.src_task]['duration']  # duration = 0 ，src源节点
    #                 )
    #             elif j == n_machines - 1:  # last task of a job   job列表中的最后一个task的index，就直接添加边了啊！！！
    #                 self.G.add_edge(
    #                     task_id - 1, task_id,  # job的最后一个节点和其上一个节点进行连接：有向，上一个---最后一个，填的是上一个节点的duration，确实没错
    #                     job_edge=True,
    #                     weight=self.G.nodes[task_id - 1]['duration'],
    #                     nweight=-self.G.nodes[task_id - 1]['duration']
    #                 )
    #             else:
    #                 self.G.add_edge(  # 其他情况：上一个节点连接当前节点，有向，因为按照1234.。。来的，相当于按照子任务工序约束直接建好了一个初始有向图
    #                     task_id - 1, task_id,
    #                     job_edge=True,
    #                     weight=self.G.nodes[task_id - 1]['duration'],
    #                     nweight=-self.G.nodes[task_id - 1]['duration']
    #                 )
    #
































# def _schedule_task(self, task_id: int) -> dict:
    #     """
    #     schedules a task/node in the graph representation if the task can be scheduled.
    #     选取子任务，在甘特图schedule中进行dispatch
    #
    #     This adding one or multiple corresponding edges (multiple when performing a left shift) and updating the
    #     information stored in the nodes.
    #     添加一条边（选了方向），或者多条边（当前子任务加入插空2个任务之间，不会影响开始时间）
    #     更新node里边的信息：即该node的开始时间！！！
    #
    #     :param task_id:     the task or node that shall be scheduled.   当前是哪个节点在进行调度安排
    #     :return:            a dict with additional information for the `DisjunctiveGraphJssEnv.step`-method. 为step方法返回一个字典，储存额外的信息
    #
    #     self.G.add_node(    # 因为是循环，所以会每一个task都建立好了，从1开始的，0是源节点
    #                 task_id,
    #                 pos=(j, -i),
    #                 color=self.machine_colors[m_id],
    #                 duration=dur,         # 这里是我们设定的node的处理时间啊，他画图画在了有向边上
    #                 scheduled=False,
    #                 machine=m_id,
    #                 job=i,
    #                 start_time=None,
    #                 finish_time=None
    #             )
    #
    #
    #     """
    #     node = self.G.nodes[task_id]   #获取当前的选择的action，即+1转换成DG上对应的那一个task
    #
    #     if node["scheduled"]:      #判断是否被调度了，
    #         if self.verbose > 0:
    #             log.info(f"task {task_id} is already scheduled. ignoring it.")
    #         return {
    #             "valid_action": False,
    #             "node_id": task_id,
    #         }
    #
    #     m_id = node["machine"]   #该task也就是node的分配的machine id是什么
    #
    #     # 很常见的在有向无环图中查找当前节点的前驱节点的做法
    #     """
    #     可以认为：
    #     1、前置节点有1个，入边只有1个，那就是最开始的子任务的约束
    #     2、如果有多个入边，说明该节点被scheduled调度了，那就不会再继续该节点了。所以默认的只有一个入边，没被调度，作为前置工序
    #     3、没被调度：咋办，看下边
    #     4、当前节点的结束时间ft我没法修改加权重，因为传给谁我不知道；（so，我只能如果用到ft结束时间之类的要比较，我要加上传输时间）：node的信息，ft的部分是错的，但是用起来比较的时候没有错！！！！！！！！！！！！！！！！！！！
    #
    #     5/ finish_time就是按照定义：ft=开始+结束，不包括运输时间；这是正确的！！！开始时间才会受到运输时间的影响！！！！！！！！！！！！！！！！！！！！！！！（上一次的finish_time + 运输过来的时间 = 作为当前任务的开始时间st！！！）
    #
    #     这段代码中的 `return` 语句的作用是向生成该方法的上一级代码返回一个 dict 对象。该 dict 对象包含了两个关键字参数，即 `"valid_action"` 和 `"node_id"`。
    #     其中，`"valid_action"` 表示该操作是否是有效的。如果前一个任务节点 `prev_job_node` 还没有被调度，那么该操作将会导致图中存在环路（cycle），因此会将 `"valid_action"` 设置为 False，表示这个操作是无效的。
    #     如果该操作是无效的，那么在调用该方法的主代码块中，该方法应该返回这个无效操作的信息。如果 `self.verbose` 属性的值大于 1，那么输出一条 log 记录，
    #     描述前一个任务节点还没有被调度，然后在 `return` 语句中返回一个 dict，并将 `valid_action` 设置为 False，表示这个操作是无效的。
    #     如果该操作是有效的（即前一个任务节点已经被调度），则该方法将继续执行，插入新任务节点，并将相关信息保存在返回的 dict 对象中，以用于后续的处理和调用。
    #     """
    #     prev_task_in_job_id, _ = list(self.G.in_edges(task_id))[0] #找当前task_id节点的所有入边，只用第一条入边，取出去节点id（why只用第一条入边？？？因为没调度之前，只有其前置子任务才会指向这个节点！）
    #     prev_job_node = self.G.nodes[prev_task_in_job_id]  #按照节点id，获取上一个节点的信息，是个字典哦： 前置工序！！！！（当前节点没调度，且同一m上有调度好的节点，必为前置工序！！！！）
    #
    #     if not prev_job_node["scheduled"]:  # 没调度，输出没调度
    #         if self.verbose > 1:
    #             log.info(f"the previous task (T{prev_task_in_job_id}) in the job is not scheduled jet. "
    #                      f"Not scheduling task T{task_id} to avoid cycles in the graph.")
    #         return {
    #             "valid_action": False,   #这里的return啥意思，直接给出结果？
    #             "node_id": task_id,
    #         }
    #
    #     len_m_routes = len(self.machine_routes[m_id])     #输出当前task/node的分配的machine的已经选择了的路径的长度，为0：该m还没分配任何一个task
    #     if len_m_routes:  # 当前节点的m已经有分配了task
    #
    #         if self.perform_left_shift_if_possible:   # 确认要采用left-shift来进一步减少动作的可能组合，插空方法，不会排得很长
    #
    #             j_lower_bound_st = prev_job_node["finish_time"] # 前置节点的结束时间，作为下界;
    #             #wrk
    #             edge_init_weight = self.edge_initial_weight  # 读取初始权重
    #             edge_add_weight_need = edge_init_weight[prev_job_node['machine'], self.G.nodes[task_id]['machine']]  # 按照machine的索引来找权重: 前置工序，到当前工序
    #
    #             # 修改BUG，防止src节点的m_id=-1查表导致错误，应该是边权重为0，但是查表得到了，担心（-1，x）的情形吧
    #             # 当前节点task_id不会出现虚拟的sink节点的，总共就几个子任务就运行几遍，所以不用担心（x，-1）的情形
    #             if prev_job_node['machine'] < 0:
    #                 edge_add_weight_need = 0  # 此时没有运输时间！！！
    #             elif edge_add_weight_need == 0:
    #                 self.energy_transport[1] += edge_add_weight_need  # 同一个设备上的传递不算次数，时间累加0
    #             else:
    #                 self.energy_transport[0] += 1  # 次数加一
    #                 self.energy_transport[1] += edge_add_weight_need
    #
    #             duration = node["duration"]    #当前节点的处理时间                                     #是按照node字典的duration属性来判断的啊，还是得改duration
    #             # j_lower_bound_ft = j_lower_bound_st + duration  #前置的结束+当前的用时 = 当前节点的结束时间（即当前节点最早的开始时间！当前节点开始时间的下界）
    #             # wrk
    #             j_lower_bound_ft = j_lower_bound_st + edge_add_weight_need + duration  #前置结束（忘加了送过来运输）+ 当前持续 = 当前节点的结束时间
    #
    #             # check if task can be scheduled between src and first task
    #             m_first = self.machine_routes[m_id][0]  # m_id：当前节点分配的machine，该machine的第一个走node是哪个
    #             # first_task_on_machine_st = self.G.nodes[m_first]["start_time"]  #读出和当前节点同样用一个设备m，该设备m要走的第一个node的信息：他的开始时间！
    #             # wrk
    #             m_first_st = find_max_st_in_inedge(self.G, m_first, edge_init_weight)  #找m_first节点的所有入边，里边的最长结束时间（ft+运输t），即m_first的最大开始时间
    #             # prev_m_first_id, _ = list(self.G.in_edges(m_first))[0]   # m序列中的第一个节点（一定被调度过了），他的入边的节点id
    #             # edge_add_weight_m_first = edge_init_weight[self.G.nodes[prev_m_first_id]['machine'], self.G.nodes[m_first]['machine']]  #序列第一个节点的前序节点，到第一个节点的运输t
    #             # first_task_on_machine_st = self.G.nodes[m_first]["start_time"] + edge_add_weight_m_first  #实际：序列第一个节点的开始时间，要加上其前置的传输时间
    #
    #             # if j_lower_bound_ft <= first_task_on_machine_st:   #当前节点的完工时间ft 小于 同一个m上的第一个节点的开始时间st时：可以插入到第一节点之前开始
    #             if j_lower_bound_ft <= m_first_st:
    #                 # schedule task as first node on machine
    #                 # self.render(show=["gantt_console"])
    #                 # 在 字典machine_routes的m_id设备的array中，第一位插入task_id，基于prev_job_node的结束时间，更新node的开始/结束时间/被调度
    #                 # 当前节点task_id，用的m_id的设备，信息在node里边，prev_job_node是其前置工序！
    #                 # 当前节点放在队列首位，然后按照其前置工序更新其开始和结束时间；原先的第一个节点后移了，整体后移（因为是插入）；更新当前节点的node字典信息；返回当前节点的node信息
    #                 info = self._insert_at_index_0(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
    #
    #                 # wrk 增加边的初始权重：边：task_id---m_first，要找到他们的对应的machine的index
    #                 edge_init_weight = self.edge_initial_weight     # 读取初始权重
    #                 edge_add_weight_leftshift = edge_init_weight[self.G.nodes[task_id]['machine'], self.G.nodes[m_first]['machine']]
    #
    #                 # 修改BUG，防止src节点的m_id=-1查表导致错误，应该是边权重为0，但是查表得到了，担心（-1，x）的情形吧
    #                 # 当前节点task_id不会出现虚拟的sink节点的，总共就几个子任务就运行几遍，所以不用担心（x，-1）的情形
    #                 if self.G.nodes[task_id]['machine'] < 0:
    #                     edge_add_weight_leftshift = 0  # 此时没有运输时间！！！
    #                 elif edge_add_weight_leftshift == 0:
    #                     self.energy_transport[1] += edge_add_weight_leftshift  # 同一个设备上的传递不算次数，时间累加0
    #                 else:
    #                     self.energy_transport[0] += 1  # 次数加一
    #                     self.energy_transport[1] += edge_add_weight_leftshift
    #
    #                 # 当前节点，和，同一个m的第一个节点的，新边！！！
    #                 self.G.add_edge(     # task_id移到当前设备的任务加工顺序的首位，与下一位的m_first连接边
    #                     task_id, m_first,
    #                     job_edge=False,    # 判断是否是原先最开始画的平行边，表示同一个job的子任务先后顺序！！
    #                     # weight=duration
    #                     # wrk 增加初始权重
    #                     weight=duration + edge_add_weight_leftshift   #duration是当前节点处理时间，现在移到前边了，edge的权重就是他的处理时间+新增的
    #                 )
    #                 # self.render(show=["gantt_console"])
    #                 return info
    #             elif len_m_routes == 1:  #当前节点的完工时间ft 大于等于 同一个m上的第一个节点的开始时间st时：没必要插空，且当前m只有一个节点
    #                 return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
    #
    #             # check if task can be scheduled between two tasks
    #             # 这种插空的方式会不会导致缺少了一种可行解，虽然是有利于时间最少的，因为还有能耗呀！！！！！！！！！！！！！！！！！！！！！！！！！！！！（不同可行解会影响等待能耗吗？？？？干活能耗选定m是可算出来的，与不同可行解无关！！！！）
    #             # m_id 当前节点用的设备的索引
    #             """
    #             1、被调度好的节点，且又是在machine_route里边的，不会只有一个入边和一个出边（会有多个的入边，那就用新写的函数find_max_st_in_inedge找最长的结束时间，即可以开始当前任务的时间）
    #             2、所以可以按照入边的节点，查询到所属的machine，然后查表得运输t
    #             """
    #             for i, (m_prev, m_next) in enumerate(zip(self.machine_routes[m_id], self.machine_routes[m_id][1:])): # 遍历machine是同一个m_id的列表，枚举每一对相邻的约束任务节点:返回是节点id
    #                 m_temp_prev_ft = self.G.nodes[m_prev]["finish_time"]
    #                 m_temp_next_st = self.G.nodes[m_next]["start_time"]    #后一个节点的开始时间是不对的，因为有运输时间的；结束时间
    #
    #                 # wrk 有节点的前后，就有运输t
    #                 m_next_st = find_max_st_in_inedge(self.G, m_next, edge_init_weight)  # 找m_next节点的所有入边，里边的最长结束时间（ft+运输t），即m_first的最大开始时间
    #
    #                 # 在你开始之前，我都搞完了，我才能插空啊！
    #                 # if j_lower_bound_ft > m_temp_next_st:   #当前节点的结束时间 大于 集合中的后一个节点的开始时间
    #                 #     continue     #Python跳过当前循环的剩余语句，然后继续进行下一轮循环
    #
    #                 # m_gap = m_temp_next_st - m_temp_prev_ft
    #                 # if m_gap < duration:
    #                 #     continue
    #
    #                 # wrk
    #                 if j_lower_bound_ft > m_next_st:   #当前节点的结束时间 大于 集合中的后一个节点的开始时间：不能插空
    #                     continue     #Python跳过当前循环的剩余语句，然后继续进行下一轮循环
    #                 m_gap = m_next_st - m_temp_prev_ft
    #                 if m_gap < duration:   #两个节点之间的时间空隙 小于 当前节点的持续时间
    #                     continue    #跳过循环，下一轮
    #
    #                 # at this point the task can fit in between two already scheduled task
    #                 # self.render(show=["gantt_console"])
    #
    #                 # remove the edge from m_temp_prev to m_temp_next
    #                 replaced_edge_data = self.G.get_edge_data(m_prev, m_next)  # 获取m_prev和m_next之间的边的属性
    #
    #                 #wrk
    #                 edge_weight_prev_job = edge_init_weight[prev_job_node['machine'], self.G.nodes[task_id]['machine']]  #查表得要新增的运输时间
    #                 edge_weight_m_prev = edge_init_weight[self.G.nodes[m_prev]['machine'], self.G.nodes[task_id]['machine']]
    #
    #                 # 修改BUG，防止src节点的m_id=-1查表导致错误，应该是边权重为0，但是查表得到了，担心（-1，x）的情形吧
    #                 # 当前节点task_id不会出现虚拟的sink节点的，总共就几个子任务就运行几遍，所以不用担心（x，-1）的情形
    #                 if prev_job_node['machine'] < 0:
    #                     edge_weight_prev_job = 0  # 此时没有运输时间！！！
    #
    #                 if self.G.nodes[m_prev]['machine'] < 0:
    #                     edge_weight_m_prev = 0  # 此时没有运输时间！！！
    #                 elif edge_weight_m_prev == 0:
    #                     self.energy_transport[1] += edge_weight_m_prev  # 同一个设备上的传递不算次数，时间累加0
    #                 else:
    #                     self.energy_transport[0] += 1  # 次数加一
    #                     self.energy_transport[1] += edge_weight_m_prev
    #
    #                 # st = max(j_lower_bound_st, m_temp_prev_ft)  #前置工序的结束时间 和 序列中前一个任务的结束时间：现在要找实际开始时间，都是有一个运输t的偏差啊
    #                 #wrk
    #                 st = max((j_lower_bound_st + edge_weight_prev_job), (m_temp_prev_ft + edge_weight_m_prev))  # 入边的节点中的最大的结束时间（+了运输时间），因为有先后约束的
    #
    #                 ft = st + duration     #ft结束时间是正常的，不包括运输的
    #                 node["start_time"] = st
    #                 node["finish_time"] = ft
    #                 node["scheduled"] = True
    #
    #                 # from previous task to the task to schedule
    #                 # 原先是m_prev到m_next的边，现在中间插入了task_id：前一个有向边是原先m_prev到m_next的边的权重，后边是当前节点的权重；当然，都会加上运输t；（边权重=发出节点的duration时间）
    #                 self.G.add_edge(
    #                     m_prev, task_id,
    #                     job_edge=replaced_edge_data['job_edge'],     # 判断是否是原先最开始画的平行边，表示同一个job的子任务先后顺序！！
    #                     # weight=replaced_edge_data['weight']
    #                     # wrk
    #                     weight=replaced_edge_data['weight'] + edge_weight_m_prev  #加上运输时间
    #                 )
    #                 # from the task to schedule to the next
    #                 # wrk
    #                 edge_weight_m_next = edge_init_weight[self.G.nodes[task_id]['machine'], self.G.nodes[m_next]['machine']]
    #
    #                 # 修改BUG，防止src节点的m_id=-1查表导致错误，应该是边权重为0，但是查表得到了，担心（-1，x）的情形吧
    #                 # 当前节点task_id不会出现虚拟的sink节点的，总共就几个子任务就运行几遍，所以不用担心（x，-1）的情形
    #                 if self.G.nodes[task_id]['machine'] < 0:
    #                     edge_weight_m_next = 0  # 此时没有运输时间！！！
    #                 elif edge_weight_m_next == 0:
    #                     self.energy_transport[1] += edge_weight_m_next  # 同一个设备上的传递不算次数，时间累加0
    #                 else:
    #                     self.energy_transport[0] += 1  # 次数加一
    #                     self.energy_transport[1] += edge_weight_m_next
    #
    #                 self.G.add_edge(
    #                     task_id, m_next,
    #                     job_edge=False,
    #                     # weight=duration
    #                     # wrk
    #                     weight=duration + edge_weight_m_next #加上查表得的运输时间t
    #                 )
    #                 # remove replaced edge
    #                 self.G.remove_edge(m_prev, m_next)  #删除之前的边
    #                 # insert task at the corresponding place in the machine routes list 更新下machine_routes，因为有插空
    #                 self.machine_routes[m_id] = np.insert(self.machine_routes[m_id], i + 1, task_id)     #在_schedule_task这里会更新所有machine的已经被选择的路径！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    #
    #                 if self.verbose > 1:
    #                     log.info(f"scheduled task {task_id} on machine {m_id} between task {m_prev:.0f} "
    #                              f"and task {m_next:.0f}")
    #                 # self.render(show=["gantt_console"])
    #                 return {                    #返回节点（子任务的）的开始和结束时间（甘特图就看子任务先后顺序，和同一个m上的先后顺序，先后用时间来表示：gantt上边的抽象时间）
    #                     "start_time": st,
    #                     "finish_time": ft,
    #                     "node_id": task_id,
    #                     "valid_action": True,
    #                     "scheduling_method": 'left_shift',
    #                     "left_shift": 1,
    #                 }
    #             else:
    #                 return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
    #
    #         else:
    #             return self._append_at_the_end(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
    #
    #     else:   #  当前节点的machin_route为None，这个当前节点直接加入序列中
    #         return self._insert_at_index_0(task_id=task_id, node=node, prev_job_node=prev_job_node, m_id=m_id)
    #
    # # 牵扯了我如果选了某个子任务，它的无向边怎么变化？都是画图相关
    # # 那个left_shift会不会无形之中减少了某一类可行解？？？？（不只看时间的）
    # def _append_at_the_end(self, task_id: int, node: dict, prev_job_node: dict, m_id: int) -> dict:
    #     """
    #     inserts a task at the end (last element) in the `DisjunctiveGraphJssEnv.machine_routes`-dictionary.
    #     在' DisjunctiveGraphJssEnv.machine_routes '字典的最后一个元素插入一个任务。
    #
    #     :param task_id:             the id oth the task with in graph representation.
    #     :param node:                the corresponding node in the graph (self.G).
    #     :param prev_job_node:       the node the is connected to :param node: via a job_edge (job_edge=True).
    #     :param m_id:                the id of the machine that corresponds to :param task_id:.
    #     :return:                    a dict with additional information for the `DisjunctiveGraphJssEnv.step`-method.
    #
    #     这段代码实现的是在已分配机器 route 列表的末尾（即最后一个元素）添加任务节点的 _append_at_the_end 方法
    #     """
    #     prev_m_task = self.machine_routes[m_id][-1]   # 当前m_id设备的最后一个子任务的id
    #     prev_m_node = self.G.nodes[prev_m_task]  #最后一个子任务的node信息
    #
    #     # wrk 增加边的权重
    #     edge_init_weight = self.edge_initial_weight  # 读取初始权重
    #     edge_add_weight_append_end = edge_init_weight[self.G.nodes[prev_m_task]['machine'], self.G.nodes[task_id]['machine']]  #按照machine的索引来找权重
    #
    #     # 修改BUG，防止src节点的m_id=-1查表导致错误，应该是边权重为0，但是查表得到了，担心（-1，x）的情形吧
    #     # 当前节点task_id不会出现虚拟的sink节点的，总共就几个子任务就运行几遍，所以不用担心（x，-1）的情形
    #     if self.G.nodes[prev_m_task]['machine'] < 0:
    #         edge_add_weight_append_end = 0  # 此时没有运输时间！！！
    #     elif edge_add_weight_append_end == 0:
    #         self.energy_transport[1] += edge_add_weight_append_end  # 同一个设备上的传递不算次数，时间累加0
    #     else:
    #         self.energy_transport[0] += 1  # 次数加一
    #         self.energy_transport[1] += edge_add_weight_append_end
    #
    #     self.G.add_edge(
    #         prev_m_task, task_id,    # 添加原先最后一个子任务prev_m_task，到，当前任务task_id的，边(权重都是上一个节点的权重)
    #         job_edge=False,
    #         # weight=prev_m_node['duration']
    #         # wrk 修改权重
    #         weight=prev_m_node['duration'] + edge_add_weight_append_end   #查表增加初始权重
    #     )
    #     self.machine_routes[m_id] = np.append(self.machine_routes[m_id], task_id)  #将任务ID task_id 插入到当前任务用的m的序列中
    #
    #     # 前置任务：分配好的 vs 列表最后一个任务，要把当前插入到此后边
    #     # 所以比较，谁的结束时间最长，当前就从哪里开始！！！
    #     # 考虑加上传输时间了，都是上一个（有两个，一个前置工序，一个同m的最后一个任务）传到当前的运输时间
    #     prev_job_node_transport_t = edge_init_weight[prev_job_node['machine'], self.G.nodes[task_id]['machine']]  # 按照machine的索引来找权重：前置工序，传到当前的传输时间
    #     prev_m_node_transport_t = edge_init_weight[prev_m_node['machine'], self.G.nodes[task_id]['machine']] # 按照machine的索引来找权重：同m的最后一个任务，传到当前的传输时间（插到最后，所以也是上一个任务）
    #
    #     # 修改BUG，防止src节点的m_id=-1查表导致错误，应该是边权重为0，但是查表得到了，担心（-1，x）的情形吧
    #     # 当前节点task_id不会出现虚拟的sink节点的，总共就几个子任务就运行几遍，所以不用担心（x，-1）的情形
    #     if prev_job_node['machine'] < 0:
    #         prev_job_node_transport_t = 0  # 此时没有运输时间！！！
    #     if prev_m_node['machine'] < 0:
    #         prev_m_node_transport_t = 0  # 此时没有运输时间！！！
    #
    #     st = max((prev_job_node["finish_time"]+prev_job_node_transport_t), (prev_m_node["finish_time"]+prev_m_node_transport_t))   #prev_job_node前置的任务，同m的列表的最后一个任务,谁大就排在谁后边！（约束摆着呢）
    #     ft = st + node["duration"]   #结束 = 开始 +持续 ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？当前不知道传给谁，怎么加传输时间呢？？？？？？？
    #     node["start_time"] = st
    #     node["finish_time"] = ft
    #     node["scheduled"] = True
    #     # return additional info
    #     return {
    #         "start_time": st,
    #         "finish_time": ft,
    #         "node_id": task_id,
    #         "valid_action": True,
    #         "scheduling_method": '_append_at_the_end',
    #         "left_shift": 0,
    #     }
    #
    # def _insert_at_index_0(self, task_id: int, node: dict, prev_job_node: dict, m_id: int) -> dict:
    #     """
    #     inserts a task at index 0 (first element) in the `DisjunctiveGraphJssEnv.machine_routes`-dictionary.
    #
    #     :param task_id:             the id oth the task with in graph representation.
    #     :param node:                the corresponding node in the graph (self.G).
    #     :param prev_job_node:       the node the is connected to :param node: via a job_edge (job_edge=True).    job_edge理解为子任务约束的边吗，固有存在的
    #     :param m_id:                the id of the machine that corresponds to :param task_id:.
    #     :return:                    a dict with additional information for the `DisjunctiveGraphJssEnv.step`-method.
    #     """
    #     self.machine_routes[m_id] = np.insert(self.machine_routes[m_id], 0, task_id)   # 在 字典machine_routes的m_id设备的array中，第一位插入task_id
    #
    #     # wrk 计算当前开始时间，需要更新上一步的结束时间（加上权重）
    #     edge_init_weight = self.edge_initial_weight  # 读取初始权重
    #     prev_job_node_transport_t = edge_init_weight[prev_job_node['machine'], self.G.nodes[task_id]['machine']]  # 按照machine的索引来找权重：前置工序，传到当前的传输时间
    #
    #     # 修改BUG，防止src节点的m_id=-1查表导致错误，应该是边权重为0，但是查表得到了，担心（-1，x）的情形吧
    #     # 当前节点task_id不会出现虚拟的sink节点的，总共就几个子任务就运行几遍，所以不用担心（x，-1）的情形
    #     if prev_job_node['machine'] < 0:
    #         prev_job_node_transport_t = 0  #此时没有运输时间！！！
    #
    #     st = prev_job_node["finish_time"] + prev_job_node_transport_t  #当前的开始时间 = 上一个节点的结束时间 + 前置节点运输消耗的时间
    #     ft = st + node["duration"]    #结束时间 = 开始时间 + 当前节点的持续时间，不用知道传给谁，反正任务已经在此节点结束了！！！
    #     node["start_time"] = st     # 更新node的各种信息
    #     node["finish_time"] = ft
    #     node["scheduled"] = True
    #     # return additional info
    #     return {                 #并返回信息
    #         "start_time": st,
    #         "finish_time": ft,
    #         "node_id": task_id,
    #         "valid_action": True,
    #         "scheduling_method": '_insert_at_index_0',
    #         "left_shift": 0,
    #     }