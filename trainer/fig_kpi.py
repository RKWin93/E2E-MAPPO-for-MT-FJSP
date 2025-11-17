import os

import psutil
import pynvml
import torch


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import random

from matplotlib.patches import Rectangle

# import matplotlib.pyplot as plt
# import numpy as np

#  ! TODO 250425-暂时没有修改这里的configs，因为画图相关，不是很重要，改用wandb！等用到了再修改！
# from parameters import configs
# todo 1224-为了实现多进程运行程序，同时可以修改参数，在每一个使用configs的文件增加下边语句： from parameters import parser + configs = parser.parse_args()
# from parameters import parser
# configs = parser.parse_args()


rows = 3  # 画布的行列！
cols = 4
# title = ['net1 policy-loss','net1 value-loss', 'net1 mEntropy-loss','?',
#          'net2 policy-loss','net2 value-loss','net2 dEntropy-loss','?',
#          'net1 real energy-transM ','net1 real energy-M','net1 real machine-Ratio in-R*100','?',
#          'net2 real energy-transM ','net2 real energy-M  in-R/10','net2 real Makespan', 'net2 real Idle-T']
# title = ['policy-loss','value-loss', 'Entropy-loss','Total-loss',
#          'net2 Makespan ','net1 Energy Consumption','net1 machine-Ratio','Total Reward',
#          'net1 Energy-transM ','net2 Idle Time','Actor Grad Avg', 'Actor Grad Std']
# title = ['policy-loss','value-loss', 'Entropy-loss','Total-loss',
#          'net2 Makespan-BatchAvg ', 'total Energy Consumption-BatchAvg','Total Accumulative Reward-BatchAvg', "Validate Accumulative Reward",
#          'net1 Energy-Machine Processing-BatchAvg', 'net1 Energy-transM-BatchAvg ','net2 Idle Time-BatchAvg','Actor Grad Avg-BatchAvg']
title = ['MachineActor-loss','JobActor-loss', 'Critic-loss','Total-loss',
         'net1-p*t+transT*1-Gt ', 'net2-makespan+idleT*1-Gt','Eval-Cost',
         "net2-Makespan-no weight",
         'net1-OperationEnergy-no weight', 'net1-TransT*1-no weight ','net2-IdleT*1-no weight','net1-UseRatio-no weight']
def ppo_result_fig(list, n_job, n_machine, n_edge, episode, show_on=True):
    gs = gridspec.GridSpec(rows, cols)  # 3 rows, 2 columns
    fig = plt.figure(figsize=(24, 16))
    fig_title = configs.fig_mask + "PPO_trained_J%sM%sE%s" % (n_job, n_machine, n_edge)
    fig.suptitle(fig_title)
    index = 0
    for i in range(rows):
        for j in range(cols):
            sub = plt.subplot(gs[i,j])
            plt.title(title[index])

            # for k in range(len(list[index])):
            # sub.plot(list[index], label='raw')
            sub.plot(list[index])

            # 平滑曲线
            if len(list[index]) > 0:  # 不为空集，才会平滑
                alpha = 0.1
                res = [list[index][0]]  # 初始化EMA值
                for k in range(1, len(list[index])):
                    ema = alpha * list[index][k] + (1 - alpha) * res[-1]
                    res.append(ema)
                list_show = res
                sub.plot(list_show, label=f'raw:{title[index][0:4]}') # 画出折线图

            if len(list[index]) > 0:
                # if type(list[index]) is np.ndarray:  # 此时，list画的一张图中有n条线！
                #     if list[index].ndim > 1:
                #         pass
                # else:
                    # 计算平均值、最大值和最小值
                    mean_y = np.mean(list[index])
                    max_y = np.max(list[index])
                    min_y = np.min(list[index])

                    # 画出平均值、最大值和最小值
                    plt.axhline(mean_y, linestyle='--', color='c', label=f'Mean ({mean_y:.2f})')
                    plt.axhline(max_y, linestyle='--', color='r', label=f'Max ({max_y:.2f})')
                    plt.axhline(min_y, linestyle='--', color='b', label=f'Min ({min_y:.2f})')

                    # 构建x序列
                    x = range(0, len(list[index]) + 1)

                    # # 画出最后一个点
                    last_x, last_y = x[-1], list[index][-1]
                    plt.plot(last_x, last_y, 'ro', label=f'Last Point ({last_x}, {last_y})')
                    # 画出最后一个点的值的直线
                    plt.axhline(last_y, linestyle='--', color='m', label=f'Last Point Value ({last_y})')
                    # # 画出第一个点
                    first_x, first_y = x[0], list[index][0]
                    plt.plot(first_x, first_y, 'go', label=f'First Point ({first_x}, {first_y})')
                    # 画出第一个点的值的直线
                    plt.axhline(first_y, linestyle='--', color='g', label=f'First Point Value ({first_y})')

            # 保存图形
            if show_on:  # 表明是在代码最后保存图
                fig_file = "./fig/PPO_trained_J%sM%sE%s_Episode%s_all.png" % (n_job, n_machine, n_edge, (episode+1))
                # plt.savefig(fig_file)
            else:
                fig_file = "./train_figure/PPO_trained_J%sM%sE%s_Episode%s.png" % (n_job, n_machine, n_edge, (episode+1))
                # plt.savefig(fig_file)
            plt.savefig(fig_file)
            # plt.savefig('./fig/result.png')
            plt.grid()  # 添加网格
            plt.legend()  # 添加图例
            index += 1


    gs.update(hspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.05)

    if show_on:
        plt.show()
    else:
        plt.close(fig)  # 只保存，不画图


def esa_ppo_result_fig(list, n_job, n_machine, n_edge, episode, show_on=True):
    gs = gridspec.GridSpec(rows, cols)  # 3 rows, 2 columns
    fig = plt.figure(figsize=(24, 16))
    fig_title = configs.fig_mask + "esa_PPO_trained_J%sM%sE%s" % (n_job, n_machine, n_edge)
    fig.suptitle(fig_title)
    index = 0
    for i in range(rows):
        for j in range(cols):
            sub = plt.subplot(gs[i,j])
            plt.title(title[index])

            # for k in range(len(list[index])):
            # sub.plot(list[index], label='raw')
            sub.plot(list[index])

            # 平滑曲线
            if len(list[index]) > 0:  # 不为空集，才会平滑
                alpha = 0.1
                res = [list[index][0]]  # 初始化EMA值
                for k in range(1, len(list[index])):
                    ema = alpha * list[index][k] + (1 - alpha) * res[-1]
                    res.append(ema)
                list_show = res
                sub.plot(list_show, label=f'raw:{title[index][0:4]}') # 画出折线图

            if len(list[index]) > 0:
                # if type(list[index]) is np.ndarray:  # 此时，list画的一张图中有n条线！
                #     if list[index].ndim > 1:
                #         pass
                # else:
                    # 计算平均值、最大值和最小值
                    mean_y = np.mean(list[index])
                    max_y = np.max(list[index])
                    min_y = np.min(list[index])

                    # 画出平均值、最大值和最小值
                    plt.axhline(mean_y, linestyle='--', color='c', label=f'Mean ({mean_y:.2f})')
                    plt.axhline(max_y, linestyle='--', color='r', label=f'Max ({max_y:.2f})')
                    plt.axhline(min_y, linestyle='--', color='b', label=f'Min ({min_y:.2f})')

                    # 构建x序列
                    x = range(0, len(list[index]) + 1)

                    # # 画出最后一个点
                    last_x, last_y = x[-1], list[index][-1]
                    plt.plot(last_x, last_y, 'ro', label=f'Last Point ({last_x}, {last_y})')
                    # 画出最后一个点的值的直线
                    plt.axhline(last_y, linestyle='--', color='m', label=f'Last Point Value ({last_y})')
                    # # 画出第一个点
                    first_x, first_y = x[0], list[index][0]
                    plt.plot(first_x, first_y, 'go', label=f'First Point ({first_x}, {first_y})')
                    # 画出第一个点的值的直线
                    plt.axhline(first_y, linestyle='--', color='g', label=f'First Point Value ({first_y})')

            # 保存图形
            if show_on:  # 表明是在代码最后保存图
                fig_file = "./fig/esa_PPO_trained_J%sM%sE%s_Episode%s_all.png" % (n_job, n_machine, n_edge, (episode+1))
                # plt.savefig(fig_file)
            else:
                fig_file = "./train_figure/esa_PPO_trained_J%sM%sE%s_Episode%s.png" % (n_job, n_machine, n_edge, (episode+1))
                # plt.savefig(fig_file)
            plt.savefig(fig_file)
            # plt.savefig('./fig/result.png')
            plt.grid()  # 添加网格
            plt.legend()  # 添加图例
            index += 1


    gs.update(hspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.05)

    if show_on:
        plt.show()
    else:
        plt.close(fig)  # 只保存，不画图



def r_loss_fig(list):
    gs = gridspec.GridSpec(rows, cols)  # 3 rows, 2 columns
    plt.figure(figsize=(24, 16))
    index = 0
    for i in range(rows):
        for j in range(cols):
            sub = plt.subplot(gs[i,j])
            plt.title(title[index])

            # for k in range(len(list[index])):
            # sub.plot(list[index], label='raw')
            sub.plot(list[index])

            # 平滑曲线
            if len(list[index]) > 0:  # 不为空集，才会平滑
                alpha = 0.1
                res = [list[index][0]]  # 初始化EMA值
                for k in range(1, len(list[index])):
                    ema = alpha * list[index][k] + (1 - alpha) * res[-1]
                    res.append(ema)
                list_show = res
                sub.plot(list_show, label=f'raw:{title[index][0:4]}') # 画出折线图

            if len(list[index]) > 0:
                # 计算平均值、最大值和最小值
                mean_y = np.mean(list[index])
                max_y = np.max(list[index])
                min_y = np.min(list[index])

                # 画出平均值、最大值和最小值
                plt.axhline(mean_y, linestyle='--', color='c', label=f'Mean ({mean_y:.2f})')
                plt.axhline(max_y, linestyle='--', color='r', label=f'Max ({max_y:.2f})')
                plt.axhline(min_y, linestyle='--', color='b', label=f'Min ({min_y:.2f})')

                # 构建x序列
                x = range(0, len(list[index]) + 1)

                # # 画出最后一个点
                last_x, last_y = x[-1], list[index][-1]
                plt.plot(last_x, last_y, 'ro', label=f'Last Point ({last_x}, {last_y})')
                # 画出最后一个点的值的直线
                plt.axhline(last_y, linestyle='--', color='m', label=f'Last Point Value ({last_y})')
                # # 画出第一个点
                first_x, first_y = x[0], list[index][0]
                plt.plot(first_x, first_y, 'go', label=f'First Point ({first_x}, {first_y})')
                # 画出第一个点的值的直线
                plt.axhline(first_y, linestyle='--', color='g', label=f'First Point Value ({first_y})')

            # 保存图形
            plt.savefig('./fig/result.png')
            plt.grid()  # 添加网格
            plt.legend()  # 添加图例
            index += 1


    gs.update(hspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.show()






# rows1 = 1
# cols1 = 3
# box_title = ['Makespan','Energy Consumption', 'Accumulative Reward',
#              'Consuming Time','?','?']
# box_title = ['Net1-PT-TransT-No weight','Net2-Makespan-IdleT-No weight', 'Accumulative Reward-weighted',
#              'Consuming Time','?','?']
# box_title = ['pt-no weight','makespan-no weight','transT-no weight',
#              'tdleT-no weight','untilNow-done step-cost- no weight','cost-weighted']
box_title = ['pt norm-no weight','makespan norm-no weight','transT norm-no weight',
             'tdleT norm-no weight','untilNow-done step-cost-weighted','cost norm-weighted']
PDRs_label = ['SPT+FIFO', 'SPT+MOR', 'SPT+LWKR_T', 'SPT+LWKR_PT', 'SPT+LWKR_IT', 'SPT+LWKR_TT',
              'SEC+FIFO', 'SEC+MOR', 'SEC+LWKR_T', 'SEC+LWKR_PT', 'SEC+LWKR_IT', 'SEC+LWKR_TT',
              'MISE+FIFO', 'MISE+MOR', 'MISE+LWKR_T', 'MISE+LWKR_PT', 'MISE+LWKR_IT', 'MISE+LWKR_TT',
              'AMU+FIFO', 'AMU+MOR', 'AMU+LWKR_T', 'AMU+LWKR_PT', 'AMU+LWKR_IT', 'AMU+LWKR_TT',
              'MIP_MO_FJSP', 'PPO']
# PDRs_label_jointActor_old = ['FIFO+SPT', 'FIFO+SEC', 'FIFO+Random',
#                          'MOR+SPT', 'MOR+SEC', 'MOR+Random',
#                          'LWKR_T+SPT', 'LWKR_T+SEC', 'LWKR_T+Random',
#                          'LWKR_PT+SPT', 'LWKR_PT+SEC', 'LWKR_PT+Random',
#                          'MWKR_T+SPT',  'MWKR_T+SEC', 'MWKR_T+SPT',
#                          'MWKR_PT+SPT', 'MWKR_PT+SEC', 'MWKR_PT+Random',
#                          'MIP_MO_FJSP', 'PPO']
PDRs_label_jointActor_old = ['FIFO+SPT', 'FIFO+SEC',
                         'MOR+SPT', 'MOR+SEC',
                         'LWKR_T+SPT', 'LWKR_T+SEC',
                         'LWKR_PT+SPT', 'LWKR_PT+SEC',
                         'MWKR_T+SPT',  'MWKR_T+SEC',
                         'MWKR_PT+SPT', 'MWKR_PT+SEC',
                         'RA+RA','MIP_MO_FJSP', 'ESA-G', 'PPO-G','PPO-S']
PDRs_label_jointActor = ['FIFO+SPT', 'FIFO+SEC', 'FIFO+RA',
                         'MOR+SPT', 'MOR+SEC', 'MOR+RA',
                         'RA+SPT', 'RA+SEC', 'RA+RA',
                         'MIP_MO_FJSP', 'PPO']
def result_box_plot(list, rows1, cols1):
    gs = gridspec.GridSpec(rows1, cols1)  # 3 rows, 2 columns todo 原始的画箱形图
    plt.figure(figsize=(32, 24)) # todo 原始的画箱形图

    # fig, axes = plt.subplots(rows1, cols1, figsize=(24, 24))
    index = 0
    for i in range(rows1):
        for j in range(cols1):
            sub = plt.subplot(gs[i,j])  # todo 原始的画箱形图
            plt.title(box_title[index])   # todo 原始的画箱形图

            # for k in range(len(list[index])):
            # sub.plot(list[index], label='raw')
            # sub.boxplot(list[index],labels=PDRs_label)
            # 创建一个矩形对象，用于填充箱体的颜色
            # box_rect = Rectangle((0, np.min(list[index])), 1, np.max(list[index]) - np.min(list[index]), facecolor='blue')

            # 添加矩形对象到图形中
            # sub.add_patch(box_rect)
            # 设置须线属性
            # whisker_props = dict(linestyle='-', linewidth=2)  # 我使用`whiskerprops`参数将须线的样式设置为实线，宽度设置为2
            # sub.boxplot(list[index], whiskerprops=whisker_props)
            # sub.boxplot(list[index])  # todo 原始的画箱形图

            # 绘制箱型图，并设置线条和填充的颜色和粗细
            boxplot = plt.boxplot(list[index], patch_artist=False, boxprops={'color': 'black', 'linewidth': 2}, # TRUE可以使箱体填充颜色， 设置箱体的属性，不填充就是边缘线
                                  whiskerprops={'color': 'black', 'linewidth': 2}, # 用于设置须的属性：上下竖线
                                  medianprops={'color': 'red', 'linewidth': 3}, # 设置中位线的属性
                                  capprops={'color': 'red', 'linewidth': 3}, # 设置边缘线的属性, 上下封顶的
                                  flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 8} # 设置异常值的属性
                                  )


            # sub.set_xticks(range(0, 19)) #  set_xticks() 方法的参数也设置为相应的位置
            if len(list[index]) > 0:
                # sub.set_xticklabels(PDRs_label, rotation=-90)  # 使用 set_xticklabels() 方法设置了坐标轴标签
                sub.set_xticklabels(PDRs_label_jointActor, rotation=-90)  # 使用 set_xticklabels() 方法设置了坐标轴标签 =  `rotation=-90`表示将标签逆时针旋转90度，使其垂直显示


            # 调整子图大小
            # sub.figure.set_size_inches(8, 6)  # 设置子图的宽度为8英寸，高度为6英寸
            # 保存图形
            # plt.savefig('./fig/test_box.png')
            plt.grid()  # 添加网格
            # plt.legend()  # 添加图例
            index += 1


    # gs.update(hspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.05)
    # plt.savefig('./fig/result_box.png')
    # plt.grid()  # 添加网格
    plt.savefig('/remote-home/iot_wangrongkai/RUN/fig/test_result_box.png')
    plt.show()

def result_box_plot_eachEpisode(list, rows1, cols1, epi, show_on=False):
    gs = gridspec.GridSpec(rows1, cols1)  # 3 rows, 2 columns todo 原始的画箱形图
    plt.figure(figsize=(32, 24)) # todo 原始的画箱形图

    # fig, axes = plt.subplots(rows1, cols1, figsize=(24, 24))
    index = 0
    for i in range(rows1):
        for j in range(cols1):
            sub = plt.subplot(gs[i,j])  # todo 原始的画箱形图
            plt.title(box_title[index])   # todo 原始的画箱形图

            # for k in range(len(list[index])):
            # sub.plot(list[index], label='raw')
            # sub.boxplot(list[index],labels=PDRs_label)
            # 创建一个矩形对象，用于填充箱体的颜色
            # box_rect = Rectangle((0, np.min(list[index])), 1, np.max(list[index]) - np.min(list[index]), facecolor='blue')

            # 添加矩形对象到图形中
            # sub.add_patch(box_rect)
            # 设置须线属性
            # whisker_props = dict(linestyle='-', linewidth=2)  # 我使用`whiskerprops`参数将须线的样式设置为实线，宽度设置为2
            # sub.boxplot(list[index], whiskerprops=whisker_props)
            # sub.boxplot(list[index])  # todo 原始的画箱形图

            # 绘制箱型图，并设置线条和填充的颜色和粗细
            boxplot = plt.boxplot(list[index], patch_artist=False, boxprops={'color': 'black', 'linewidth': 2}, # TRUE可以使箱体填充颜色， 设置箱体的属性，不填充就是边缘线
                                  whiskerprops={'color': 'black', 'linewidth': 2}, # 用于设置须的属性：上下竖线
                                  medianprops={'color': 'red', 'linewidth': 3}, # 设置中位线的属性
                                  capprops={'color': 'red', 'linewidth': 3}, # 设置边缘线的属性, 上下封顶的
                                  flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 8} # 设置异常值的属性
                                  )


            # sub.set_xticks(range(0, 19)) #  set_xticks() 方法的参数也设置为相应的位置
            if len(list[index]) > 0:
                # sub.set_xticklabels(PDRs_label, rotation=-90)  # 使用 set_xticklabels() 方法设置了坐标轴标签
                sub.set_xticklabels(PDRs_label_jointActor, rotation=-90)  # 使用 set_xticklabels() 方法设置了坐标轴标签 =  `rotation=-90`表示将标签逆时针旋转90度，使其垂直显示


            # 调整子图大小
            # sub.figure.set_size_inches(8, 6)  # 设置子图的宽度为8英寸，高度为6英寸
            # 保存图形
            # plt.savefig('./fig/test_box.png')
            plt.grid()  # 添加网格
            # plt.legend()  # 添加图例
            index += 1


    # gs.update(hspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.05)
    # plt.savefig('./fig/result_box.png')
    # plt.grid()  # 添加网格

    # 保存图形
    if show_on:  # 表明是在代码最后保存图
        fig_file = '/remote-home/iot_wangrongkai/RUN/fig/test_result_box_J%sM%sE%s_Epi%s.png'\
                   % (configs.n_job, configs.n_machine, configs.n_edge, (epi + 1))
    else:
        fig_file = "/remote-home/iot_wangrongkai/RUN/Test-fig/test_J%sM%sE%s_Episode%s_all.png" \
                   % (configs.n_job, configs.n_machine, configs.n_edge, (epi + 1))

    # plt.savefig('/remote-home/iot_wangrongkai/RUN/fig/test_result_box.png')
    plt.savefig(fig_file)

    if show_on:
        plt.show()
    else:
        plt.close()


"""
画箱型图
每组数据，传进来好几个大lst，其中每个大lst包含所有方法的小lst，每个小lst里边存着当前执行过的样例
[[1,2,...],[2,3,...],...,[2,4,...]] * N(画什么)
"""
def result_box_plot_eachEpisode_1217(args, list, rows1, cols1, epi, title_name, fig_pth, method_name, show_on=False):   # 不改变原有的代码，新的画图程序，画的更多PDRs
    gs = gridspec.GridSpec(rows1, cols1)  # 3 rows, 2 columns todo 原始的画箱形图
    plt.figure(figsize=(42, 40)) # todo 原始的画箱形图

    # fig, axes = plt.subplots(rows1, cols1, figsize=(24, 24))
    index = 0
    for i in range(rows1):
        for j in range(cols1):
            sub = plt.subplot(gs[i,j])  # todo 原始的画箱形图
            plt.title(title_name[index])   # todo 原始的画箱形图
            # plt.title(box_title[index])   # todo 原始的画箱形图

            # for k in range(len(list[index])):
            # sub.plot(list[index], label='raw')
            # sub.boxplot(list[index],labels=PDRs_label)
            # 创建一个矩形对象，用于填充箱体的颜色
            # box_rect = Rectangle((0, np.min(list[index])), 1, np.max(list[index]) - np.min(list[index]), facecolor='blue')

            # 添加矩形对象到图形中
            # sub.add_patch(box_rect)
            # 设置须线属性
            # whisker_props = dict(linestyle='-', linewidth=2)  # 我使用`whiskerprops`参数将须线的样式设置为实线，宽度设置为2
            # sub.boxplot(list[index], whiskerprops=whisker_props)
            # sub.boxplot(list[index])  # todo 原始的画箱形图

            # 绘制箱型图，并设置线条和填充的颜色和粗细
            boxplot = plt.boxplot(list[index], patch_artist=False, boxprops={'color': 'black', 'linewidth': 2}, # TRUE可以使箱体填充颜色， 设置箱体的属性，不填充就是边缘线
                                  whiskerprops={'color': 'black', 'linewidth': 2}, # 用于设置须的属性：上下竖线
                                  medianprops={'color': 'red', 'linewidth': 3}, # 设置中位线的属性
                                  capprops={'color': 'red', 'linewidth': 3}, # 设置边缘线的属性, 上下封顶的
                                  flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 8} # 设置异常值的属性
                                  )


            # sub.set_xticks(range(0, 19)) #  set_xticks() 方法的参数也设置为相应的位置
            if len(list[index]) > 0:
                # sub.set_xticklabels(PDRs_label, rotation=-90)  # 使用 set_xticklabels() 方法设置了坐标轴标签
                sub.set_xticklabels(method_name, rotation=-90)  # 使用 set_xticklabels() 方法设置了坐标轴标签 =  `rotation=-90`表示将标签逆时针旋转90度，使其垂直显示


            # 调整子图大小
            # sub.figure.set_size_inches(8, 6)  # 设置子图的宽度为8英寸，高度为6英寸
            # 保存图形
            # plt.savefig('./fig/test_box.png')
            plt.grid()  # 添加网格
            # plt.legend()  # 添加图例
            index += 1


    # gs.update(hspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.05)
    # plt.savefig('./fig/result_box.png')
    # plt.grid()  # 添加网格
    # todo 1225-画图的保存位置，重新换下！直接改下列的路径即可
    # save_pth = "/remote-home/iot_wangrongkai/RUN/continuous_model/1225-all-pth-newMasking/"
    fig_file = fig_pth + 'Test_result_box_J%sM%sE%s_Epi%s.png'\
                   % (args['n_job'], args['n_machine'], args['n_edge'], (epi + 1))

    # 保存图形
    # if show_on:  # 表明是在代码最后保存图，最后一次的结果图
    #     fig_file = fig_pth + 'Test_result_box_J%sM%sE%s_Epi%s.png'\
    #                % (args['n_job'], args['n_machine'], args['n_edge'], (epi + 1))
    # else: # 记录每一次测试的结果图
    #     fig_file = fig_pth + "Test_J%sM%sE%s_Episode%s_all.png" \
    #                % (args['n_job'], args['n_machine'], args['n_edge'], (epi + 1))

    # plt.savefig('/remote-home/iot_wangrongkai/RUN/fig/test_result_box.png')
    plt.savefig(fig_file)

    if show_on:
        plt.show()
    else:
        plt.close()







"""
期望在训练过程中随时画出fig
1、传入的都是list列表数据
"""
def plot_show(da_loss,da_loss_operation,dv_loss,rewards_sum,rewards_sum_operation,vali_cost_lst,
              totalCost_t,net1_totalCost_e,net1_totalCost_trans,totalCost_idle,net1_totalCost_ratio,
              n_job, n_machine, n_edge, episode,show_on=False):
    list_fig = []  # 要在画布上子图上循环画图的数据lst

    if da_loss:  # 有些episode太少了，还没有update，防止是空集合
        # 还需要从gpu转到cpu，然后detach去掉梯度，才能转成numpy
        da_loss, da_loss_operation = torch.stack(da_loss).cpu().detach().numpy(), \
            torch.stack(da_loss_operation).cpu().detach().numpy()  # 先堆叠tensor，list转成大tensor，再转array，画图用
        if dv_loss:
            dv_loss = torch.stack(dv_loss).cpu().detach().numpy()
            # d_loss = np.array(dv_loss) + np.array(da_loss) + np.array(dEntropy_loss)  # 把权重比例加上去
            d_loss = da_loss + da_loss_operation + dv_loss  # 总的loss(numpy数组相加元素)，两个loss列表每次都扩充，直接相加即可，把权重比例加上去
        else:
            d_loss = da_loss + da_loss_operation   # 总的loss(numpy数组相加元素)，两个loss列表每次都扩充，直接相加即可，把权重比例加上去

    list_fig.append(da_loss)  # actor的loss，因为我loss=数值就加了负号，这里画出来应该下降曲线，不是就有问题？？？（loss之前累加了，我的错！）
    list_fig.append(da_loss_operation)  # actor的loss，因为我loss=数值就加了负号，这里画出来应该下降曲线，不是就有问题？？？（loss之前累加了，我的错！）
    if len(dv_loss) != 0: # 存在任意一个数？
        list_fig.append(dv_loss)  # value的loss
    else:
        list_fig.append([])
    list_fig.append(d_loss)  # 总loss

    # if a_grad_norm:
    #     a_grad_norm, a_grad_norm_operation, v_grad_norm = torch.stack(a_grad_norm).cpu().detach().numpy(), \
    #         torch.stack(a_grad_norm_operation).cpu().detach().numpy(), \
    #         torch.stack(v_grad_norm).cpu().detach().numpy()  # list中tensor元素转成numpy。画图用

    list_fig.append(rewards_sum)  # 总r
    list_fig.append(rewards_sum_operation)  # 总r
    list_fig.append(vali_cost_lst)  # 验证数据的总的累积奖励 = -cost

    list_fig.append(totalCost_t)  # 总加工时间mk
    list_fig.append(net1_totalCost_e)  # 总加工能耗 = t=1 * e1
    list_fig.append(net1_totalCost_trans)  # m中运输t*1
    list_fig.append(totalCost_idle)  # idle t
    list_fig.append(net1_totalCost_ratio)  # 设备利用率，ratio

    ppo_result_fig(list_fig, n_job, n_machine, n_edge, episode, show_on=show_on)  # 只保存图片


def esa_plot_show(da_loss,da_loss_operation,dv_loss,rewards_sum,rewards_sum_operation,vali_cost_lst,
              totalCost_t,net1_totalCost_e,net1_totalCost_trans,totalCost_idle,net1_totalCost_ratio,
              n_job, n_machine, n_edge, episode,show_on=False):
    list_fig = []  # 要在画布上子图上循环画图的数据lst

    if da_loss:  # 有些episode太少了，还没有update，防止是空集合
        # 还需要从gpu转到cpu，然后detach去掉梯度，才能转成numpy
        da_loss, da_loss_operation = torch.stack(da_loss).cpu().detach().numpy(), \
            torch.stack(da_loss_operation).cpu().detach().numpy()  # 先堆叠tensor，list转成大tensor，再转array，画图用
        if dv_loss:
            dv_loss = torch.stack(dv_loss).cpu().detach().numpy()
            # d_loss = np.array(dv_loss) + np.array(da_loss) + np.array(dEntropy_loss)  # 把权重比例加上去
            d_loss = da_loss + da_loss_operation + dv_loss  # 总的loss(numpy数组相加元素)，两个loss列表每次都扩充，直接相加即可，把权重比例加上去
        else:
            d_loss = da_loss + da_loss_operation   # 总的loss(numpy数组相加元素)，两个loss列表每次都扩充，直接相加即可，把权重比例加上去

    list_fig.append(da_loss)  # actor的loss，因为我loss=数值就加了负号，这里画出来应该下降曲线，不是就有问题？？？（loss之前累加了，我的错！）
    list_fig.append(da_loss_operation)  # actor的loss，因为我loss=数值就加了负号，这里画出来应该下降曲线，不是就有问题？？？（loss之前累加了，我的错！）
    if len(dv_loss) != 0: # 存在任意一个数？
        list_fig.append(dv_loss)  # value的loss
    else:
        list_fig.append([])
    list_fig.append(d_loss)  # 总loss

    # if a_grad_norm:
    #     a_grad_norm, a_grad_norm_operation, v_grad_norm = torch.stack(a_grad_norm).cpu().detach().numpy(), \
    #         torch.stack(a_grad_norm_operation).cpu().detach().numpy(), \
    #         torch.stack(v_grad_norm).cpu().detach().numpy()  # list中tensor元素转成numpy。画图用

    list_fig.append(rewards_sum)  # 总r
    list_fig.append(rewards_sum_operation)  # 总r
    list_fig.append(vali_cost_lst)  # 验证数据的总的累积奖励 = -cost

    list_fig.append(totalCost_t)  # 总加工时间mk
    list_fig.append(net1_totalCost_e)  # 总加工能耗 = t=1 * e1
    list_fig.append(net1_totalCost_trans)  # m中运输t*1
    list_fig.append(totalCost_idle)  # idle t
    list_fig.append(net1_totalCost_ratio)  # 设备利用率，ratio

    esa_ppo_result_fig(list_fig, n_job, n_machine, n_edge, episode, show_on=show_on)  # 只保存图片



"""
test的3D图和4个横截面图：

python，假设现在4个指标是时间MK、能耗PT、运输TT和空闲IT，然后有3组不同的数据，代表不同的方法123，分别用星星、三角、圆圈图案表示。
我需要画一个三维图，XYZ轴分别对应MK、PT、TT，IT通过图中图案W的颜色深浅代表不同的大小，最好都是深色系。
现在需要画出一个三维图，图布大一点，然后再画出四个截面图，分别是XY、XZ、YZ和WW，并给出legend
"""
# import matplotlib.pyplot as plt
# import numpy as np
def plot_test_3d_cross_fig(data, fig_pth, label_name):
    """

    :param data:
    :return:
    """
    # 示例数据的格式
    """data = {
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
    }"""

    # 创建一个大图
    fig = plt.figure(figsize=(15, 12))

    # 创建3D散点图
    ax = fig.add_subplot(111, projection='3d')

    # PDRs_label_jointActor = ['FIFO+SPT', 'FIFO+SEC', 'FIFO+RA',
    #                          'MOR+SPT', 'MOR+SEC', 'MOR+RA',
    #                          'RA+SPT', 'RA+SEC', 'RA+RA',
    #                          'MIP_MO_FJSP', 'PPO']
    
    # PDRs_label_jointActor = label_name

    # 为不同方法的数据绘制散点图
    colors = ['k', 'r', 'b']  # 用于方法1、2、3的颜色
    # markers = ['*', 'o', '^']  # 用于星星、圆圈、三角
    # markers = ['*', 'o', '^', 's', 'D', 'x', '+', '|', '_', '1', '2', '3', '4', '8', 'p', 'H', 'v', '<', '>', 'h']
    # markers = ['.', 'o', 'v', '^', 's', '<', '>', 'p', 'h', '+', 'X', '|', 'D', '8', 'd', 'P', '_', '1', 'H', '*']
    markers = ['o', 'v', 's', 'p', 'h', 'd', 'D', '<', 'X', 'P', '*', '.', '8', 'H', '>', '^', ',', '+', '|', '_', '1']

    # # 如果需要随机排序这些标记，可以使用 random.shuffle() 函数
    # random.shuffle(markers)
    # # 现在，markers 列表包含20个不同的标记样式
    # print(markers)

    # for i, (method, method_data) in enumerate(data.items()):
    #     ax.scatter(method_data['MK'], method_data['PT'], method_data['TT'], c=method_data['IT'], cmap='Reds', s=100, label=f'Method {i + 1}', marker=markers[i])

    # 创建自定义的 colormap
    # custom_cmap = cm.get_cmap('RdYlBu')  # 这里使用了 'RdYlBu' colormap
    # 创建自定义的'warmcool'颜色映射
    colors = ['#FF0000', '#FF8000', '#FFFF00', '#00FFFF', '#0000FF']
    cmap = mcolors.LinearSegmentedColormap.from_list('warmcool', colors)  # todo 1109-自定义的暖色到冷色，越小的越红！看看怎么用？ + 改成只画平均数！

    """ 颜色深浅代表第四个维度，IT """
    for i, (method, method_data) in enumerate(data.items()):
        ax.scatter(
            method_data['MK'],
            method_data['PT'],
            method_data['IT'],
            c=method_data['TT'],
            # cmap='Reds',
            cmap='coolwarm',#  冷色到暖色的双色调 colormap。
            # cmap= 'magma',    # 一种黑色到橙色的 colormap，其中橙色更加饱和
            # cmap='cividis', #一种为色盲设计的 colormap，提供良好的对比度。
            # cmap=custom_cmap,
            s=100,  # 参数s表示散点图中点的大小
            # label=f'Method {i + 1}',
            label=label_name[i],
            marker=markers[i],
            edgecolors='k'  # 使用黑色边缘 edgecolors 参数来指定标记的边缘颜色
        )

    # 设置轴标签
    ax.set_xlabel('MK')
    ax.set_ylabel('PT')
    ax.set_zlabel('IT')

    # 添加颜色深浅的colorbar
    cbar = plt.colorbar(ax.scatter([], [], [], c=[], cmap='coolwarm'))
    cbar.set_label('TT')

    # 添加图例
    # ax.legend()
    # 调整标签的位置
    # bbox_to_anchor=(1.05, 1.0) 意味着将图例放置在轴的右侧，水平方向偏移为1.05，垂直方向偏移为1.0。这将导致图例略微超出轴的右边，并放在轴的上方。
    ax.legend(loc='upper left', bbox_to_anchor=(-0.05, 1.0), fontsize='small') # 相对坐标的形式指定图例的位置，其中 (0,0) 表示轴的左下角，(1,1) 表示轴的右上角

    # 保存3d图
    plt.savefig(fig_pth + '3D_plots.png')
    


    # 创建XY、XZ、YZ和WW的截面图
    fig, axes = plt.subplots(2, 2, figsize=(24, 15))  # 调整画布大小
    for i, plane in enumerate(['XY', 'XZ', 'YZ', 'WW']):
        ax = axes[i // 2, i % 2]
        ax.set_title(f'{plane} Plane')

        for j, (method, method_data) in enumerate(data.items()):
            # method_data = np.array(method_data)
            if plane == 'XY':
                ax.scatter(method_data['MK'], method_data['PT'], c=method_data['TT'], cmap='coolwarm', s=100, label=label_name[j], marker=markers[j])
                ax.set_xlabel('MK')
                ax.set_ylabel('PT')
            elif plane == 'XZ':
                ax.scatter(method_data['MK'], method_data['IT'], c=method_data['TT'], cmap='coolwarm', s=100, label=label_name[j], marker=markers[j])
                ax.set_xlabel('MK')
                ax.set_ylabel('IT')
            elif plane == 'YZ':
                ax.scatter(method_data['PT'], method_data['IT'], c=method_data['TT'], cmap='coolwarm', s=100, label=label_name[j], marker=markers[j])
                ax.set_xlabel('PT')
                ax.set_ylabel('IT')
            elif plane == 'WW':
                """对应横纵坐标都是IT数值的画法，看不太清楚"""
                lenx = len(method_data['TT'])
                ax.scatter(method_data['TT'], method_data['TT'], c=method_data['TT'], cmap='coolwarm', s=100, label=label_name[j], marker=markers[j])
                ax.set_xlabel('TT')
                ax.set_ylabel('TT')

        cbar = plt.colorbar(ax.scatter([], [], c=[], cmap='coolwarm'), ax=ax)
        cbar.set_label('TT')
        # ax.legend()
        # ax.legend(loc='upper left', bbox_to_anchor=(-0.05, 1.0),fontsize='small')  # 相对坐标的形式指定图例的位置，其中 (0,0) 表示轴的左下角，(1,1) 表示轴的右上角
        # 调整图例大小
        for ax in axes.flatten():
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1.0),fontsize='small')  # 调整图例字体大小

    # 保存截面图
    plt.savefig(fig_pth + 'XY_XZ_YZ_WW_plots.png')
    plt.tight_layout()  # 自动调整子图之间的间距
    plt.show()



# if __name__ == '__main__':

# list_fig.append(ma_loss)
# list_fig.append(mv_loss)
# list_fig.append(mEntropy_loss)
# list_fig.append(da_loss)
# list_fig.append(dv_loss)
# list_fig.append(dEntropy_loss)
#
# # 系统cost都是没有缩放的版本
# list_fig.append(totalCost_trans)  # m中运输t*1
# list_fig.append(totalCost_e) # 总能耗 = t * e1
# list_fig.append(totalCost_t) # makespan
# list_fig.append(totalCost_idle) # idle t



# gs = gridspec.GridSpec(3, 2) # 3 rows, 2 columns
# sub1 = plt.subplot(gs[0, 0]) # 第1行第1列
# sub2 = plt.subplot(gs[0, 1]) # 第1行第2列
# sub3 = plt.subplot(gs[1, 0]) # 第2行第1列
# sub4 = plt.subplot(gs[1, 1]) # 第2行第2列
# sub5 = plt.subplot(gs[2, 0]) # 第3行第1列
# sub6 = plt.subplot(gs[2, 1]) # 第3行第2列
#
# # 对6个子图进行绘制
# sub1.plot()
# sub2.plot()
# sub3.plot()
# sub4.plot()
# sub5.plot()
# sub6.plot()
#
# #调整每个子图相对位置和大小，以及面板的宽度、高度等
# gs.update(hspace=0.5, left=0.05, right=0.95, top=0.95, bottom=0.05)
#
# plt.show()


# plt.subplot(3, 2, 1)
# plt.title('m training loss')
# plt.plot(m_losses)
# plt.grid()
#
# plt.subplot(3, 2, 2)
# plt.title('d training loss')
# plt.plot(losses)
# plt.grid()
#
# plt.subplot(3, 2, 3)
# plt.title('each episode normalized-reward m')
# # 归一化下reward，不然数太大不好看
# # rewards_sum = rewards_sum - np.mean(rewards_sum)  # -均值
# # rewards_sum = rewards_sum / (np.std(rewards_sum) + 1e-8)  #
# # rewards_sum = (rewards_sum - np.min(rewards_sum)) / (np.max(rewards_sum) - np.min(rewards_sum))
# m_rewards_sum = np.array(m_rewards_sum) / 1.0  # 注意列表没有除法运算！
# plt.plot(m_rewards_sum)
# plt.grid()
#
# plt.subplot(3, 2, 4)
# plt.title('each episode normalized-reward d')
# rewards_sum = np.array(rewards_sum) / 1.0  # 注意列表没有除法运算！
# plt.plot(rewards_sum)
# plt.grid()
#
#
# plt.subplot(3, 2, 5)
# plt.title('m grad_norm parameters')
# plt.plot(m_avg_grad_norm, label='avg')
# plt.plot(m_max_grad_norm, label='max')
# plt.plot(m_std_grad_norm, label='std')
# plt.legend()
# plt.grid()
#
# plt.subplot(3, 2, 6)
# plt.title('d grad_norm parameters')
# plt.plot(d_avg_grad_norm, label='avg')
# plt.plot(d_max_grad_norm, label='max')
# plt.plot(d_std_grad_norm, label='std')
# plt.legend()
# plt.grid()
# plt.show()
#
# # 增加子图之间的垂直间距（修改子图边界）
# plt.subplots_adjust(hspace=0.5, top=0.5, bottom=0.2)
# plt.show()


""" =============================以下是关于GPU和memory的代码============================================   """
# """"
# 获取当前进程pid内存的使用量
# """
# def get_process_gpu_memory(pid):
#     process = psutil.Process(pid)
#     if hasattr(process, 'memory_info'):
#         mem_info = process.memory_info()
#         return mem_info.rss / (1024 * 1024)  # 转换为MB
#
#     return 0
#
# """"
# 获取当前进程pid的内存使用率
# """
# def get_process_gpu_usage(pid):
#     process = psutil.Process(pid)
#     if hasattr(process, 'memory_percent'):
#         return process.memory_percent()
#
#     return 0

"""
获取当前 Python 文件正在使用的 GPU 的索引
"""
def get_current_gpu_index():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        return current_device
    else:
        return None


def get_GPU_usage(current_device=True):
    # # 获取当前 Python 文件的进程 ID (PID)
    # pid = os.getpid()
    # # 获取当前使用的GPU
    # gpu_index = get_current_gpu_index()
    # # 获取进程显存使用量（以MB为单位）
    # memory_usage = get_process_gpu_memory(pid)
    # # 获取进程显存使用率（百分比）
    # memory_usage_percent = get_process_gpu_usage(pid)

    pynvml.nvmlInit()  # 初始化`pynvml`库：
    device_count = pynvml.nvmlDeviceGetCount() # 获取GPU设备数量
    # 获取当前使用的GPU
    gpu_index = get_current_gpu_index()

    pid = psutil.Process().pid  # 获取当前进程的ID(有问题，和云平台上边的进程号不一样)
    # print("pid = ", pid)
    if current_device: # 只返回当前使用的gpu的显存使用量
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory = info.used / 1024 ** 2  # 显存使用量（以MB为单位）
        total_memory = info.total / 1024 ** 2  # 显存总量（以MB为单位）
        usage_percentage = used_memory / total_memory * 100  # 显存使用率（百分比）

        # 获取当前进程的内存使用量（非显存）
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        used_cpu_memory = memory_info.rss / 1024 ** 2  # 内存使用量（以MB为单位）
        # print(f"当前进程内存使用量: {used_memory:.2f}MB")

        # print("Current PID:{}, GPU:{}, MEMORY:{}, USAGE:{}".format(pid,
        #                                                            gpu_index,
        #                                                            used_memory,
        #                                                            usage_percentage,
        #                                                            used_cpu_memory))

        pynvml.nvmlShutdown()  # 最后，记得在程序结束时清理`pynvml`资源 (默认形参是true，所以一定会执行)

        return gpu_index, used_memory, usage_percentage, used_cpu_memory
    # else:
    #     for i in range(device_count): # 遍历每个GPU设备，获取当前进程的显存使用量
    #         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    #         info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    #         used_memory = info.used / 1024**2  # 显存使用量（以MB为单位）
    #         total_memory = info.total / 1024 ** 2  # 显存总量（以MB为单位）
    #         usage_percentage = used_memory / total_memory * 100  # 显存使用率（百分比）
    #         print(f"GPU {i}: 当前进程显存使用量 {used_memory:.2f}MB")
    #         print(f"GPU {i}: 使用率 {usage_percentage:.2f}%")









# if __name__ == '__main__':
#     import os
#
#     # 获取当前 Python 文件的进程 ID (PID)
#     pid = os.getpid()
#     print("Current PID:", pid)
#
#     # 获取进程显存使用量（以MB为单位）
#     memory_usage = get_process_gpu_memory(pid)
#     print("Memory Usage:", memory_usage, "MB")
#
#     # 获取进程显存使用率（百分比）
#     memory_usage_percent = get_process_gpu_usage(pid)
#     print("Memory Usage Percent:", memory_usage_percent, "%")


# import torch
#
# # 指定在第一个 GPU 上执行
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)  # 设置当前使用的 GPU
#
# # 在设备上创建张量
# x = torch.tensor([1, 2, 3], device=device)
#
# # 在指定的设备上执行操作
# y = torch.tensor([4, 5, 6], device=device)
# z = x + y
#
# print(z)
#
# import torch
#
#
# def get_current_gpu_index():
#     if torch.cuda.is_available():
#         current_device = torch.cuda.current_device()
#         return current_device
#     else:
#         return None
#
#     # 获取当前 Python 文件正在使用的 GPU 的索引
#
#
# gpu_index = get_current_gpu_index()
# if gpu_index is not None:
#     print("Current GPU Index:", gpu_index)
# else:
#     print("No GPU available.")

"""
获取可用GPU的显存使用 + 使用率
"""
# import psutil
# import pynvml
#
# pynvml.nvmlInit() # 初始化`pynvml`库：
# device_count = pynvml.nvmlDeviceGetCount() # 获取GPU设备数量
# pid = psutil.Process().pid  # 获取当前进程的ID
# print("pid = ", pid)
# for i in range(device_count): # 遍历每个GPU设备，获取当前进程的显存使用量
#     handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#     info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     used_memory = info.used / 1024**2  # 显存使用量（以MB为单位）
#     total_memory = info.total / 1024 ** 2  # 显存总量（以MB为单位）
#     usage_percentage = used_memory / total_memory * 100  # 显存使用率（百分比）
#     print(f"GPU {i}: 当前进程显存使用量 {used_memory:.2f}MB")
#     print(f"GPU {i}: 使用率 {usage_percentage:.2f}%")
#
# # 获取当前进程的内存使用量（非显存）
# process = psutil.Process(pid)
# memory_info = process.memory_info()
# used_memory = memory_info.rss / 1024**2  # 内存使用量（以MB为单位）
# print(f"当前进程内存使用量: {used_memory:.2f}MB")
#
# pynvml.nvmlShutdown() # 最后，记得在程序结束时清理`pynvml`资源
#
# import os
#
# # 获取当前 Python 文件的进程 ID (PID)
# pid = os.getpid()
# print("Current PID:", pid)

