import pandas as pd 
import wandb

"""
运行完程序之后，建议下载wandb的数据，可以自己后来画图用，configs和result，甚至你print的log都会再wandb里边显示！！！！！可以回头查看！
"""

# 初始化 W&B 运行
# run = wandb.init()

api = wandb.Api()  # 初始化 W&B 运行

# Project is specified by <entity/project-name>
# 获取运行数据: 用户名 + 项目名称 +runs + 此次运行的8为代号
# run_path = "rk_wang-zhejiang-university/E2E-MAPPO_for_MT-FJSP/runs/w2bjidh0"   # 运行网络断开直接挂掉，16h10m
# run_path = "rk_wang-zhejiang-university/E2E-MAPPO_for_MT-FJSP/runs/17tl5uqw"  # Tmux方式，完整运行12800数据集的训练， 15h16m
# run_path = "rk_wang-zhejiang-university/E2E-MAPPO_for_MT-FJSP/runs/w737wluf"  # Tmux方式，运行无LR衰减版本，直接crash，怀疑是服务器断掉了。3h22m
# run_path = "rk_wang-zhejiang-university/E2E-MAPPO_for_MT-FJSP/runs/lrexzm2e"  # Tmux方式，完整运行无LR衰减版本，收敛。18h21m TODO 效果好
# run_path = "rk_wang-zhejiang-university/E2E-MAPPO_for_MT-FJSP/runs/il2m18pl"  # Tmux方式，完整运行只关注job选完的mash（旧版）+无LR衰减版本，收敛。23h20m

run_path = "rk_wang-zhejiang-university/Test_for_MT-FJSP/runs/i5ajkwoy"  # 无LR衰减版本top1和paper模型对比进行test，效果更好。 15m24s测试seed3




runs = api.run(run_path)

summary_list, config_list, name_list = [], [], []
 
# .summary contains the output keys/values for metrics like accuracy.
#  We call ._json_dict to omit large files 
summary_list.append(runs.summary._json_dict)   # TODO 记录的是最后一组数据！
 
# .config contains the hyperparameters.
#  We remove special values that start with _.
config_list.append( {k: v for k,v in runs.config.items() if not k.startswith('_')})   # 记录的是超参数

# .name is the human-readable name of the run.
name_list.append(runs.name)   # 当前运行的名称

runs_df = pd.DataFrame({
    "summary": summary_list,   # 
    "config": config_list,
    "name": name_list
    })

pth = "/remote-home/iot_wangrongkai/FJSP-LLM-250327/20241229-DTr-FJSP/MOFJSP-DRL/wandb_data/"
project_f = pth + f"config_{name_list}.csv"

runs_df.to_csv(project_f)
print("数据已成功导出到 config.csv")


# 获取历史数据，wandb画图所用的数据
history = runs.history(keys=None, samples=10000) # 注意，这里的samples=10000表示最少返回10000行数据，有时候没有设置这个的话，可能只会返回几百行数据。

# 将历史数据转换为 DataFrame
df = pd.DataFrame(history)

history_data_f = pth + f"wandb_history_data_{name_list}.csv"

# 保存为 CSV 文件
df.to_csv(history_data_f, index=False)

print("数据已成功导出到 wandb_history_data.csv")