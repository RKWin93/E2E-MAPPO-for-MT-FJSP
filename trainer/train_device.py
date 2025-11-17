
from instance.generate_allsize_mofjsp_dataset import Logger
import torch
import os

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)  # 设置当前使用的 GPU
Logger.log("Training/device", f"device={device}, torch.cuda.set_device{device}", print_true=1)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,4,5,6,7"  # 选择可用的GPU, 指定哪些GPU对程序可见



# if __name__ == '__main__':

#     import torch
#     print("Available GPUs:", torch.cuda.device_count())   # 返回2，我只有01gpu能用，因为我的实例就是01gpu上边的 TODO 250506-修改为014567这6个GPU
