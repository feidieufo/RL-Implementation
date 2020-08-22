import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from user_config import DEFAULT_IMG_DIR, DEFAULT_DATA_DIR
import argparse
import glob

def smooth(data, sm=1):
    if sm > 1:
        y = np.ones(sm)*1.0/sm
        data = np.convolve(y, data, "same")


    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_name', default=None)
    parser.add_argument('--seed', default='10', type=int)
    parser.add_argument('--output_name', default=None, type=str)
    args = parser.parse_args()

    plt.style.use('fivethirtyeight')

    if args.plot_name is None:
        file_list = glob.glob(os.path.join(DEFAULT_DATA_DIR, "ppo_mask*"))
        for file in file_list:
            for file_seed in glob.glob(os.path.join(file, "*")):
                data_file = os.path.join(file_seed, "progress.txt")
                pd_data = pd.read_table(data_file)    
                mean_name = "Averagetest_reward"
                std_name = "Stdtest_reward"
                if mean_name not in pd_data.columns or std_name not in pd_data.columns:
                    continue
                mean = pd_data[mean_name]
                std = pd_data[std_name]
                x = pd_data["Epoch"]
                mean = smooth(mean, sm=3)

                plt.plot(x, mean, c="deepskyblue", linewidth=1)
                plt.fill_between(x, mean+std, mean, color="lightskyblue")
                plt.fill_between(x, mean-std, mean, color="lightskyblue")

                output_name = file_seed.split(os.sep)[-1]
                plt.title(output_name)
                plt.xlabel("epoch")
                plt.ylabel("return")
                # plt.legend(loc = 'lower right',       # 默认在左上角即 upper left 可以通过loc进行修改
                #     fancybox = True,           # 边框
                #     framealpha = 0.5,          # 透明度
                #     shadow = True,             # 阴影
                #     borderpad = 1)             # 边框宽度

                if not os.path.exists(DEFAULT_IMG_DIR):
                    os.mkdir(DEFAULT_IMG_DIR)
                out_file = os.path.join(DEFAULT_IMG_DIR, output_name + ".png")
                plt.savefig(out_file)   
                plt.clf()

    else:             

        from utils.run_utils import setup_logger_kwargs
        logger_kwargs = setup_logger_kwargs(args.plot_name, args.seed)
        data_file = os.path.join(logger_kwargs["output_dir"], "progress.txt")

        pd_data = pd.read_table(data_file)    
        mean_name = "Averagetest_reward"
        std_name = "Stdtest_reward"
        mean = pd_data[mean_name]
        std = pd_data[std_name]
        x = pd_data["Epoch"]

        plt.plot(x, mean)
        plt.fill_between(x, mean+std, mean-std)

        output_name = args.output_name
        if args.output_name is None:
            output_name = args.plot_name + "_s" + str(args.seed)
        if not os.path.exists(DEFAULT_IMG_DIR):
            os.mkdir(DEFAULT_IMG_DIR)
        out_file = os.path.join(DEFAULT_IMG_DIR, output_name + ".png")
        plt.savefig(out_file)
