import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from user_config import DEFAULT_IMG_DIR, DEFAULT_DATA_DIR
import argparse
import glob

def smooth(data, sm=1, value="Averagetest_reward"):
    if sm > 1:
        smooth_data = []
        for d in data:
            x = np.asarray(d[value])
            y = np.ones(sm)*1.0/sm
            d[value] = np.convolve(y, x, "same")

            smooth_data.append(d)

        return smooth_data
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_name', default=None)
    parser.add_argument('--seed', default='10', type=int)
    parser.add_argument('--smooth', default='2', type=int)
    parser.add_argument('--output_name', default=None, type=str)
    parser.add_argument('--x', default="Epoch", type=str)
    parser.add_argument('--y', default="Averagetest_reward", type=str)
    args = parser.parse_args()

    tasks = ["hopper", "halfcheetah", "walker2d"]
    for task in tasks:
        plt.cla()
        data = []
        for kl in [0.03, 0.05, 0.07]:
            for seed in [20, 30, 40]:
                taskname = "ppo_kl_" + task + "_clipv_maxgrad_anneallr3e-4_normal_maxkl" + str(kl) \
                    + "_gae_norm-state-return_steps2048_batch64_notdone_lastv_4_entropy_update10"
                file_dir = os.path.join(DEFAULT_DATA_DIR, taskname)
                file_seed = os.path.join(file_dir, taskname+"_s" + str(seed), "progress.txt")
                pd_data = pd.read_table(file_seed)
                pd_data["KL"] = "max_kl" + str(kl)

                data.append(pd_data)

        smooth(data, sm=args.smooth)
        data = pd.concat(data, ignore_index=True)

        sns.set(style="darkgrid", font_scale=1.5)
        sns.lineplot(data=data, x=args.x, y=args.y, hue="KL")

        output_name = "ppo_" + task + "_smooth"
        out_file = os.path.join(DEFAULT_IMG_DIR, output_name + ".png")
        plt.legend(loc='best').set_draggable(True)
        plt.tight_layout(pad=0.5)
        plt.savefig(out_file)


