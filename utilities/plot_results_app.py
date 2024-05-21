import numpy as np
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import os.path
sys.path.append("./utility")

DATA_NAMES = ["hainmueller_immigrant", "hainmueller_candidate"]
TOTAL_SIZES = [100*i+100 for i in range(8)]

def compare_ACC(args):
    MAXSEED = int(args["seed"])
    MODELS = ["UCB", "UNIFORM", "DE", "GRADDE", "BALD"] # "GRADBALD" 

    results = np.zeros((len(DATA_NAMES),len(MODELS),len(TOTAL_SIZES),MAXSEED))
    for i in range(len(DATA_NAMES)):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        for SEED in range(1,MAXSEED+1):
            for k in range(len(MODELS)):
                result_filename = "./results_application/ACC_"+ DATA_NAMES[i] \
                            + "_" + MODELS[k] + "_SEED" + str(SEED) + ".csv"
                data = pd.read_csv(result_filename, header=None).to_numpy()
                results[i, k, :, SEED-1] = data.reshape((-1,))
    
    fig, ax = plt.subplots(nrows=1, ncols=len(DATA_NAMES), figsize=(12, 3), dpi=100)
    colors = ["forestgreen",  "limegreen", "gold", "darkseagreen",  "blue"]
    for i in range(len(DATA_NAMES)):
        ax[i].set_title(DATA_NAMES[i], fontsize=14)
        if i==0:
            ax[i].set_ylabel("Pred ACC", fontsize=14)
        bplots = []
        for k in range(len(MODELS)):
            tmp = results[i,k,:,:]
            tmp = [(tmp[j,:]-np.mean(tmp[j,:]))/np.sqrt(MAXSEED)+np.mean(tmp[j,:])\
                    for j in range(len(TOTAL_SIZES))]  
            bplot = ax[i].boxplot(tmp, positions=4+4*np.arange(len(TOTAL_SIZES))+(k-2)*0.75, showfliers=False,\
                                patch_artist=True, widths=0.45)
            # plot connected line
            if k==len(MODELS)-1:
                ax[i].plot(4+4*np.arange(len(TOTAL_SIZES))+(k-2)*0.75, np.median(tmp,axis=1), \
                            color=colors[k], linestyle="solid")
            else:
                ax[i].plot(4+4*np.arange(len(TOTAL_SIZES))+(k-2)*0.75, np.median(tmp,axis=1), \
                            color=colors[k], linestyle="dotted")
            for patch in bplot['boxes']:
                patch.set_facecolor(colors[k])
            bplots.append(bplot)
        ax[i].set_xticks(4+4*np.arange(len(TOTAL_SIZES)))
        ax[i].set_xticklabels(TOTAL_SIZES)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].tick_params(left=False, bottom=False)
        if i==0:
            ax[i].legend([bplots[j]["boxes"][0] for j in range(len(MODELS))],MODELS)
    params = {
        'axes.labelsize': 8,
        'font.size': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [10, 6]
    }
    plt.rcParams.update(params)
    fig.subplots_adjust(left=0.06, bottom=0.10, right=0.99, top=0.90, wspace=0.12)
    plt.savefig("./results_application/application_ACC.pdf", format="pdf", dpi=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='-s seed')
    parser.add_argument('-s','--seed', help='random seed', required=True)
    args = vars(parser.parse_args())
    compare_ACC(args)