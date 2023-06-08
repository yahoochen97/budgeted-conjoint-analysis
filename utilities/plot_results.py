import numpy as np
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
sys.path.append("./utility")

DATA_NAMES = ["2Dplane", "Friedman"]
NS = [25, 50, 100, 200, 400, 800, 1600]
MEASURES = ["RMSE","CORRELATION","COVERAGE","LL"]

def main(args):
    MODELS = ["diffinmean", "gppoint", "gpGMM"]
    MAXSEED = int(args["seed"])
    TA = int(args["TA"])
    
    results = np.zeros((len(MEASURES), len(DATA_NAMES),len(NS),len(MODELS),MAXSEED))
    for i in range(len(DATA_NAMES)):
        for SEED in range(1,MAXSEED+1):
            for j in range(len(NS)):
                result_filename = "./results/"+ DATA_NAMES[i] + "_N" + str(NS[j]) \
                    + "_TA" + str(TA) + "_SEED" + str(SEED) + ".csv"
                data = pd.read_csv(result_filename)
                for k in range(len(MODELS)):
                    tmp = data[data.model==MODELS[k]]
                    est_mu = tmp["mean"].to_numpy()
                    est_std = tmp["std"].to_numpy()
                    true_effect = tmp["effect"].to_numpy()
                    est_mu = est_mu[~np.isnan(true_effect)]
                    est_std = est_std[~np.isnan(true_effect)]
                    true_effect = true_effect[~np.isnan(true_effect)]
                    # ratio = np.std(true_effect) / np.std(est_mu)
                    # bias = np.mean(true_effect) - np.mean(est_mu)
                    # est_mu = est_mu * ratio + bias
                    # est_std = est_std * ratio + bias
                    RMSE = np.sqrt(np.mean((est_mu-true_effect)**2))
                    CORRELATION = np.corrcoef(est_mu, true_effect)[0,1]
                    COVERAGE = np.mean(np.logical_and((est_mu-1.96*est_std)<=true_effect,\
                                                       true_effect<=(est_mu+1.96*est_std)))
                    LL = -np.log(2*np.pi)/2+np.mean(-np.log(est_std+1e-8)-(est_mu-true_effect)**2)
                    results[0,i,j,k,SEED-1] = RMSE
                    results[1,i,j,k,SEED-1] = CORRELATION
                    results[2,i,j,k,SEED-1] = COVERAGE
                    results[3,i,j,k,SEED-1] = LL
    
    MODELS = ["diff-in-mean", "gp-GMM-1", "gp_GMM-10"]
    fig, ax = plt.subplots(nrows=len(DATA_NAMES), ncols=len(MEASURES), figsize=(15, 8), dpi=100)
    colors = ["blue", "red", "limegreen"]
    for i in range(len(DATA_NAMES)):
        for m in range(len(MEASURES)):
            if i==0:
                ax[i,m].set_title(MEASURES[m], fontsize=14)
            if m==0:
                ax[i,m].set_ylabel(DATA_NAMES[i])
            bplots = []
            for k in range(len(MODELS)):
                tmp = results[m,i,:,k,:]
                # ax[i,m].plot(1+np.arange(len(NS)), np.mean(tmp,axis=1), color=colors[k])
                # ax[i,m].errorbar(1+np.arange(len(NS)), np.mean(tmp,axis=1),\
                #                 yerr=np.std(tmp,axis=1), marker='o', # mfc=colors[k],mec=colors[k], 
                #                 ms=1, mew=2, capsize=4, elinewidth=1) 
                tmp = [tmp[j,:] for j in range(len(NS))]
                bplot = ax[i,m].boxplot(tmp, positions=4+4*np.arange(len(NS))+(k-1)*0.75, showfliers=False,\
                                    patch_artist=True, widths=0.45)
                for patch in bplot['boxes']:
                    patch.set_facecolor(colors[k])
                bplots.append(bplot)
            ax[i,m].set_xticks(4+4*np.arange(len(NS)))
            ax[i,m].set_xticklabels(NS)
            ax[i,m].spines[['right', 'top']].set_visible(False)
            ax[i,m].tick_params(left=False, bottom=False)
            ax[i,m].grid(axis='y', color='gray', linestyle='dashed', linewidth=1)
            if i==0 and m==0:
                ax[i,m].legend([bplots[j]["boxes"][0] for j in range(len(MODELS))],MODELS)
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
    fig.subplots_adjust(left=0.043, bottom=0.035, right=0.99, top=0.96, wspace=0.12)
    plt.savefig("./results/simulation_plot" + "_TA" + str(TA) + ".pdf", format="pdf", dpi=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='-s seed')
    parser.add_argument('-s','--seed', help='random seed', required=True)
    parser.add_argument('-t','--TA', help='test anchor', required=True)
    args = vars(parser.parse_args())
    main(args)