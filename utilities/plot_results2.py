import numpy as np
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import os.path
sys.path.append("./utility")

DATA_NAMES = ["twoDplane", "Friedman"]
N = 1000
TOTAL_SIZES = [25*i+50 for i in range(7)]
MEASURES = ["RMSE","COVERAGE","LL"]

def main(args):
    MAXSEED = int(args["seed"])
    MODELS = ["UNIFORM", "US", "GRADUS", "BALD", "GRADBALD"]
    
    results = np.zeros((len(MEASURES), len(DATA_NAMES),len(TOTAL_SIZES),len(MODELS),MAXSEED))
    for i in range(len(DATA_NAMES)):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        for SEED in range(1,MAXSEED+1):
            for j in range(len(TOTAL_SIZES)):
                for k in range(len(MODELS)):
                    result_filename = "./results2/"+ DATA_NAMES[i] + "_N" + str(N) \
                        + "_S" + str(TOTAL_SIZES[j]) + "_" + MODELS[k] + "_SEED" + str(SEED) + ".csv"
                    data = pd.read_csv(result_filename)
                    tmp = data[data.policy==MODELS[k]]
                    est_mu = tmp["mean"].to_numpy()
                    est_std = tmp["std"].to_numpy()
                    true_effect = tmp["effect"].to_numpy()
                    flag = ~np.isnan(true_effect) & ~np.isnan(est_mu) & ~np.isnan(est_std) & np.array(est_std!=0)
                    est_mu = est_mu[flag]
                    est_std = est_std[flag]
                    true_effect = true_effect[flag]
                    # ratio = np.std(true_effect)/np.std(est_mu)
                    # est_mu = est_mu*ratio
                    # est_std = est_std*ratio

                    if est_mu.shape[0]==0:
                        results[0,i,j,k,SEED-1] = RMSE
                        # results[1,i,j,k,SEED-1] = CORRELATION
                        results[1,i,j,k,SEED-1] = COVERAGE
                        results[2,i,j,k,SEED-1] = LL
                        continue
                    
                    RMSE = np.sqrt(np.mean((est_mu-true_effect)**2))
                    CORRELATION = np.corrcoef(est_mu, true_effect)[0,1]
                    COVERAGE = np.mean(np.logical_and((est_mu-1.96*est_std)<=true_effect,\
                                                        true_effect<=(est_mu+1.96*est_std)))
                    LL = -np.log(2*np.pi) -np.mean(np.log(est_std**2)/2)-np.mean((est_mu-true_effect)**2/2/est_std**2)
                    results[0,i,j,k,SEED-1] = RMSE
                    # results[1,i,j,k,SEED-1] = CORRELATION
                    results[1,i,j,k,SEED-1] = COVERAGE
                    results[2,i,j,k,SEED-1] = LL
    
    fig, ax = plt.subplots(nrows=len(DATA_NAMES), ncols=len(MEASURES), figsize=(15, 8), dpi=100)
    colors = ["forestgreen", "limegreen", "darkseagreen",  "blue","steelblue"]
    for i in range(len(DATA_NAMES)):
        for m in range(len(MEASURES)):
            if i==0:
                ax[i,m].set_title(MEASURES[m], fontsize=14)
            if m==0:
                ax[i,m].set_ylabel(DATA_NAMES[i])
            bplots = []
            for k in range(len(MODELS)):
                tmp = results[m,i,:,k,:]
                tmp = [tmp[j,:] for j in range(len(TOTAL_SIZES))]
                bplot = ax[i,m].boxplot(tmp, positions=4+4*np.arange(len(TOTAL_SIZES))+(k-1)*0.75, showfliers=False,\
                                    patch_artist=True, widths=0.45)
                for patch in bplot['boxes']:
                    patch.set_facecolor(colors[k])
                bplots.append(bplot)
            ax[i,m].set_xticks(4+4*np.arange(len(TOTAL_SIZES)))
            ax[i,m].set_xticklabels(TOTAL_SIZES)
            ax[i,m].spines['top'].set_visible(False)
            ax[i,m].spines['right'].set_visible(False)
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
    fig.subplots_adjust(left=0.05, bottom=0.035, right=0.99, top=0.96, wspace=0.12)
    plt.savefig("./results2/simulation2_plot.pdf", format="pdf", dpi=100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='-s seed')
    parser.add_argument('-s','--seed', help='random seed', required=True)
    args = vars(parser.parse_args())
    main(args)