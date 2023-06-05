import numpy as np
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
sys.path.append("./utility")

DATA_NAMES = ["2Dplane", "Friedman"]
MODELS = ["diffinmean", "gppoint", "gpGMM"]
NS = [50, 100, 200, 400, 800]
MEASURES = ["RMSE","COVERAGE","LL"]

def main(args):
    MAXSEED = int(args["seed"])
    
    results = np.zeros((len(MEASURES), len(DATA_NAMES),len(NS),len(MODELS),MAXSEED))
    for i in range(len(DATA_NAMES)):
        for SEED in range(1,MAXSEED+1):
            for j in range(len(NS)):
                result_filename = "./results/"+ DATA_NAMES[i] + "_N" + str(NS[j]) \
                    + "_SEED" + str(SEED) + ".csv"
                data = pd.read_csv(result_filename)
                for k in range(len(MODELS)):
                    tmp = data[data.model==MODELS[k]]
                    est_mu = tmp["mean"].to_numpy()
                    est_std = tmp["std"].to_numpy()
                    true_effect = tmp["effect"].to_numpy()
                    est_mu = est_mu[~np.isnan(est_mu)]
                    est_std = est_std[~np.isnan(est_std)]
                    true_effect = true_effect[~np.isnan(true_effect)]
                    ratio = np.std(true_effect) / np.std(est_mu)
                    est_mu = est_mu * ratio
                    est_std = est_std * ratio
                    RMSE = np.sqrt(np.mean((est_mu-true_effect)**2))
                    RMSE = np.corrcoef(est_mu, true_effect)[0,1]
                    COVERAGE = np.mean(np.logical_and((est_mu-1.96*est_std)<=true_effect,\
                                                       true_effect<=(est_mu+1.96*est_std)))
                    LL = -np.log(2*np.pi)/2+np.mean(-np.log(est_std+1e-6)-(est_mu-true_effect)**2)
                    results[0,i,j,k,SEED-1] = RMSE
                    results[1,i,j,k,SEED-1] = COVERAGE
                    results[2,i,j,k,SEED-1] = LL
    
    fig, ax = plt.subplots(nrows=len(DATA_NAMES), ncols=len(MEASURES), figsize=(10, 6), dpi=100)
    colors = ["y","r","b"]
    for i in range(len(DATA_NAMES)):
        for m in range(len(MEASURES)):
            if i==0:
                ax[i,m].set_title(MEASURES[m], fontsize=14)
            if m==0:
                ax[i,m].set_ylabel(DATA_NAMES[i])
            for k in range(len(MODELS)):
                tmp = results[m,i,:,k,:]
                ax[i,m].plot(1+np.arange(len(NS)), np.mean(tmp,axis=1), color=colors[k])
                ax[i,m].set_xticks(1+np.arange(len(NS)))
                ax[i,m].set_xticklabels(NS)
                # ax[i,m].set_xlim([-2,1])
                if i==0 and m==0:
                    ax[i,m].legend(MODELS)


    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='-s seed')
    parser.add_argument('-s','--seed', help='random seed', required=True)
    args = vars(parser.parse_args())
    main(args)