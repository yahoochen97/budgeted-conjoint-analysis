import numpy as np
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
sys.path.append("./utility")

DATA_NAMES = ["twoDplane", "Friedman"]
N = 300
MEASURES = ["RMSE","CORRELATION", "COVERAGE","LL"]

def main(args):
    MODELS = ["diffinmean", "lm",  "gppoint", "gpGMM"]
    MAXSEED = int(args["seed"])
    TA = int(args["TA"])
    effect_type = args["effect"]
    
    results = np.zeros((len(MEASURES), len(DATA_NAMES), len(MODELS),MAXSEED))
    for i in range(len(DATA_NAMES)):   
        for SEED in range(1,MAXSEED+1):
            if effect_type=="pop":
                result_filename = "./results/"+ DATA_NAMES[i] + "_N" + str(N) \
                    + "_TA" + str(TA) + "_SEED" + str(SEED) + ".csv"
            else:
                result_filename = "./results/ind_"+ DATA_NAMES[i] + "_N" + str(N) \
                    + "_TA" + str(TA) + "_SEED" + str(SEED) + ".csv"
            data = pd.read_csv(result_filename)
            for k in range(len(MODELS)):
                # print("summarizing " + MODELS[k] + " SEED " + str(SEED) + " N " + str(NS[j]) + " ...")
                tmp = data[data.model==MODELS[k]]
                est_mu = tmp["mean"].to_numpy().reshape((-1,))
                est_std = tmp["std"].to_numpy().reshape((-1,))
                true_effect = tmp["effect"].to_numpy().reshape((-1,))
                flag = ~np.isnan(true_effect) & ~np.isnan(est_mu) & ~np.isnan(est_std) & np.array(est_std!=0)
                est_mu = est_mu[flag]
                est_std = est_std[flag] 
                est_std += 1e-3
                true_effect = true_effect[flag]
                if est_mu.shape[0]==0:
                    results[0,i,k,SEED-1] = RMSE
                    results[1,i,k,SEED-1] = CORRELATION
                    results[2,i,k,SEED-1] = COVERAGE
                    results[3,i,k,SEED-1] = LL
                    continue
                
                RMSE = np.sqrt(np.mean((est_mu-true_effect)**2))
                CORRELATION = np.abs(np.corrcoef(est_mu, true_effect)[0,1])
                COVERAGE = np.mean(np.logical_and((est_mu-1.96*est_std)<=true_effect,\
                                                    true_effect<=(est_mu+1.96*est_std)))
                LL = -np.log(2*np.pi)/2 -np.mean(np.log(est_std))-np.mean((est_mu-true_effect)**2/2/est_std**2)
                results[0,i,k,SEED-1] = RMSE
                results[1,i,k,SEED-1] = CORRELATION
                results[2,i,k,SEED-1] = COVERAGE
                results[3,i,k,SEED-1] = LL
    
    MODELS = ["diff-in-mean", "lm-GMM", "gp-point", "gp-GMM"]
    for i in range(len(DATA_NAMES)):
        print("summarize "+DATA_NAMES[i]+"...\n")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        mu = np.mean(results[:,i,:,:], axis=2).T
        noise = np.std(results[:,i,:,:], axis=2).T / np.sqrt(MAXSEED)
        df = pd.DataFrame(data=mu, index=pd.Index(MODELS), columns=MEASURES)
        print(df)
        if effect_type=="pop":
            df.to_csv("./results/summary_mu" + DATA_NAMES[i] + "_TA" + str(TA) + "_N" + str(N) + ".csv", header=True)
        else:
            df.to_csv("./results/summary_mu" + DATA_NAMES[i] + "_TA" + str(TA) + "_N" + str(N) + "_ind.csv", header=True)
        
        df = pd.DataFrame(data=noise, index=pd.Index(MODELS), columns=MEASURES)
        # print(df)
        if effect_type=="pop":
            df.to_csv("./results/summary_noise" + DATA_NAMES[i] + "_TA" + str(TA) + "_N" + str(N) + ".csv", header=True)
        else:
            df.to_csv("./results/summary_noise" + DATA_NAMES[i] + "_TA" + str(TA) + "_N" + str(N) + "_ind.csv", header=True)

def compare_ACC(args):
    MODELS = ["diffinmean", "lm",  "gpGMM"]
    MAXSEED = int(args["seed"])
    TA = int(args["TA"])
    effect_type = args["effect"]
    
    results = np.zeros((len(DATA_NAMES), len(MODELS),MAXSEED))
    for i in range(len(DATA_NAMES)):   
        for SEED in range(1,MAXSEED+1):
            result_filename = "./results/ACC_"+ DATA_NAMES[i] + \
                  "_N" + str(N) + "_TA" + str(TA) + "_SEED" + str(SEED) + ".csv"
            data = pd.read_csv(result_filename).acc.to_numpy()
            results[i, :, SEED-1] = data.reshape((-1,))
            print("ACC "+DATA_NAMES[i]+"...\n")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
            mu = np.mean(results[i,:,:], axis=1).T
            noise = np.std(results[i,:,:], axis=1).T / np.sqrt(MAXSEED)
            df = pd.DataFrame(data=mu, index=pd.Index(MODELS), columns=1)
            print(df)
            df = pd.DataFrame(data=noise, index=pd.Index(MODELS), columns=MEASURES)
            print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='-s seed -t TA -e effect')
    parser.add_argument('-s','--seed', help='random seed', required=True)
    parser.add_argument('-t','--TA', help='test anchor', required=True)
    parser.add_argument('-e','--effect', help='pop/ind effects', required=True)
    args = vars(parser.parse_args())
    main(args)
    compare_ACC(args)