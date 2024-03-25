
import argparse as ap

p = ap.ArgumentParser(description="Run model comparison")
p.add_argument('-a', action='store') # algorithm 
p.add_argument('-n', action='store') # noise
p.add_argument('-c', action='store') # cutoff
p.add_argument('-i', action='store') # number of iterations
p.add_argument('-d', action='store') # number of iterations
args = p.parse_args()

cond = str(args.d)
alg = str(args.a) 
noise = float(args.n )
c = int(args.c)
iterations = int(args.i)
cond_str = "mc_n"+str(noise)+"_c"+str(c)+"_i"+str(iterations)+"_"+alg

import numpy as np
from scipy.optimize import minimize
from groo.groo import get_root
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
from groo.groo import get_root
rf = get_root(".hidden_root_mc")
sys.path.append(os.path.join(rf))

from models_and_funcs import *

N = 180
#noise_level= [5, 10, 20]
#cutoff = [90, 120, 150]
models = [rw1_choice, rw2_val_choice, rw3_choice, rw6_val_choice] 
model_names = ["rw1_choice", "rw2_val_choice", "rw3_choice", "rw6_val_choice"]
bounds = {"rw1_choice": ((0,1),(0.001,50)),
          "rw2_val_choice": ((0,1),(0,1),(0.001,50)), 
          "rw3_choice": ((0,1),(0,1),(0,1),(0.001,50)), 
          "rw6_val_choice": ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0.001,50))}
df = pd.DataFrame()


for ii in range(iterations):
    print(np.round(ii/iterations,2))
    # Generate data
    rew_all, options_all = gen_bandits(N, nbandit=3)
    #indata = {"options": repeated_df, "r":rewards, "nbandit":3, "model":rw6_val_choice, "generate_choices":1} 

    # trainign data 
    rw_train = rew_all[0:c,:]
    op_train = options_all.iloc[0:c, :]

    # testing data 
    rw_test = rew_all[c:,:]
    op_test = options_all.iloc[c:, :]

    for m_idx, (m, mname) in enumerate(zip(models, model_names)): # loop over model GENERATING the data
        params_in = gen_rand_vals(bounds[mname])
        indata = {"options": options_all, "r":rew_all, "nbandit":3, "model":m, "generate_choices":1} 

        mpred = m(params_in, indata)

        #induce noice by randomly replaceing proportion of choices
        mpred["choices"] = replace_random_values(mpred["choices"], indata["options"], noise)

        ch_train = mpred["choices"][0:c]
        ch_test = mpred["choices"][c:]

        # Fit data 
        AIC =[] 
        BIC = []
        AICc = [] 
        HQC = [] #https://en.wikipedia.org/wiki/Hannan%E2%80%93Quinn_information_criterion
        P = {}
        for mfit, mname_fit in zip(models, model_names):
            # prepare data generated above for fit - all trainingn    
            indata = {"options": op_train, "r":rw_train, "nbandit":3, "model":mfit, "generate_choices":0, "choices":ch_train} 

            # fit
            opt = minimize(fun=lklhd_choice, x0=gen_rand_vals(bounds[mname_fit]), args=(indata), method=alg, bounds=bounds[mname_fit], options={'verbose': 0})

            # Get IC 
            M = lklhd_choice_m(opt.x, indata)

            AIC.append(M["AIC"])
            P[mname_fit] = opt.x
            BIC.append(M["BIC"])
            AICc.append(M["AICc"])
            HQC.append(M["HQC"])
           
        
        # which model fit best
        best_idx =[np.argmin(AIC), np.argmin(AICc), np.argmin(BIC), np.argmin(HQC)]
        # is this the correct model
        #if aic_idx == m_idx:
        #    m_recovery_AIC = 1 # etc

        # get predictive error per trial of the best model
        acc = []
        for idxx, best in enumerate(best_idx):
            #test all 
            indata = {"options": op_test, "r":rw_test, "nbandit":3, "model":models[best_idx[idxx]], "generate_choices":0, "choices":ch_test} 
            Mbest = lklhd_choice_m(P[model_names[best_idx[idxx]]], indata)
            acc.append(Mbest["acc"])

        # Gather data
        D = {"noise": noise, "cutoff":c, "true_model": mname, "algo":alg,
            "best_model_AIC": model_names[best_idx[0]], 
            "best_model_AICc": model_names[best_idx[1]],  
            "best_model_BIC": model_names[best_idx[2]], 
            "best_model_HQC": model_names[best_idx[3]], 
            "acc_AIC": acc[0],
            "acc_AICc": acc[1],
            "acc_BIC": acc[2], 
            "acc_HQC": acc[3]                  
            }
        dfrow = pd.DataFrame.from_dict(D, orient="index").T
        df = pd.concat([df, dfrow], axis=0)
df.to_csv(os.path.join(rf, "data", cond, "model_comparison_iter"+cond_str+".csv") )