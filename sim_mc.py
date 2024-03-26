
import argparse as ap

p = ap.ArgumentParser(description="Run model comparison")
p.add_argument('-a', action='store') # algorithm 
p.add_argument('-n', action='store') # noise
p.add_argument('-v', action='store') # noise
p.add_argument('-c', action='store') # cutoff
p.add_argument('-i', action='store') # number of iterations
p.add_argument('-d', action='store') # number of iterations
args = p.parse_args()

cond = str(args.d)
alg = str(args.a) 
noise = int(args.n )
value_noise = int(args.v )
c = int(args.c)
iterations = int(args.i)
cond_str = "mc_n"+str(noise)+"vn"+str(value_noise)+"_c"+str(c)+"_i"+str(iterations)+"_"+alg

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
models = [rw1, rw1_att, rw2_val, ph_basic] 
model_names = ["rw1", "rw1_att", "rw2_val", "ph_basic"]
bounds = {"rw1": ((0,100),(0,1)),
          "rw1_att": ((0,100),(0,1)), 
          "rw2_val": ((0,100),(0,1),(0,1)),
          "ph_basic": ((0,100),(0,1),(0,1))}
df = pd.DataFrame()


for ii in range(iterations):
    print(np.round(ii/iterations,2))
    # Generate data
    r_all = gaus_walk(N=180, vol = 5, noise=noise)
    #o_all = gen_states( lvls=[20,80], ch=20, n=N)+np.random.normal(0, noise, N) 


    r_train = r_all[0:c]
    r_test = r_all[c:]

    for m_idx, (m, mname) in enumerate(zip(models, model_names)): # loop over model GENERATING the data
        params_in = gen_rand_vals(bounds[mname])
        full_pred = m(params_in, {"o": r_all, "model": m})

        # add noise to values
        full_pred["Q"] = full_pred["Q"] + np.random.normal(0, value_noise, N+2) 

        value_train = full_pred["Q"][0:c]
        value_test = full_pred["Q"][c:]
        
        # Fit data 
        AIC =[] 
        BIC = []
        AICc = [] 
        HQC = [] #https://en.wikipedia.org/wiki/Hannan%E2%80%93Quinn_information_criterion
        P = {}
        for mfit, mname_fit in zip(models, model_names):
            other_data = {"o": r_train, "model": mfit, "bounds": bounds, "alg":alg, "model_name":mname_fit}
            opt = minimize(fun=lklhd, x0=gen_rand_vals(bounds[mname_fit]), args=(value_train,other_data), method=alg, bounds=bounds[mname_fit], options={'verbose': 0})

            #cv_res = custom_CV(other_data)

            # Get IC 
            M = lklhd_m(opt.x, value_train, other_data)
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
        pred_err = []
        for idxx, best in enumerate(best_idx):
            test_data = {"o": r_test, "model": models[best_idx[idxx]]}
            Mbest = lklhd_m(P[model_names[best_idx[idxx]]], value_test, test_data)
            pred_err.append(Mbest["err_per_n"])

        # Gather data
        D = {"noise": noise, "cutoff":c, "true_model": mname, "algo":alg,
            "best_model_AIC": model_names[best_idx[0]], 
            "best_model_AICc": model_names[best_idx[1]],  
            "best_model_BIC": model_names[best_idx[2]], 
            "best_model_HQC": model_names[best_idx[3]], 
            "mean_err_AIC": pred_err[0],
            "mean_err_AICc": pred_err[1],
            "mean_err_BIC": pred_err[2], 
            "mean_err_HQC": pred_err[3]                  
            }
        dfrow = pd.DataFrame.from_dict(D, orient="index").T
        df = pd.concat([df, dfrow], axis=0)
df.to_csv(os.path.join(rf, "data", cond, "model_comparison_iter"+cond_str+".csv") )