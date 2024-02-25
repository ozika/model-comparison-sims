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


iterations = 2
N = 180
noise_level= [5, 10, 20]
cutoff = [90, 120, 150]
models = [rw1, rw1_att, rw2_val, ph_basic] 
model_names = ["rw1", "rw1_att", "rw2_val", "ph_basic"]
bounds = {"rw1": ((0,100),(0,1)),
          "rw1_att": ((0,100),(0,1)), 
          "rw2_val": ((0,100),(0,1),(0,1)),
          "ph_basic": ((0,100),(0,1),(0,1))}
df = pd.DataFrame()
fitting_algos = ["COBYLA", "Nelder-Mead"]

for alg in fitting_algos:
    for i_idx, noise in enumerate(noise_level):
        for c in cutoff:
            o_all = gen_states( lvls=[20,80], ch=20, n=N)+np.random.normal(0, noise, N) 
            lvl_all = gen_states( lvls=[0,1], ch=20, n=N)

            o = o_all[0:c]
            lvl = lvl_all[0:c]

            o_test = o_all[c:]
            lvl_test = lvl_all[c:]
            for ii in range(iterations):
                # split data 
                for m_idx, (m, mname) in enumerate(zip(models, model_names)): # loop over model GENERATING the data
                    params_in = gen_rand_vals(bounds[mname])
                    rewards = m(params_in, {"o": o_all, "model": m, "state":lvl_all})
                    rew_test = rewards["Q"][c:]
                    rewards["Q"] = rewards["Q"][0:c]


                    # Fit data 
                    AIC =[] 
                    BIC = []
                    AICc = [] 
                    P = {}
                    for mfit, mname_fit in zip(models, model_names):
                        other_data = {"o": o, "model": mfit, "state":lvl}
                        opt = minimize(fun=lklhd, x0=gen_rand_vals(bounds[mname_fit]), args=(rewards["Q"],other_data), method='COBYLA', bounds=bounds[mname_fit], options={'verbose': 0})

                        # Get IC 
                        M = lklhd_m(opt.x, rewards["Q"],other_data)
                        AIC.append(M["AIC"])
                        P[mname_fit] = opt.x
                        BIC.append(M["BIC"])
                        AICc.append(M["AICc"])
                    
                    # which model fit best
                    best_idx =[np.argmin(AIC), np.argmin(AICc), np.argmin(BIC)]
                    # is this the correct model
                    #if aic_idx == m_idx:
                    #    m_recovery_AIC = 1 # etc

                    # get predictive error per trial of the best model
                    pred_err = []
                    for idxx, best in enumerate(best_idx):
                        other_data = {"o": o_test, "model": models[best_idx[idxx]], "state":lvl_test}
                        Mbest = lklhd_m(P[model_names[best_idx[idxx]]], rew_test, other_data)
                        pred_err.append(Mbest["err_per_n"])

                    # Gather data
                    D = {"noise": noise, "cutoff":c, "true_model": mname, "algo":alg,
                        "best_model_AIC": model_names[best_idx[0]], 
                        "best_model_AICc": model_names[best_idx[1]],  
                        "best_model_BIC": model_names[best_idx[2]], 
                        "mean_err_AIC": pred_err[0],
                        "mean_err_AICc": pred_err[1],
                        "mean_err_BIC": pred_err[2]               
                        }
                    dfrow = pd.DataFrame.from_dict(D, orient="index").T
                    df = pd.concat([df, dfrow], axis=0)
df.to_csv(os.path.join(rf, "data", "model_comparison_iter"+str(iterations)+".csv") )