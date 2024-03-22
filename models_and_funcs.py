import numpy as np
from random import random
import scipy.stats as stats

def gen_states(lvls=[20,80], ch=30, n=120):
    reps = int(n/ch)
    vec = np.array([])
    lvlidx = 0
    for i in range(reps):
        vec = np.append(vec, np.zeros(ch)+lvls[lvlidx])
        if lvlidx == 0:
            lvlidx = 1
        elif lvlidx == 1:
            lvlidx = 0
    return vec

def rw1(params, odata=[]):
    Q0 = params[0] # p[0] ...starting value
    alpha = params[1]  # p[2] ... learning rate
    Q = [Q0]
    for o_idx, o in enumerate(odata["o"]):
        Q.append(Q[o_idx] + alpha*(o - Q[o_idx])   )
    mod = {"Q": Q}        
    return mod

def rw1_att(params, odata=[]):
    # "attention model" updating weighted by reward magnitude
    Q0 = params[0] # p[0] ...starting value
    alpha = params[1]  # p[2] ... learning rate
    Q = [Q0]
    for o_idx, o in enumerate(odata["o"]):
        Q.append(Q[o_idx] + (Q[o_idx]/100)*alpha*(o - Q[o_idx])   )
    mod = {"Q": Q}        
    return mod

def rw2_val(params, odata=[]):
    Q0 = params[0] # p[0] ...starting value
    alpha_pos = params[1]  # p[1] ... learning rate for positive PEs
    alpha_neg = params[2]  # p[2] ... learning rate for negative PEs
    Q = [Q0]
    for o_idx, o in enumerate(odata["o"]):
        if o - Q[o_idx] > 0:
            Q.append(Q[o_idx] + alpha_pos*(o - Q[o_idx])   )
        elif o - Q[o_idx] <= 0:
            Q.append(Q[o_idx] + alpha_neg*(o - Q[o_idx])   )
    mod = {"Q": Q}        
    return mod

def ph_basic(params, odata=[]):
    Q0 = params[0] # p[0] ...starting value
    alpha0 = params[1]  # p[1] ... starting learning rate
    pi = params[2]  # p[2] ... associability
    al = [alpha0]
    Q = [Q0]
    for o_idx, o in enumerate(odata["o"]):
        abspe = np.abs(o - Q[o_idx])/100
        al.append( (1-pi)*al[o_idx-1] + pi*(abspe)  ) 
        Q.append(Q[o_idx] + al[o_idx]*(o - Q[o_idx])    )
    mod = {"Q": Q, "al":al}        
    return mod

def rw2_state(params, odata=[]):
    Q0 = params[0] # p[0] ...starting value
    alpha_high = params[1]  # p[1] ... learning rate for high state
    alpha_low = params[2]  # p[2] ... learning rate for low state
    Q = [Q0]
    for o_idx, o in enumerate(odata["o"]):
        if odata["state"][o_idx] == 0:
            Q.append(Q[o_idx] + alpha_low*(o - Q[o_idx])   )
        elif odata["state"][o_idx] == 1:
            Q.append(Q[o_idx] + alpha_high*(o - Q[o_idx])   )
    mod = {"Q": Q}        
    return mod

def rw4_val_state(params, odata=[]):
    Q0 = params[0] # p[0] ...starting value
    alpha_pos_high = params[1]  #
    alpha_pos_low = params[2]  #
    alpha_neg_high = params[3]  #
    alpha_neg_low = params[4]  #

    Q = [Q0]
    for o_idx, o in enumerate(odata["o"]):
        if o - Q[o_idx] > 0:
            if odata["state"][o_idx] == 0:
                Q.append(Q[o_idx] + alpha_pos_low*(o - Q[o_idx])   )
            elif odata["state"][o_idx] == 1:
                Q.append(Q[o_idx] + alpha_pos_high*(o - Q[o_idx])   )
        elif o - Q[o_idx] <= 0:
            if odata["state"][o_idx] == 0:
                Q.append(Q[o_idx] + alpha_neg_low*(o - Q[o_idx])   )
            elif odata["state"][o_idx] == 1:
                Q.append(Q[o_idx] + alpha_neg_high*(o - Q[o_idx])   )


    mod = {"Q": Q}        
    return mod

def lklhd(params, data, odata):
    model = odata["model"]
    m1 = model(params, odata)
    nglklhd = []
    for idx,outcm in enumerate(odata["o"]):
        # using gaussian likelihood
        cp  = stats.norm(m1["Q"][idx], 20).cdf([data[idx]-0.01, data[idx]+0.01])
        lklhd = -np.log(cp[1]-cp[0])
        if np.isinf(lklhd) | np.isnan(lklhd):
            lklhd = 9999
        nglklhd.append(lklhd)
    ll=sum(nglklhd)
    return ll

def lklhd_m(params, data,odata):
    model = odata["model"]
    m1 = model(params, odata)
    nglklhd = []
    rse = []
    abserr = []
    for idx,outcm in enumerate(odata["o"]):
        # using gaussian likelihood
        cp  = stats.norm(m1["Q"][idx], 20).cdf([data[idx]-0.01, data[idx]+0.01])
        lklhd = -np.log(cp[1]-cp[0])
        abserr.append(abs(data[idx] - m1["Q"][idx]))
        rse.append(np.sqrt(pow(data[idx] - m1["Q"][idx], 2)))
        if np.isinf(lklhd) | np.isnan(lklhd):
            lklhd = 9999
        nglklhd.append(lklhd)
    ll=sum(nglklhd)
    n = len(odata["o"])
    m1["abserr"] = np.sum(abserr)
    m1["n"] = n 
    m1["err_per_n"] = m1["abserr"] / m1["n"] 
    m1["rmse"] = np.sum(rse) / n
    
    m1["nglklhd"] = nglklhd
    m1["negLL"] = ll
    noparams = len(params)
    m1["noparams"] = noparams
    m1["AIC"] = 2*noparams + 2*ll #ll is already negative log, thus +
    
    
    m1["AICc"] = m1["AIC"] + ((2*noparams**2 + 2*noparams ) / (n - noparams - 1))
    m1["BIC"] = noparams*np.log(n) + 2*ll #ll is already negative log, thus +
    m1["HQC"] = 2*ll + 2*noparams*np.log(np.log(n))
    
    return m1

def gen_rand_vals(bounds):
    b = []
    for p in bounds: 
        b.append(p[0] + (random() * (p[1] - p[0])))
    return b