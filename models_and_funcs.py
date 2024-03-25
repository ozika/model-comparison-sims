import numpy as np
from random import random
import scipy.stats as stats
import pandas as pd
import itertools as it 

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

def gen_bandits(N=180, nbandit=3, noise=2):

    rewards = np.zeros((N,nbandit))*np.nan 
    rewards[:,0] = gen_states( lvls=[20, 20], ch=20, n=N) + np.random.normal(0, noise, N) 
    rewards[:,1] = gen_states( lvls=[20, 80], ch=20, n=N) + np.random.normal(0, noise, N) 
    rewards[:,2] = gen_states( lvls=[80, 80], ch=20, n=N) + np.random.normal(0, noise, N) 

    c = pd.DataFrame(list(it.combinations([x for x in range(nbandit)], 2)))
    reps = N//c.shape[0]
    repeated_df = pd.concat([c] * reps, ignore_index=True)
    repeated_df.columns = ["option1", "option2"]

    return rewards, repeated_df

def softmax(values, beta, parametrization="inverse"):
  denom = 0   
  if parametrization=="inverse":
    # expected rnage [0 Inf] igher values make choices more equal
    for v in values:
      denom = denom + np.exp(v/beta) 
    probs = np.exp(values/beta) / (denom)
  #http://incompleteideas.net/book/ebook/node17.html
  elif parametrization=="log_inverse":
    for v in values:
      denom = denom + np.exp(v/np.log(beta)) 
    probs = np.exp(values/np.log(beta)) / (denom)
  elif parametrization=="normal":
    for v in values:
      denom = denom + np.exp(v*beta) 
    probs = np.exp(values*beta) / (denom)
  return(probs)


def rw1_choice(params=[0.2], indata=[]):
    alpha = params[0]  # p[1] ... learning rate
    beta = params[1] # inverse temperature
    Qs = np.zeros((indata["r"].shape[0]+1,indata["r"].shape[1]))*np.nan
    Qs[0,0:3] = 50
    ntr = indata["r"].shape[0]
    choices = np.zeros((indata["r"].shape[0]))*np.nan
    ch_prob = np.zeros((indata["r"].shape[0]))*np.nan
    for tr in range(ntr):
        # get options available on this trial
        op1 = indata["options"]["option1"].iloc[tr] 
        op2 = indata["options"]["option2"].iloc[tr] 
        
        #values on this trial 
        values = Qs[tr,[op1, op2]]
        
        # choice
        probs = softmax(values, beta, parametrization="inverse")


        if any(np.isnan(probs)) | any(np.isinf(probs)):
           print(indata["model"])
           print(params)
           print(probs)
           print(values)
           probs = np.array([0.5, 0.5])
           stop= 1
           # generate choices or use probabilities to make choices
           if indata["generate_choices"] == 1:
               choice = np.random.choice([op1, op2])
           else: 
               choice = int(indata["choices"][tr] )
        else:
           # generate choices or use probabilities to make choices
            if indata["generate_choices"] == 1:
                choice = np.random.choice([op1, op2], p=probs)
            else: 
                choice = int(indata["choices"][tr] )

        # probability of choice (for likelihood)  
        if choice == op1: 
           chosen_prob = probs[0]
        elif choice == op2: 
           chosen_prob = probs[1]
        choices[tr] = choice
        ch_prob[tr] = chosen_prob

        # update chosen
        Qs[tr+1,choice] = Qs[tr,choice] + alpha*(indata["r"][tr, choice] - Qs[tr,choice])

        #update unchosen 
        for ch in list(set(list([0,1,2])) - set(list([choice]))):
           Qs[tr+1,ch] = Qs[tr,ch]

        #Q.append(Q[o_idx] + alpha*(o - Q[o_idx])   )
    mod = {"Qs": Qs, "choices":choices, "choice_prob":ch_prob}        
    return mod

def rw2_val_choice(params=[0.2, 0.2, 1], indata=[]):
    alpha_pos = params[0]
    alpha_neg = params[1]

    beta = params[2] # inverse temperature
    Qs = np.zeros((indata["r"].shape[0]+1,indata["r"].shape[1]))*np.nan
    Qs[0,0:3] = 50
    ntr = indata["r"].shape[0]
    choices = np.zeros((indata["r"].shape[0]))*np.nan
    ch_prob = np.zeros((indata["r"].shape[0]))*np.nan
    for tr in range(ntr):
        # get options available on this trial
        op1 = indata["options"]["option1"].iloc[tr] 
        op2 = indata["options"]["option2"].iloc[tr] 
        
        #values on this trial 
        values = Qs[tr,[op1, op2]]
        
        # choice
        probs = softmax(values, beta, parametrization="inverse")

        if any(np.isnan(probs)) | any(np.isinf(probs)):
           print(indata["model"])
           print(params)
           print(probs)
           print(values)
           probs = np.array([0.5, 0.5])
           stop= 1
           # generate choices or use probabilities to make choices
           if indata["generate_choices"] == 1:
               choice = np.random.choice([op1, op2])
           else: 
               choice = int(indata["choices"][tr] )
        else:
           # generate choices or use probabilities to make choices
            if indata["generate_choices"] == 1:
                choice = np.random.choice([op1, op2], p=probs)
            else: 
                choice = int(indata["choices"][tr] )

        # probability of choice (for likelihood)  
        if choice == op1: 
           chosen_prob = probs[0]
        elif choice == op2: 
           chosen_prob = probs[1]
        choices[tr] = choice
        ch_prob[tr] = chosen_prob

        # update chosen
        if (indata["r"][tr, choice] - Qs[tr,choice]) > 0:
          Qs[tr+1,choice] = Qs[tr,choice] + alpha_pos*(indata["r"][tr, choice] - Qs[tr,choice])
        elif (indata["r"][tr, choice] - Qs[tr,choice]) <= 0:
           Qs[tr+1,choice] = Qs[tr,choice] + alpha_neg*(indata["r"][tr, choice] - Qs[tr,choice])

        #update unchosen 
        for ch in list(set(list([0,1,2])) - set(list([choice]))):
           Qs[tr+1,ch] = Qs[tr,ch]

        #Q.append(Q[o_idx] + alpha*(o - Q[o_idx])   )
    mod = {"Qs": Qs, "choices":choices, "choice_prob":ch_prob}        
    return mod

def rw3_choice(params=[0.2, 0.2, 0.2, 5], indata=[]):
    # model with separate alphas for the three cues
    alpha = params[0:3]
    #alpha[0] = params[0]  # p[1] ... learning rate
    #alpha[1]= params[1]  # p[1] ... learning rate
    #alpha[2] = params[2]  # p[1] ... learning rate
    beta = params[3] # inverse temperature
    Qs = np.zeros((indata["r"].shape[0]+1,indata["r"].shape[1]))*np.nan
    Qs[0,0:3] = 50
    ntr = indata["r"].shape[0]
    choices = np.zeros((indata["r"].shape[0]))*np.nan
    ch_prob = np.zeros((indata["r"].shape[0]))*np.nan
    for tr in range(ntr):
        # get options available on this trial
        op1 = indata["options"]["option1"].iloc[tr] 
        op2 = indata["options"]["option2"].iloc[tr] 
        
        #values on this trial 
        values = Qs[tr,[op1, op2]]
        
        # choice
        probs = softmax(values, beta, parametrization="inverse")

        if any(np.isnan(probs)) | any(np.isinf(probs)):
           print(indata["model"])
           print(params)
           print(probs)
           print(values)
           probs = np.array([0.5, 0.5])
           stop= 1
           # generate choices or use probabilities to make choices
           if indata["generate_choices"] == 1:
               choice = np.random.choice([op1, op2])
           else: 
               choice = int(indata["choices"][tr] )
        else:
           # generate choices or use probabilities to make choices
            if indata["generate_choices"] == 1:
                choice = np.random.choice([op1, op2], p=probs)
            else: 
                choice = int(indata["choices"][tr] )

        # probability of choice (for likelihood)  
        if choice == op1: 
           chosen_prob = probs[0]
        elif choice == op2: 
           chosen_prob = probs[1]
        choices[tr] = choice
        ch_prob[tr] = chosen_prob

        # update chosen
        Qs[tr+1,choice] = Qs[tr,choice] + alpha[choice]*(indata["r"][tr, choice] - Qs[tr,choice])

        #update unchosen 
        for ch in list(set(list([0,1,2])) - set(list([choice]))):
           Qs[tr+1,ch] = Qs[tr,ch]

        #Q.append(Q[o_idx] + alpha*(o - Q[o_idx])   )
    mod = {"Qs": Qs, "choices":choices, "choice_prob":ch_prob}        
    return mod

def rw6_val_choice(params=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 5], indata=[]):
    # model with separate alphas for the three cues
    alpha_pos = params[0:3]
    alpha_neg = params[3:6]
    beta = params[6] # inverse temperature
    Qs = np.zeros((indata["r"].shape[0]+1,indata["r"].shape[1]))*np.nan
    Qs[0,0:3] = 50
    ntr = indata["r"].shape[0]
    choices = np.zeros((indata["r"].shape[0]))*np.nan
    ch_prob = np.zeros((indata["r"].shape[0]))*np.nan
    for tr in range(ntr):
        # get options available on this trial
        op1 = indata["options"]["option1"].iloc[tr] 
        op2 = indata["options"]["option2"].iloc[tr] 
        
        #values on this trial 
        values = Qs[tr,[op1, op2]]
        
        # choice
        probs = softmax(values, beta, parametrization="inverse")

        if any(np.isnan(probs)) | any(np.isinf(probs)):
           print(indata["model"])
           print(params)
           print(probs)
           print(values)
           probs = np.array([0.5, 0.5])
           stop= 1
           # generate choices or use probabilities to make choices
           if indata["generate_choices"] == 1:
               choice = np.random.choice([op1, op2])
           else: 
               choice = int(indata["choices"][tr] )
        else:
           # generate choices or use probabilities to make choices
            if indata["generate_choices"] == 1:
                choice = np.random.choice([op1, op2], p=probs)
            else: 
                choice = int(indata["choices"][tr] )
           
        

        # probability of choice (for likelihood)  
        if choice == op1: 
           chosen_prob = probs[0]
        elif choice == op2: 
           chosen_prob = probs[1]
        choices[tr] = choice
        ch_prob[tr] = chosen_prob

        # update chosen
        if (indata["r"][tr, choice] - Qs[tr,choice]) > 0:
          Qs[tr+1,choice] = Qs[tr,choice] + alpha_pos[choice]*(indata["r"][tr, choice] - Qs[tr,choice])
        elif (indata["r"][tr, choice] - Qs[tr,choice]) <= 0:
           Qs[tr+1,choice] = Qs[tr,choice] + alpha_neg[choice]*(indata["r"][tr, choice] - Qs[tr,choice])

        #update unchosen 
        for ch in list(set(list([0,1,2])) - set(list([choice]))):
           Qs[tr+1,ch] = Qs[tr,ch]

        #Q.append(Q[o_idx] + alpha*(o - Q[o_idx])   )
    mod = {"Qs": Qs, "choices":choices, "choice_prob":ch_prob}        
    return mod



def lklhd_choice(params, indata):
    model = indata["model"]
    m1 = model(params, indata)
    ll=np.sum(-np.log(m1["choice_prob"]))
    return ll


def lklhd_choice_m(params, indata):
    # 1/ evaluate model given participant's choices 
    model = indata["model"]
    indata["data_choices"] = indata["choices"]
    m1 = model(params, indata)
    negLL=np.sum(-np.log(m1["choice_prob"]))
   
    n = indata["r"].shape[0]
    m1["n"] = n 
    
    m1["negLL"] = negLL
    noparams = len(params)
    m1["noparams"] = noparams
    m1["AIC"] = 2*noparams + 2*negLL #ll is already negative log, thus +
    
    
    m1["AICc"] = m1["AIC"] + ((2*noparams**2 + 2*noparams ) / (n - noparams - 1))
    m1["BIC"] = noparams*np.log(n) + 2*negLL #ll is already negative log, thus +
    m1["HQC"] = 2*negLL + 2*noparams*np.log(np.log(n))

    # 2/ use parameters to get choice accuracy measure
    indata2=indata 
    indata2["generate_choices"]=1
    m2 = model(params, indata2)
    m1["acc"] = np.mean(m2["choices"] == indata["data_choices"])
    return m1

def replace_random_values(arr, options_df, proportion):
    num_to_replace = int(len(arr) * proportion)
    indices_to_replace = np.random.choice(len(arr), num_to_replace, replace=False)

    for i, idx in enumerate(indices_to_replace):
        option1, option2 = options_df.loc[idx, ["option1", "option2"]]
        random_option = np.random.choice([option1, option2])
        arr[idx] = random_option

    return arr