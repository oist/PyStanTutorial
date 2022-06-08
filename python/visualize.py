import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ExponentialMovingAverage(choice, tau=8, init_value=0.5):
    value = init_value
    EMA = np.zeros(len(choice))
    m = np.exp(-1/tau)
    n = 1 - m
    
    for i in range(len(choice)):
        value = (value * m) + (n * choice[i])
        EMA[i] = value
    return EMA

def plot_two_choice_data(df):
    # if session_type = 1, photostimulation was given
    actions = np.array(df["Choice (0: Left, 1: Right)"])
    rewards = np.array(df["Outcome (0: Omission, 1: Rewarded)"])
    rp_left = np.array(df["Reward Probability of the Left"])
    rp_right = np.array(df["Reward Probability of the Right"])
    MA = ExponentialMovingAverage(actions, tau=8, init_value=0.5)
    n_trial    = len(actions)
    r_rewarded_ys = []
    r_rewarded_ts = []
    l_rewarded_ys = []
    l_rewarded_ts = []
    r_unrewarded_ys = []
    r_unrewarded_ts = []
    l_unrewarded_ys = []
    l_unrewarded_ts = []
    for t in range(n_trial):
        if rewards[t] > 0:
            if actions[t] == 0:
                l_rewarded_ys.append(-0.2)
                l_rewarded_ts.append(t)
            else:
                r_rewarded_ys.append(1.2)
                r_rewarded_ts.append(t)
        else:
            if actions[t] == 0:
                l_unrewarded_ys.append(-0.1)
                l_unrewarded_ts.append(t)
            else:
                r_unrewarded_ys.append(1.1)
                r_unrewarded_ts.append(t)
    plt.figure(figsize=(14, 7))
    plt.plot(range(n_trial), MA, '-g', linewidth = 1, label='Exponential Moving average')
    plt.plot(range(n_trial), rp_left, label="Reward Prob of Left Choice")
    plt.plot(range(n_trial), rp_right, label="Reward Prob of Right Choice")
    plt.plot(r_rewarded_ts, r_rewarded_ys, 'ob', markersize =4, label ='Right Rewarded')
    plt.plot(l_rewarded_ts, l_rewarded_ys, 'or', markersize =4, label ='Left Rewarded')  
    plt.plot(r_unrewarded_ts, r_unrewarded_ys, 'xb', markersize =5, label ='Right Non-Rewarded')
    plt.plot(l_unrewarded_ts, l_unrewarded_ys, 'xr', markersize =5, label ='Left Non-Rewarded')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Trials")
    plt.ylabel("Probability of Right Choice")
    plt.show()

def plot_two_step_data(session):
    # if session_type = 1, photostimulation was given
    b = session.print_lines
    trial = b[0].split(" ")
    n_trial    = len(b)

    trans_state = int(trial[8][3] == 'A') # 0 = B, 1 = A
    choice     = np.zeros(n_trial, dtype = int) 
    second_state = np.zeros(n_trial, dtype = int) 
    outcome = np.zeros(n_trial, dtype = int) 
    exp_mov    = np.zeros(n_trial, dtype = float)
    trial_type = np.zeros(n_trial, dtype = int)
    reward_state = np.zeros(n_trial, dtype = int)


    for i in range(n_trial):
        d = b[i].split(" ")
        if d[4] == 'C:1': # 0 = right, 1 = left
            choice[i] = 1
        if d[5] == 'S:1': # 0 = down, 1 = up 
            second_state[i] = 1
        if d[6] == 'O:1': # 0 = unrewarded, 1 = rewarded
            outcome[i] = 1        
        exp_mov[i] = float(d[7][3:8])
        if d[8][2] == 'D':
            reward_state[i] = 0
        elif d[8][2] == 'U':
            reward_state[i] = 1
        else:
            reward_state[i] = 2
        if d[9] == 'CT:FC': # 2 = free choice, 1 = left forced choice, 2 = right forced choice
            trial_type[i] = 2
        elif d[9] == 'CT:L':
            trial_type[i] = 1
        else:
            trial_type[i] = 0
    
    block_info = {'block_type':[], 'start':[0], 'end':[], 'block_length':[]}
    block_counter = 0

    for i in range (n_trial-1):
        rew_state = reward_state[i]
        if rew_state != reward_state[i+1]:
            if rew_state == 1:
                block_info['block_type'].append('Up')
            elif rew_state == 2:
                block_info['block_type'].append('Neutral')
            else:    
                block_info['block_type'].append('Down')

            block_info['end'].append(i)
            block_info['start'].append(i+1)
            length = (block_info['end'][block_counter] - block_info['start'][block_counter]) + 1         
            block_info['block_length'].append(length)
            block_counter += 1
        if i == n_trial-2:
            block_info['end'].append(n_trial-1)
            length = (block_info['end'][block_counter] - block_info['start'][block_counter]) + 1
            block_info['block_length'].append(length)
            if rew_state == 1:
                block_info['block_type'].append('Up')
            elif rew_state == 2:
                block_info['block_type'].append('Neutral')
            else:    
                block_info['block_type'].append('Down')    

    MA = ExponentialMovingAverage(choice, tau=8, init_value=0.5)

    n_block = len(block_info['block_type'])
    second_prob = np.zeros(len(choice))
    oRRloc = np.zeros(len(choice))
    oRNloc = np.zeros(len(choice))
    oLRloc = np.zeros(len(choice))
    oLNloc = np.zeros(len(choice))    
    ttlocRR = np.zeros(len(choice)) - 0.2
    ttlocLR = np.zeros(len(choice)) - 0.2
    ttlocRN = np.zeros(len(choice)) - 0.2
    ttlocLN = np.zeros(len(choice)) - 0.2    
    for i in range(n_block):
        if i  < n_block-1:
            start = block_info['start'][i]
            end = block_info['end'][i]
            if block_info['block_type'][i] == 'Up':
                if trans_state:
                    second_prob[start:end+1] = 0.8
                else:
                    second_prob[start:end+1] = 0.2
            elif block_info['block_type'][i] == 'Neutral':
                second_prob[start:end+1] = 0.5
            else:
                if trans_state:
                    second_prob[start:end+1] = 0.2
                else:
                    second_prob[start:end+1] = 0.8
        if i == n_block-1:
            start = block_info['start'][i]
            if block_info['block_type'][i] == 'Up':
                if trans_state:
                    second_prob[start:] = 0.8
                else:
                    second_prob[start:] = 0.2
            elif block_info['block_type'][i] == 'Neutral':
                second_prob[start:] = 0.5
            else:
                if trans_state:
                    second_prob[start:] = 0.2
                else:
                    second_prob[start:] = 0.8    
    
    for i in range(len(outcome)):
        if choice[i] == 1:
            oRRloc[i] = -0.2
            oRNloc[i] = -0.2
            if outcome[i] == 1:
                oLRloc[i] = 1.0
                oLNloc[i] = -0.2
                if trial_type[i] == 1:
                    ttlocLR[i] = MA[i]
                    oLRloc[i] = -0.2
            else:
                oLNloc[i] = 0.95
                oLRloc[i] = -0.2
                if trial_type[i] == 1:
                    ttlocLN[i] = MA[i]
                    oLNloc[i] = -0.2
        elif choice[i] == 0: 
            oLRloc[i] = -0.2
            oLNloc[i] = -0.2
            if outcome[i] == 1:
                oRRloc[i] = 0.0
                oRNloc[i] = -0.2
                if trial_type[i] == 0:
                    ttlocRR[i] = MA[i]
                    oRRloc[i] = -0.2
            else:
                oRNloc[i] = 0.05
                oRRloc[i] = -0.2
                if trial_type[i] ==0:
                    ttlocRN[i] = MA[i]
                    oRNloc[i] = -0.2
                    
    
    cm = 1/2.54
    x = np.arange(len(choice))+1
    fig,ax = plt.subplots(figsize = (12.56,9))
       
    ax.plot(x,MA, '-k', linewidth = 2)
    ax.plot(x,oRRloc, 'ob', markersize =4, label ='Right Rewarded')
    ax.plot(x,oLRloc, 'or', markersize =4, label ='Left Rewarded')  
    ax.plot(x,oRNloc, 'xb', markersize =5, label ='Right Non-Rewarded')
    ax.plot(x,oLNloc, 'xr', markersize =5, label ='Left Non-Rewarded')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=14,frameon=False)
    ax.set_ylabel('Exponential Moving average',fontsize=14)  
    ax.tick_params(axis='both', which='major', labelsize=14)    
    ax.set_ylim([-0.02,1.05])
    ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.set_xlabel('# trials',fontsize=14)
    
    ax2=ax.twinx()   
    ax2.plot(x,second_prob,'-g', linewidth = 2)
    ax2.tick_params(axis='both', which='major', labelsize=14)        
    ax2.set_xlim([1,x[-1]])
    ax2.set_ylim([-0.02,1.05])
    ax2.set_yticks([0.2,0.5,0.8])
    ax2.set_ylabel('Reward probability ($p_{1}$)',color="green",fontsize=14)