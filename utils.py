import random
import math
from tqdm import tqdm
import pandas as pd

arms = ["low", "medium", "high"]

def dose2str(dose):
    if dose < 21:
        return "low"
    elif dose > 49:
        return "high"
    else:
        return "medium"

def dose2int(dose):
    if dose < 21:
        return 0
    elif dose > 49:
        return 2
    else:
        return 1
def arm2dose(arm_ind): #TODO: tune this to make sense.
    if arm_ind == 0:
        return 21
    elif arm_ind == 2:
        return 49
    else:
        return 35

def is_correct(arm_ind, actual_dose):
    return dose2int(actual_dose) == arm_ind

def is_fuzz_correct(arm_ind, actual_dose, eps = 7):
    boundaries = [[0,21], [21, 49], [49, 1000]]
    (lower,upper) = boundaries[arm_ind]

    return lower - eps <= actual_dose <= upper + eps

def get_sample_order(n):
    lst = list(range(n))
    random.shuffle(lst)
    return lst

def run_iters(n, func):
    res = []
    for _ in tqdm(range(n)):
        res.append(func())
    res_df = pd.DataFrame(res)
    return res_df

def calculate_reward(arm_ind, y, style, p_vals=None):
    if style == "standard":
        reward = 0 if arms[arm_ind] == dose2str(y) else -1
        regret = -reward
    elif style == "risk-sensitive":
        #TODO tune this
        if abs(dose2int(y) - arm_ind) == 2:
            reward = -2
        elif abs(dose2int(y) - arm_ind) == 1:
            reward = -1
        else:
            reward = 0
        regret = -reward
    elif style == "prob-based":
        correct_arm = dose2int(y)
        if max(p_vals) == min(p_vals):
            reward = 0
        else:
            reward = (p_vals[correct_arm] - p_vals[arm_ind]) / (max(p_vals) - min(p_vals))
        regret = -reward
    elif style == "proportional":
        #reward is distance between dose and arm pulled (arm pulled dose chosen wrt arm2dose)
        #TODO tune this 
        reward = -abs(y - arm2dose(arm_ind))/7
        regret = -reward

    return regret, reward
