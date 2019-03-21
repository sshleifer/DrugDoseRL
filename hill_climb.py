import numpy as np
import math
from tqdm import tqdm
from get_features import get_features
from utils import dose2str, get_sample_order, run_iters, is_correct, is_fuzz_correct, show_hist
from linucb import UCB_FEATURES, LinUCB

arms = ["low", "medium", "high"]


def calculate_reward(arm_ind, y, new_reward):
    if is_correct(arm_ind, y):
        reward = 0
    elif is_fuzz_correct(arm_ind, y):
        reward = -0.4375
    else:
        reward = -0.88125
    regret = -reward
    return regret, reward


def hill_climb(new_reward):
    show_history = False
    reward_style = "standard"

    X, y = get_features()
    X['bias'] = 1
    feature_select = UCB_FEATURES
    X_subset = X[feature_select].astype(float)
    num_features = len(feature_select)
    linucb = LinUCB(num_features, X_subset.shape[0])

    total_regret = 0
    num_correct = 0
    num_fuzz_correct = 0
    hist = []

    for i in tqdm(range(X_subset.shape[0])):
        row_num = linucb.order[i]
        features = np.array(X_subset.iloc[row_num])
        arm, p_vals = linucb.select_arm(features)
        dose = y.iloc[row_num]
        regret, reward = calculate_reward(arm, dose, new_reward)
        
        num_correct += 1 if is_correct(arm, dose) else 0
        num_fuzz_correct += 1 if is_fuzz_correct(arm, dose, eps=7) else 0
        total_regret += regret

        linucb.update(arm, reward, features)
        hist.append(arm)

    if show_history:
        show_hist(hist)

    acc = num_correct / X.shape[0]
    fuzz_acc = num_fuzz_correct / X.shape[0]
    return fuzz_acc

if __name__ == "__main__":
    new_reward = -0.4375
    step_size = 0.1
    num_iters = 20
    prev_acc = 0
    multiplier = -1 

    while step_size > 0.005:
    	print("Trying with intermediate reward of: " + str(new_reward))
    	cur_accs = []
    	for j in range(10):
    		cur_accs.append(hill_climb(new_reward))
    	cur_acc = np.mean(cur_accs)
    	print("Fuzzy accuracy was: " + str(cur_acc))
    	if cur_acc < prev_acc:
    		multiplier = -multiplier
    		step_size = step_size / 2
    	else:
    		step_size = step_size
    	new_reward += multiplier * step_size
    	prev_acc = cur_acc

