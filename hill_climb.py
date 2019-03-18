import numpy as np
import math
from tqdm import tqdm
from get_features import get_features
from utils import dose2str, get_sample_order, run_iters, is_correct, is_fuzz_correct, show_hist

arms = ["low", "medium", "high"]

ALL_CYP_FEATURES = ['Combined QC CYP2C9_*1/*1', 'Combined QC CYP2C9_*1/*2',
       'Combined QC CYP2C9_*1/*3', 'Combined QC CYP2C9_*2/*2',
       'Combined QC CYP2C9_*2/*3', 'Combined QC CYP2C9_*3/*3',
       'CYP2C9 consensus_*1/*1', 'CYP2C9 consensus_*1/*11',
       'CYP2C9 consensus_*1/*13', 'CYP2C9 consensus_*1/*14',
       'CYP2C9 consensus_*1/*2', 'CYP2C9 consensus_*1/*3',
       'CYP2C9 consensus_*1/*5', 'CYP2C9 consensus_*1/*6',
       'CYP2C9 consensus_*2/*2', 'CYP2C9 consensus_*2/*3',
       'CYP2C9 consensus_*3/*3']

UCB_FEATURES = [
    'Age', 'Weight (kg)', 'Height (cm)',
    'taking_amiodarone',
    'enzyme_score_2',
    'Race_White',
    'Race_Black or African American',
    'Race_Asian',
    'Race_Unknown',
    'Valve Replacement',
    'Current Smoker',
    'Ethnicity_not Hispanic or Latino',
    'Diabetes', 'Congestive Heart Failure and/or Cardiomyopathy', 'bias',

    # proxies for unknown CYP/VKOR
    'cyp_sum', 'vko_sum',

    'CYP2C9 consensus_*1/*2',
    'CYP2C9 consensus_*1/*3',
    'CYP2C9 consensus_*2/*2',
    'CYP2C9 consensus_*2/*3',
    'CYP2C9 consensus_*3/*3',


    'VKORC1 -1639 consensus_G/G',
    'VKORC1 1542 consensus_G/G',
    'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G_C/C',
    'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G_G/G',
    'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T_G/G',
    'VKORC1 1173 consensus_C/C',

]
assert len(UCB_FEATURES) == len(set(UCB_FEATURES)), 'Please remove duplicate feature.'


class LinUCB:
    def __init__(self, num_features, num_samples):
        self.order = get_sample_order(num_samples)
        self.alpha = 0.5
        # formula for alpha in paper

        self.A_lst = [np.identity(num_features)] * 3
        self.b_lst = [np.zeros(num_features)] * 3

    def select_arm(self, features):
        """A method that returns the index of the Arm that the Bandit object
        selects on the current play.

        Arm 0 = "low", Arm 1 = "medium", Arm 2 = "high"
        """
        p_vals = []
        for i in range(3):
            theta = np.dot(np.linalg.inv(self.A_lst[i]), self.b_lst[i])
            p = np.dot(theta, features)
            #import ipdb; ipdb.set_trace()
            p += self.alpha * math.sqrt(np.dot(features, np.dot(np.linalg.inv(self.A_lst[i]), features)))
            p_vals.append(p)
        return np.argmax(p_vals), p_vals

    def update(self, chosen_arm, reward, features):
        """A method that updates the internal state of the Bandit object in 
        response to its most recently selected arm's reward.
        """
        self.A_lst[chosen_arm] = self.A_lst[chosen_arm] + np.outer(features, features)
        self.b_lst[chosen_arm] = self.b_lst[chosen_arm] + features * reward

def calculate_reward(arm_ind, y, new_reward):
    if is_correct(arm_ind, y):
        reward = 0
    elif is_fuzz_correct(arm_ind, y):
        reward = - 0.275
    else:
        reward = new_reward
    regret = -reward
    return regret, reward


def run_linucb(new_reward):
    logging = False
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
        num_fuzz_correct += 1 if is_fuzz_correct(arm, dose, eps=3.5) else 0
        total_regret += regret

        linucb.update(arm, reward, features)
        hist.append(arm) #useful for plotting results
    if show_history:
        show_hist(hist)
    acc = num_correct / X.shape[0]
    fuzz_acc = num_fuzz_correct / X.shape[0]

    return fuzz_acc


'''
linucb with stand reward struct gives fuzzacc = 0.7880451812716344
linucb with fuzzy reward struct gives fuzzacc = 0.7930205866278012

Things I've tried: (tuning is_fuzz_acc reward)
Started with -0.5, descending, gave value of -0.725 with fuzzacc =  0.791956640553835
Started with -0.725, descending, gave value of -0.78125 with fuzzacc =  0.7903261067589726

Started with -0.5, ascending, gave value of -0.38125 with fuzzacc = 0.7935143013299325
Started with -0.3, descending, gave value of -0.196875 with fuzzacc = 0.7943159045363455

Started with -0.2, descending, gave value of -0.265625 with fuzzacc =  0.794279468026963


(tuning is incorrect reward (is fuzz rew = -.275)):
Started with -0.275, descending, gave value of -0.65 with fuzzacc = 0.7922663508835852
Started with -1, descending, gave value of 
Things to do:
'''
if __name__ == "__main__":
    new_reward = -1.1
    step_size = 0.1
    num_iters = 20
    prev_acc = 0
    multiplier = -1 #marks if we're adding or subtracting from new_reward

    while step_size > 0.005:
    	print("Trying with intermediate reward of: " + str(new_reward))
    	cur_accs = []
    	for j in range(50):
    		cur_accs.append(run_linucb(new_reward))
    	cur_acc = np.mean(cur_accs)
    	print("Fuzzy accuracy was: " + str(cur_acc))
    	if cur_acc < prev_acc:
    		multiplier = -multiplier
    		step_size = step_size/1.5
    	else:
    		step_size = step_size * 1.5
    	new_reward += multiplier * step_size
    	prev_acc = cur_acc

