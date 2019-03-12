import numpy as np
import math
from tqdm import tqdm
from get_features import get_features
from utils import dose2str, get_sample_order


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
        return np.argmax(p_vals)

    def update(self, chosen_arm, reward, features):
        """A method that updates the internal state of the Bandit object in 
        response to its most recently selected arm's reward.
        """
        self.A_lst[chosen_arm] = self.A_lst[chosen_arm] + np.outer(features, features)
        self.b_lst[chosen_arm] = self.b_lst[chosen_arm] + features * reward


def run_ucb():
    logging = True
    if logging:
        log = open("log.txt", "w+")
    arms = ["low", "medium", "high"]
    X, y = get_features()
    X['bias'] = 1
    feature_select = UCB_FEATURES

    # Use all features from paper
    X_subset = X[feature_select].astype(float)
    num_features = len(feature_select)
    linucb = LinUCB(num_features, X_subset.shape[0])
    total_regret = 0
    for i in tqdm(range(X_subset.shape[0])):
        row_num = linucb.order[i]
        features = np.array(X_subset.iloc[row_num])
        arm = linucb.select_arm(features)
        reward = 0 if arms[arm] == dose2str(y.iloc[row_num]) else -1
        total_regret += 0 - reward
        linucb.update(arm, reward, features)
        if logging:
            log.write("Sample %s: Using features %s\n" % (row_num, features))
            log.write("Chose arm %s with reward %s\n" % (arms[arm], reward))
            log.write("Updated parameters for arm:\n A = %s \n b = %s\n" % (linucb.A_lst[arm], linucb.b_lst[arm]))

    results = open("linucb_results.txt", "a+")

    acc = (1 - total_regret / X.shape[0])

    print("Total regret: %s" % total_regret)
    print("Overall accuracy: %s" % acc)
    results.write("Regret: %s, Accuracy: %s\n" % (total_regret, 1 - total_regret / X.shape[0]))
    return acc, total_regret


if __name__ == "__main__":
    run_ucb()

