import numpy as np
import math
from tqdm import tqdm
from get_features import get_features
from utils import dose2str, get_sample_order, run_iters, calculate_reward, is_correct, is_fuzz_correct

arms = ["low", "medium", "high"]

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
        self.A_lst = [np.identity(num_features)] * 3
        self.b_lst = [np.zeros(num_features)] * 3

    def select_arm(self, features):
        """A method that returns the index of the Arm that the Bandit object
        selects on the current play.
        """
        p_vals = []
        for i in range(3):
            theta = np.dot(np.linalg.inv(self.A_lst[i]), self.b_lst[i])
            p = np.dot(theta, features)
            p += self.alpha * math.sqrt(np.dot(features, np.dot(np.linalg.inv(self.A_lst[i]), features)))
            p_vals.append(p)
        return np.argmax(p_vals), p_vals

    def update(self, chosen_arm, reward, features):
        """A method that updates the internal state of the Bandit object in 
        response to its most recently selected arm's reward.
        """
        self.A_lst[chosen_arm] = self.A_lst[chosen_arm] + np.outer(features, features)
        self.b_lst[chosen_arm] = self.b_lst[chosen_arm] + features * reward


def run_linucb(reward_style, eps, logging):
    if logging:
        log = open("log_linucb.txt", "w+")
    show_history = False
    
    X, y = get_features()
    X['bias'] = 1
    feature_select = UCB_FEATURES
    X_subset = X[feature_select].astype(float)
    num_features = len(feature_select)
    linucb = LinUCB(num_features, X_subset.shape[0])

    total_regret = 0
    num_correct = 0
    num_fuzz_correct = 0

    for i in tqdm(range(X_subset.shape[0])):
        if i > 0 and i % 100 == 0:
            step_results = open("results/linucb/%s/results_linucb_%s_step_%s.txt" % (reward_style, reward_style, i), "a+")
            step_results.write("Regret: %s, Accuracy: %s, Fuzzy Accuracy: %s\n" % (total_regret, num_correct / i, num_fuzz_correct / i))
            step_results.close()
        row_num = linucb.order[i]
        features = np.array(X_subset.iloc[row_num])
        arm, p_vals = linucb.select_arm(features)
        dose = y.iloc[row_num]
        regret, reward = calculate_reward(arm, dose, reward_style, p_vals)
        
        num_correct += 1 if is_correct(arm, dose) else 0
        num_fuzz_correct += 1 if is_fuzz_correct(arm, dose, eps=7) else 0
        total_regret += regret

        linucb.update(arm, reward, features)
        if logging:
            log.write("Sample %s: Using features %s\n" % (row_num, features))
            log.write("Chose arm %s with reward %s\n" % (arms[arm], reward))
            log.write("Correct dose was %s (%s)\n" % (dose2str(dose), dose))

    results = open("results/linucb/results_linucb_%s.txt" % reward_style, "a+")

    acc = num_correct / X.shape[0]
    fuzz_acc = num_fuzz_correct / X.shape[0]

    print("logging for: " + reward_style)
    print("Total regret: %s" % total_regret)
    print("Overall accuracy: %s" % acc)
    print("Overall fuzzy accuracy: %s" % fuzz_acc)
    results.write("Regret: %s, Accuracy: %s, Fuzzy Accuracy: %s\n" % (total_regret, acc, fuzz_acc))
    results.close()

    return acc, total_regret


if __name__ == "__main__":
    logging = True
    reward_styles = ["standard", "risk-sensitive", "prob-based", "proportional", "fuzzy", "hill-climbing"]
    run_all = False
    eps = 7
    num_iters = 50
    if run_all:
        for r in reward_styles:
            print("Running reward style %s" % r)
            run_iters(num_iters, run_linucb, r, eps, logging)
    else:
        run_iters(num_iters, run_linucb, reward_styles[5], eps, logging)

