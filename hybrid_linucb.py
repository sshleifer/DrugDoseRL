import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from get_features import get_features
from utils import dose2str, get_sample_order, run_iters, calculate_reward
from linucb import UCB_FEATURES

arms = ["low", "medium", "high"]

class HybridLinUCB:
    def __init__(self, num_z_features, num_x_features, num_samples):
        self.order = get_sample_order(num_samples)
        self.alpha = 0.5
        self.A_lst = [np.identity(num_x_features)] * 3
        self.B_lst = [np.zeros((num_x_features, num_z_features))] * 3
        self.b_lst = [np.zeros(num_x_features)] * 3
        self.A_shared = np.identity(num_z_features)
        self.b_shared = np.zeros(num_z_features)


    def select_arm(self, z_features, x_features):
        """A method that returns the index of the Arm that the Bandit object
        selects on the current play.

        We will use the same features for both hybrid and individual features.

        Arm 0 = "low", Arm 1 = "medium", Arm 2 = "high"
        """
        beta = np.dot(np.linalg.inv(self.A_shared), self.b_shared)
        p_vals = []
        for i in range(3):
            theta = np.dot(np.linalg.inv(self.A_lst[i]), self.b_lst[i] - np.matmul(self.B_lst[i], beta))
            
            s = np.dot(z_features, np.dot(np.linalg.inv(self.A_shared), z_features))
            s_mat = np.matmul(np.linalg.inv(self.A_shared), np.matmul(self.B_lst[i].T, np.linalg.inv(self.A_lst[i])))
            s -= 2 * np.dot(z_features, np.dot(s_mat, x_features))
            s += np.dot(x_features, np.dot(np.linalg.inv(self.A_lst[i]), x_features))
            s_mat2 = np.matmul(np.linalg.inv(self.A_lst[i]), np.matmul(self.B_lst[i], s_mat))
            s += np.dot(x_features, np.dot(s_mat2, x_features))

            p = np.dot(z_features, beta) + np.dot(x_features, theta) + self.alpha * math.sqrt(s)
            p_vals.append(p)
        return np.argmax(p_vals), p_vals

    def update(self, chosen_arm, reward, z_features, x_features):
        """A method that updates the internal state of the Bandit object in 
        response to its most recently selected arm's reward.
        """
        self.A_shared = self.A_shared + np.matmul(self.B_lst[chosen_arm].T, 
                                np.matmul(np.linalg.inv(self.A_lst[chosen_arm]), self.B_lst[chosen_arm]))
        self.b_shared = self.b_shared + np.matmul(self.B_lst[chosen_arm].T, 
                                np.dot(np.linalg.inv(self.A_lst[chosen_arm]), self.b_lst[chosen_arm]))

        self.A_lst[chosen_arm] = self.A_lst[chosen_arm] + np.outer(x_features, x_features)
        self.B_lst[chosen_arm] = self.B_lst[chosen_arm] + np.outer(x_features, z_features)
        self.b_lst[chosen_arm] = self.b_lst[chosen_arm] + x_features * reward

        self.A_shared = self.A_shared + (np.outer(z_features, z_features)
            - np.matmul(self.B_lst[chosen_arm].T, np.matmul(np.linalg.inv(self.A_lst[chosen_arm]), self.B_lst[chosen_arm])))
        self.b_shared = self.b_shared + (z_features * reward
            - np.matmul(self.B_lst[chosen_arm].T, np.dot(np.linalg.inv(self.A_lst[chosen_arm]), self.b_lst[chosen_arm])))

def run_hybrid_linucb():
    logging = True
    # reward_style = "standard"
    # reward_style = "risk-sensitive"
    reward_style = "prob-based"
    if logging:
        log = open("log_hybrid_linucb.txt", "w+")
    arms = ["low", "medium", "high"]
    X, y = get_features()
    X['bias'] = 1
    z_feature_select = UCB_FEATURES
    x_feature_select = UCB_FEATURES
    X_z_subset = X[z_feature_select]
    X_x_subset = X[x_feature_select]
    z_num_features = len(z_feature_select)
    x_num_features = len(x_feature_select)
    hybrid_linucb = HybridLinUCB(z_num_features, x_num_features, X_z_subset.shape[0])
    total_regret = 0
    num_correct = 0
    for i in tqdm(range(X_z_subset.shape[0])):
        row_num = hybrid_linucb.order[i]
        z_features = np.array(X_z_subset.iloc[row_num])
        x_features = np.array(X_x_subset.iloc[row_num])
        arm, p_vals = hybrid_linucb.select_arm(z_features, x_features)
        regret, reward = calculate_reward(arm, y.iloc[row_num], reward_style, p_vals)
        num_correct += 1 if not regret else 0
        total_regret += regret
        hybrid_linucb.update(arm, reward, z_features, x_features)
        if logging:
            log.write("Sample %s: Using z features %s\n, x features %s\n" % (row_num, z_features, x_features))
            log.write("Chose arm %s with reward %s\n" % (arms[arm], reward))

    results = open("results_hybrid_linucb_%s.txt" % reward_style, "a+")
    acc = (num_correct / X.shape[0])

    print("Total regret: %s" % total_regret)
    print("Overall accuracy: %s" % acc)
    results.write("Regret: %s, Accuracy: %s\n" % (total_regret, num_correct / X.shape[0]))
    return acc, total_regret

if __name__ == "__main__":
    run_iters(10, run_hybrid_linucb)
    # run_hybrid_linucb()
