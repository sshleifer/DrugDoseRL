import numpy as np
import math
from tqdm import tqdm
from get_features import get_features
from utils import dose2str, get_sample_order

class HybridLinUCB:
    def __init__(self, num_features, num_samples):
        self.order = get_sample_order(num_samples)
        self.alpha = 0.5
        self.A_lst = [np.identity(num_features)] * 3
        self.B_lst = [np.zeros((num_features, num_features))] * 3
        self.b_lst = [np.zeros(num_features)] * 3
        self.A_shared = np.identity(num_features)
        self.b_shared = np.zeros(num_features)


    def select_arm(self, features):
        """A method that returns the index of the Arm that the Bandit object
        selects on the current play.

        We will use the same features for both hybrid and individual features.

        Arm 0 = "low", Arm 1 = "medium", Arm 2 = "high"
        """
        beta = np.dot(np.linalg.inv(self.A_shared), self.b_shared)
        p_vals = []
        for i in range(3):
            theta = np.dot(np.linalg.inv(self.A_lst[i]), self.b_lst[i] - np.matmul(self.B_lst[i], beta))
            
            s = np.dot(features, np.dot(np.linalg.inv(self.A_shared), features))
            s_mat = np.matmul(np.linalg.inv(self.A_shared), np.matmul(self.B_lst[i].T, np.linalg.inv(self.A_lst[i])))
            s -= 2 * np.dot(features, np.dot(s_mat, features))
            s += np.dot(features, np.dot(np.linalg.inv(self.A_lst[i]), features))
            s_mat2 = np.matmul(np.linalg.inv(self.A_lst[i]), np.matmul(self.B_lst[i], s_mat))
            s += np.dot(features, np.dot(s_mat2, features))

            p = np.dot(features, beta) + np.dot(features, theta) + self.alpha * math.sqrt(s)
            p_vals.append(p)
        return np.argmax(p_vals)

    def update(self, chosen_arm, reward, features):
        """A method that updates the internal state of the Bandit object in 
        response to its most recently selected arm's reward.
        """
        self.A_shared = self.A_shared + np.matmul(self.B_lst[chosen_arm].T, 
                                np.matmul(np.linalg.inv(self.A_lst[chosen_arm]), self.B_lst[chosen_arm]))
        self.b_shared = self.b_shared + np.matmul(self.B_lst[chosen_arm].T, 
                                np.dot(np.linalg.inv(self.A_lst[chosen_arm]), self.b_lst[chosen_arm]))

        self.A_lst[chosen_arm] = self.A_lst[chosen_arm] + np.outer(features, features)
        self.B_lst[chosen_arm] = self.B_lst[chosen_arm] + np.outer(features, features)
        self.b_lst[chosen_arm] = self.b_lst[chosen_arm] + features * reward

        self.A_shared = self.A_shared + (np.outer(features, features)
            - np.matmul(self.B_lst[chosen_arm].T, np.matmul(np.linalg.inv(self.A_lst[chosen_arm]), self.B_lst[chosen_arm])))
        self.b_shared = self.b_shared + (features * reward
            - np.matmul(self.B_lst[chosen_arm].T, np.dot(np.linalg.inv(self.A_lst[chosen_arm]), self.b_lst[chosen_arm])))

def main():
    logging = True
    if logging:
        log = open("log_hybrid_linucb.txt", "w+")
    arms = ["low", "medium", "high"]
    X, y = get_features()
    X['bias'] = 1
    feature_select = [
        'Age',
        'Weight (kg)',
        'Height (cm)',
        'VKORC1 genotype: 1173 C>T(6484); chr16:31012379; rs9934438; A/G_C/C',
        'CYP2C9 consensus_*1/*2',
        'VKORC1 -1639 consensus_G/G',
        'VKORC1 1542 consensus_G/G',
        'Valve Replacement',
        'VKORC1 genotype: 1542G>C (6853); chr16:31012010; rs8050894; C/G_G/G',
        'VKORC1 genotype: -1639 G>A (3673); chr16:31015190; rs9923231; C/T_G/G',
        'Race_White',
        'Current Smoker',
        'VKORC1 1173 consensus_C/C',
        'Ethnicity_not Hispanic or Latino',
        'Diabetes',
        'Congestive Heart Failure and/or Cardiomyopathy',
        'bias'
    ]
    X_subset = X[feature_select]
    num_features = len(feature_select)
    hybrid_linucb = HybridLinUCB(num_features, X_subset.shape[0])
    total_regret = 0
    for i in tqdm(range(X_subset.shape[0])):
        row_num = hybrid_linucb.order[i]
        features = np.array(X_subset.iloc[row_num])
        arm = hybrid_linucb.select_arm(features)
        reward = 0 if arms[arm] == dose2str(y.iloc[row_num]) else -1
        total_regret += 0 - reward
        hybrid_linucb.update(arm, reward, features)
        if logging:
            log.write("Sample %s: Using features %s\n" % (row_num, features))
            log.write("Chose arm %s with reward %s\n" % (arms[arm], reward))

    results = open("hybrid_linucb_results.txt", "a+")
    print("Total regret: %s" % total_regret)
    print("Overall accuracy: %s" % (1 - total_regret / X.shape[0]))
    results.write("Regret: %s, Accuracy: %s\n" % (total_regret, 1 - total_regret / X.shape[0]))

if __name__ == "__main__":
    main()

