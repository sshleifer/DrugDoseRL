import numpy as np
import math
from get_features import get_features
from utils import dose2str, get_sample_order

class LinUCB:
    def __init__(self, num_features, num_samples):
        self.order = get_sample_order(num_samples)
        self.alpha = 0.5
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
            p += self.alpha * math.sqrt(np.dot(features, np.dot(np.linalg.inv(self.A_lst[i]), features)))
            p_vals.append(p)
        # print("p-vals:", p_vals)
        return np.argmax(p_vals)

    def update(self, chosen_arm, reward, features):
        """A method that updates the internal state of the Bandit object in 
        response to its most recently selected arm's reward.
        """
        self.A_lst[chosen_arm] = self.A_lst[chosen_arm] + np.outer(features, features)
        self.b_lst[chosen_arm] = self.b_lst[chosen_arm] + features * reward


def main():
    logging = True
    if logging:
        log = open("log.txt", "w+")
    arms = ["low", "medium", "high"]
    X, y = get_features()
    X['bias'] = 1
    # features to use
    feature_select = [
        'Age',
        'Weight (kg)',
        'Height (cm)',
        'bias'
    ]
    X_subset = X.loc[:, feature_select]
    num_features = len(feature_select)
    linucb = LinUCB(num_features, X_subset.shape[0])
    total_regret = 0
    for i in range(X_subset.shape[0]):
        if i % 50 == 0:
            print("Processed %s samples" % i)
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

    print("Total regret: %s" % total_regret)
    print("Overall accuracy: %s" % (1 - total_regret / X.shape[0]))

if __name__ == "__main__":
    main()

