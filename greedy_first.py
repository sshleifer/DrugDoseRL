import numpy as np
import math
from tqdm import tqdm
from get_features import get_features
from sklearn import linear_model as lm
from utils import dose2str, get_sample_order, run_iters, calculate_reward, is_correct, is_fuzz_correct
from linucb import UCB_FEATURES

arms = ["low", "medium", "high"]

class GreedyFirst:
    def __init__(self, num_features, num_samples):
        self.order = get_sample_order(num_samples)
        self.q = 2
        self.h = 5
        self.num_features = num_features
        self.t_0 = 8 * 3 * num_features
        self.lambda_0 = 0

        self.iter = 0
        self.n = 0
        self.j_low = [-self.q + k for k in range(1, self.q + 1)]
        self.j_med = [k for k in range(1, self.q + 1)]
        self.j_high = [self.q + k for k in range(1, self.q + 1)]

        self.T_sets = [[], [], []]
        self.S_sets = [[], [], []]

        self.T_beta = [np.zeros(num_features)] * 3
        self.S_beta = [np.zeros(num_features)] * 3

        self.X = np.array([]).reshape(0, num_features)
        self.y = np.array([]).reshape(0, 1)

        self.ols = False

    def select_arm(self, features):
        """A method that returns the index of the Arm that the Bandit object
        selects on the current play.

        Arm 0 = "low", Arm 1 = "medium", Arm 2 = "high"
        """
        # participating in force-sampling
        self.iter += 1
        arm = -1
        force_base = (2**self.n - 1) * 3 * self.q
        diff = self.iter - force_base
        if diff in self.j_low:
            arm, p_vals = 0, [1, 0, 0]
        elif diff in self.j_med:
            arm, p_vals = 1, [0, 1, 0]
        elif diff in self.j_high:
            if diff == self.j_high[-1]:
                self.n += 1
            arm, p_vals = 2, [0, 0, 1]

        if self.ols and arm != -1:
            self.T_sets[arm].append(self.iter)
            return arm, p_vals

        if self.ols:
            K_hat = []
            T_beta_vals = [np.dot(features, self.T_beta[i]) for i in range(3)]
            max_T_beta = max(T_beta_vals)

            for i in range(3):
                if T_beta_vals[i] >= max_T_beta - self.h / 2:
                    K_hat.append(i)

            p_vals = []
            for i in K_hat:
                p_vals.append((i, np.dot(features, self.S_beta[i])))

            chosen_arm = max(p_vals, key=lambda p: p[1])[0]
            all_p_vals = p_vals + [(i, 0) for i in range(3) if i not in K_hat]
            return chosen_arm, [p[1] for p in sorted(all_p_vals)]
        else:
            p_vals = [np.dot(features, self.S_beta[i]) for i in range(3)]
            return np.argmax(p_vals), p_vals


    def update(self, chosen_arm, reward, features):
        """A method that updates the internal state of the Bandit object in 
        response to its most recently selected arm's reward.
        """
        self.X = np.append(self.X, features.reshape(1, self.num_features), axis=0)
        self.y = np.append(self.y, reward)

        self.S_sets[chosen_arm].append(self.iter)

        S_idx = np.array(self.S_sets[chosen_arm])
        if np.size(S_idx) > 0:
            s_linreg = lm.LinearRegression(fit_intercept=False)
            s_linreg.fit(self.X[S_idx - 1, :], self.y[S_idx - 1])
            self.S_beta[chosen_arm] = s_linreg.coef_

        if self.ols:
            for i in range(3):  # forced sampling
                T_idx = np.array(self.T_sets[i])
                T_idx = T_idx[T_idx <= self.iter]
                if np.size(T_idx) > 0:
                    t_ols = lm.LinearRegression(fit_intercept=False)
                    t_ols.fit(self.X[T_idx - 1, :], self.y[T_idx - 1])
                    self.T_beta[i] = t_ols.coef_
        else:
            cov_lst = []
            for i in range(3):
                X_s = self.X[S_idx - 1, :]
                cov = np.matmul(X_s.T, X_s)
                cov_lst.append(cov)

            cov_eigen = [np.min(np.linalg.eigvals(m)) for m in cov_lst]

            if self.t_0 == self.iter:
                self.lambda_0 = min(cov_eigen) / (2 * self.t_0)

            if self.iter > self.t_0 and min(cov_eigen) < self.lambda_0 * self.iter / 4:
                self.ols = True
                self.S_sets = [[], [], []]



def run_greedy_first():
    logging = True
    # reward_style = "standard"
    # reward_style = "risk-sensitive"
    reward_style = "prob-based"
    # reward_style = "proportional"
    # reward_style = "fuzzy"
    eps = 7
    if logging:
        log = open("log_greedy_first.txt", "w+")

    X, y = get_features()
    X['bias'] = 1
    feature_select = UCB_FEATURES
    X_subset = X[feature_select]
    num_features = len(feature_select)
    greedy_first = GreedyFirst(num_features, X_subset.shape[0])
    
    total_regret = 0
    num_correct = 0
    num_fuzz_correct = 0
    hist = []

    for i in tqdm(range(X_subset.shape[0])):
        row_num = greedy_first.order[i]
        features = np.array(X_subset.iloc[row_num])
        arm, p_vals = greedy_first.select_arm(features)
        dose = y.iloc[row_num]
        regret, reward = calculate_reward(arm, dose, reward_style, p_vals)

        num_correct += 1 if is_correct(arm, dose) else 0
        num_fuzz_correct += 1 if is_fuzz_correct(arm, dose, eps) else 0
        total_regret += regret

        greedy_first.update(arm, reward, features)
        hist.append(arm)
        if logging:
            log.write("Sample %s: Using features %s\n" % (row_num, features))
            log.write("Chose arm %s with reward %s\n" % (arms[arm], reward))
            log.write("Correct dose was %s (%s)\n" % (dose2str(dose), dose))

    results = open("results_greedy_first_%s.txt" % reward_style, "a+")
    acc = (num_correct / X.shape[0])
    fuzz_acc = num_fuzz_correct / X.shape[0]

    print("Total regret: %s" % total_regret)
    print("Overall accuracy: %s" % acc)
    print("Overall fuzzy accuracy: %s" % fuzz_acc)
    results.write("Regret: %s, Accuracy: %s, Fuzzy Accuracy: %s\n" % (total_regret, acc, fuzz_acc))

    return acc, total_regret

if __name__ == "__main__":
    run_iters(10, run_greedy_first)
    # run_greedy_first()

