'''
Obtains meta data about results_* files
'''

import numpy as np

file_names = ["results_linucb_risk-sensitive.txt", "results_linucb_standard.txt", "results_linucb_fuzzy.txt"]
for file_name in file_names:
	print("Calculating metadata for " + file_name)
	log = open(file_name, "r")

	accs = []
	fuzz_accs = []

	for line in log:
		l = line.split()
		accs.append(float(l[3].strip(',')))
		fuzz_accs.append(float(l[-1].strip(',')))

	print("Average of accuracy = " + str(np.mean(accs)))
	print("Variance accuracy = " + str(np.var(accs)))
	print()
	print("Average of fuzzy accuracy = " + str(np.mean(fuzz_accs)))
	print("Variance of fuzzy accuracy = " + str(np.var(fuzz_accs)))
	print()
	print()
	log.close()