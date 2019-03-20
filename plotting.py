import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns

def make_plot():
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}

    matplotlib.rc('font', **font)
    plt.figure(figsize=(10,10))
    df = pd.read_csv('all_last_runs.csv', index_col=0)
    pldata = df[df['Reward Structure'].isin(['hill-climbing', 'risk-sensitive', 'standard', 'fuzzy', 'prob-based'])]
    ax = sns.boxplot(x='model', y='Accuracy', hue='Reward Structure', data=pldata)
    plt.legend(
        #bbox_to_anchor=(1.04, 1),
        loc='bottom left')
    ax.set(ylim=0.45)
