import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns

def make_plots():
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 14}

    matplotlib.rc('font', **font)
    plt.figure(figsize=(10, 10))
    df = pd.read_csv('./results/all_last_runs.csv', index_col=0)
    pldata = df[df['Reward Structure'].isin(['hill-climbing', 'adjacent-arm', 'standard', 'fuzzy'])]
    ax = sns.boxplot(x='model', y='Fuzzy Accuracy', hue='Reward Structure', data=pldata)

    ax.set(ylim=0.8)
    # for fuzzy_accuracy
    plt.axhline(.8457, c = 'red', label = 'Clinical Baseline')
    plt.axhline(.8679, c = 'blue', label = 'Fixed Dose Baseline')

    plt.legend(
        #bbox_to_anchor=(1.04, 1),
        loc='bottom left')
    
    plt.savefig('./plots/fuzzy_acc-no-prob.png')
    plt.clf()



    pldata = df[df['Reward Structure'].isin(['hill-climbing', 'adjacent-arm', 'standard', 'fuzzy'])]
    ax = sns.boxplot(x='model', y='Accuracy', hue='Reward Structure', data=pldata)

    ax.set(ylim=0.6)

    #for accuracy
    plt.axhline(.6422, c = 'red', label = 'Clinical Baseline')
    plt.axhline(.6127, c = 'blue', label = 'Fixed Dose Baseline')
    
    plt.legend(
        #bbox_to_anchor=(1.04, 1),
        loc='upper right')
    
    plt.savefig('./plots/acc-no-prob.png')
    plt.clf()



    pldata = df[df['Reward Structure'].isin(['hill-climbing', 'adjacent-arm', 'standard', 'fuzzy', 'prob-based'])]
    ax = sns.boxplot(x='model', y='Fuzzy Accuracy', hue='Reward Structure', data=pldata)

    ax.set(ylim=0.7)
    # for fuzzy_accuracy
    plt.axhline(.8457, c = 'red', label = 'Clinical Baseline')
    plt.axhline(.8679, c = 'blue', label = 'Fixed Dose Baseline')
    plt.legend(
        #bbox_to_anchor=(1.04, 1),
        loc='bottom left')
    
    plt.savefig('./plots/fuzz_acc-prob.png')
    plt.clf()




    pldata = df[df['Reward Structure'].isin(['hill-climbing', 'adjacent-arm', 'standard', 'fuzzy', 'prob-based'])]
    ax = sns.boxplot(x='model', y='Accuracy', hue='Reward Structure', data=pldata)

    ax.set(ylim=0.45)

    plt.axhline(.6422, c = 'red', label = 'Clinical Baseline')
    plt.axhline(.6127, c = 'blue', label = 'Fixed Dose Baseline')

    
    plt.legend(
        #bbox_to_anchor=(1.04, 1),
        loc='bottom left')
    
    plt.savefig('./plots/acc-prob.png')


make_plots()
