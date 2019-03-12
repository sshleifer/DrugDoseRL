import random
from tqdm import tqdm
import pandas as pd

def dose2str(dose):
    if dose < 21:
        return "low"
    elif dose > 49:
        return "high"
    else:
        return "medium"

def  dose2int(dose):
    if dose < 21:
        return 0
    elif dose > 49:
        return 2
    else:
        return 1


def get_sample_order(n):
    lst = list(range(n))
    random.shuffle(lst)
    return lst


def run_iters(n, func):
    res = []
    for _ in tqdm(range(n)):
        res.append(func())
    res_df = pd.DataFrame(res)
    return res_df
