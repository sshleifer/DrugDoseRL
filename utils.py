import random

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
