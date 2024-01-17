import numpy as np

def print_avgs(avgs, a=0):
    for key in avgs.keys():
        print("{0}: {1:.4}, ".format(key, np.mean(avgs[key][a:])), end="")
    print("")
