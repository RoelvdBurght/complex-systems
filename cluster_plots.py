"""
File reads in the data from the experiments
Next generates plots with errorbars from it
including labels. Needed is the generated data folders
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from os import listdir
from os.path import isfile, join
import regex as re

def paths(main_path):
    all_files = []
    # for path in all_paths:
    all_files +=[main_path + f for f in listdir(main_path) if isfile(join(main_path, f))]
    return all_files

labels = ['housing', 'stores', 'industry']
color=['C2', 'C10', 'C8']
main_paths = ['data/housing_thres_industry_activities/', 'data/housing_thres_stores_activities/',
    'data/industry_thres_housing_activities/', 'data/industry_thres_stores_activities/',\
    'data/store_thres_housing_activities/', 'data/store_thres_industry_activities/',]
             

for i in range(len(main_paths)):
    files = paths(main_paths[i])

    fig = plt.figure(dpi=120)
    x = ["{}/8".format(i) for i in range(8)]
    for j in range(len(files)):
        if 'var' not in files[j]:
            size = np.loadtxt(files[j])
            var = np.loadtxt(files[j+1])
            plt.errorbar(x=x,y=size, yerr=var, label=labels[j%3],c=color[j%3])
    m = re.search(r'^[^_]+', main_paths[i][5:])
    first = re.search(r"\_(.*?)\/",main_paths[i][5:]).group(1)
    to_build = m.group(0)
    print(first[6:-11])
    plt.xlabel('{} threshold to build {}'.format(first[6:-11],to_build))
    plt.ylabel('Cluster size fraction')
    plt.legend()
    plt.show()
