import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from os import listdir
from os.path import isfile, join

def paths(main_path):
    all_files = []
    # for path in all_paths:
    all_files +=[main_path + f for f in listdir(main_path) if isfile(join(main_path, f))]
    return all_files

labels = ['housing', 'stores', 'industry']
color=['C2', 'C10', 'C8']
main_path = 'data/store_thres_housing_activities/'

files = paths(main_path)
print(files)

fig = plt.figure(dpi=120)
x = ["{}/8".format(i) for i in range(8)]
for i in range(len(files)):
    if 'var' not in files[i]:
        size = np.loadtxt(files[i])
        var = np.loadtxt(files[i+1])
        plt.errorbar(x=x,y=size, yerr=var, label=labels[i%3],c=color[i%3])

plt.xlabel('Housing threshold to build stores')
plt.ylabel('Cluster size fraction')
plt.legend()
plt.show()
