import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import run_experiment
from run_experiment import SafeSpace
import gc # Garbage Collector
import sys
import os


if __name__ == '__main__':

    for iteration in range(4, 6):
        print(iteration)
        run_experiment.main(iteration)
        gc.collect()
    # res = pickle.load(open('results/street_params/activity_0.0/run_0.p', 'rb'))
    # print(res.housing_over_time)
    # fig, ax = plt.subplots()
    # cmap = sns.color_palette(["black", "forestgreen", "gold", "navy", "grey"])
    # activity_grid = res.activity_grid
    # sns.heatmap(activity_grid, cmap=cmap, ax=ax)
    # print(res.parameters)
    # plt.show()
