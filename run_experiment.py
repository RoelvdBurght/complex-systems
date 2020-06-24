import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import gc
import city_class_test as cc
import time
import pickle
import os

import sys


class SafeSpace(object):
    def __init__(self, parameters, city):
        self.parameters = parameters
        self.activity_grid = np.array([obj.value for row in city.grid for obj in row]).reshape(100, 100)
        self.housing_over_time = city.activities[0]
        self.industry_over_time = city.activities[1]
        self.stores_over_time = city.activities[2]

def main(iteration):

    print('hoi')
    plot = False
    # for iteration in range(10):

    start = time.time()

    params_to_test = []
    locations = []
    for a in np.linspace(0, 7 / 8, 8):
        param = {'street_thresholds': {'activity': a}}
        params_to_test.append(param)
        locations.append('results/street_params/activity_{}'.format(h))

    for i in np.linspace(0,7/8,8):
        param = {'industry_thresholds': {'housing': i, ' industry':1, 'stores':0.4, 'streets':0.15}}
        params_to_test.append(param)
        locations.append('results/industry_thresholds/housing_activity_{}'.format(i))

    for i in np.linspace(0,7/8,8):
        param = {'industry_threshold' : {'housing':0.4, 'industry':1, 'stores':i, 'streets':0.15}}
        params_to_test.append(param)
        locations.append('results/industry_thresholds/stores_activity_{}'.format(i))

    for i in np.linspace(0,7/8,8):
        param = {'industry_thresholds': {'housing': 0.4, ' industry':1, 'stores':0.4, 'streets':i}}
        params_to_test.append(param)
        locations.append('results/industry_thresholds/housing_activity_{}'.format(i))

    r = cc.Runner(params_to_test, iterations=1000)
    r.run_experiment()

    # save results
    for city, param, loc in zip(r.cities, params_to_test, locations):
        savert = SafeSpace(param, city)
        if not os.path.isdir(loc):
            os.makedirs(loc)

        pickle.dump(savert, open('{}/run_{}.p'.format(loc, iteration), 'wb'))

    del savert
    r.clean_memory()
    del r
    gc.collect()

    end = time.time()
    print('Execution time: ', end - start)
    if plot:
        for i in range(len(r.cities)):
            fig = r.plot_grid(i)
            plt.show()


if __name__ == '__main__':
    main()
