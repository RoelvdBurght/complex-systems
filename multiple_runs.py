import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import run_experiment
from run_experiment import SafeSpace
import gc # Garbage Collector
import sys
import os


if __name__ == '__main__':

    for iteration in range(10):
        run_experiment.main(iteration)
        gc.collect()
