from subprocess import Popen
import run_experiment
import gc # Garbage Collector
import sys
import os


if __name__ == '__main__':

    for iteration in range(10):
        run_experiment.main(iteration)
        gc.collect()