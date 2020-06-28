"""File containing all the function and classes to run the created model
    Contains city, activity class, key components of the model.
    Furthermore, a Runner class to run and save experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import multiprocessing as mp
import os
import gc
from time import time
from bresenham import bresenham
from itertools import combinations_with_replacement
from numba import int32, float32, types, typed, jit    # import the types
# from numba.experimental import jitclass


# creates a grid that of nxn with a single unspecified activity in the middle
class City(object):
    def __init__(self, n=100, n_radius=1, field_radius=2, street_field_radius=3, n_init=4, n_house_inits=2, init_decay=1/30, mature_decay=1/20, dist_decay=0.9, \
                                                                    street_thresholds={'activity':0.2}, \
                                                                    industry_threshold={'housing':0.2, 'industry':1, 'stores':0.4, 'streets':0.15},
                                                                    store_threshold={'housing':0.4, 'industry':0.2, 'stores':1, 'streets':0.05},
                                                                    housing_threshold={'housing':1, 'industry':0.15, 'stores':0.25, 'streets':0.05}):

        self.inMatrix = np.array([[0.9, 0.05, 0.05], [0.025, 0.95, 0.025], [0.025, 0.025, 0.95]])
        self.n = n
        self.t = 0
        self.n_radius = n_radius
        self.field_radius = field_radius
        self.street_field_radius = street_field_radius

        self.init_decay = init_decay
        self.mature_decay = mature_decay
        self.dist_decay = dist_decay

        self.street_thresholds = street_thresholds
        self.industry_threshold = industry_threshold
        self.housing_threshold = housing_threshold
        self.store_threshold = store_threshold
        self.all_activities = []

        self.types = [Housing, Industry, Stores]
        self.house_activity = 0
        self.industry_activity = 0
        self.shopping_activity = 0
        self.activities = [[],[],[],[]]
        self.street_nodes = []

        self.grid = self.initialise_grid()
        self.init_streets(n_init)
        self.init_activity(n_house_inits)
        # self.add_activity((n//2,n//2), Housing)
        # self.add_activity((0,0))
        # self.add_activity((n//2+1,n//2+1))    
        # self.history = [self.grid]


    def initialise_grid(self):
        """
        Create an empty grid with empty cells, and determine the neighbors and field neighbours of the cell
        :return: grid
        """
        grid = np.empty((self.n, self.n), dtype=object)
        for i in range(self.n):
            for j in range(self.n):
                neighbors = self.get_neighbors((i,j), self.n_radius)
                field_neighbors = self.get_neighbors((i,j), self.field_radius)
                cell = Empty((i,j), neighbors, field_neighbors, self.n)
                grid[(i,j)] = cell
        return grid

    def init_streets(self, n_init):
        """
        Make given number of initial street nodes at random locations, and make two streets connected to one node
        :param n_init: number of initial streets
        """
        i=0
        while i < n_init:
            init_pos = (np.random.randint(0,self.n), np.random.randint(0,self.n))
            try:
                empty_cell = self.grid[init_pos]
                self.grid[init_pos] = StreetNode(init_pos, empty_cell, self.get_neighbors((init_pos[0],init_pos[1]),
                                                                                        self.street_field_radius), self)
                self.street_nodes += [self.grid[init_pos]]
                del empty_cell
                second_pos = (init_pos[0]+np.random.randint(-5,6), init_pos[1])
                third_pos = (init_pos[0], init_pos[1]+np.random.randint(-5,6))

                self.lay_street(init_pos, second_pos)
                self.lay_street(init_pos, third_pos)
            except:
                continue
            i+=1

    def lay_street(self, init_pos, second_pos):
        """
        Take the two positions and lay a street in between them. Update list of street nodes with node
        :param init_pos: first position
        :param second_pos: second position
        """
        empty_cell = self.grid[second_pos]
        self.grid[second_pos] = StreetNode(second_pos, empty_cell, self.get_neighbors((second_pos[0],second_pos[1]), self.street_field_radius), self)
        self.street_nodes += [self.grid[second_pos]]
        inter_cells = list(bresenham(init_pos[0], init_pos[1], second_pos[0], second_pos[1]))
        for cell in inter_cells[1:-1]:
            self.grid[cell] = StreetRoute(cell, self.grid[cell], self)
        del empty_cell

    def init_activity(self, n_house_inits):
        """
        For every street node add a house if the random chosen cell is empty
        :param n_house_inits: number of initial houses to be build
        """
        for streetNode in self.street_nodes:
            i = 0
            while i < n_house_inits:
                pos = (streetNode.pos[0]+np.random.randint(-2,3),streetNode.pos[1]+np.random.randint(-2,3))
                try:
                    if self.grid[pos].value != 0:
                        continue
                except:
                    continue
                clas = self.types[0]
                self.add_activity(tuple(pos), clas)
                i += 1

    def add_activity(self, pos, clas):
        """
        Add activity to the new given cell, add it to the all_activities list
        :param pos: given position of the to be determined cell
        :param clas: given class of the cell which spawned the new cell
        """
        empty_cell = self.grid[pos]
        self.grid[pos] = clas(pos, empty_cell, self, self.init_decay, self.mature_decay)
        del empty_cell
        self.all_activities += [self.grid[pos]]

    def delete_activity(self, act):
        """
        Remove the given object from the grid and from the all_activities list
        :param act: given object to be deleted
        """
        self.grid[act.pos] = Empty(act.pos, act.neighborhood, act.field_neighbors, self.n)
        self.all_activities.remove(act)
        del act

    def get_all_activities(self):
        """
        Loop through grid, check if an activity is located on a specific location, if so add to list
        :return: list with all activities of the grid
        """
        activities = []
        for pos in combinations_with_replacement(range(self.n), 2):
            if self.grid[pos]:
                activities.append(self.grid[pos])
        return activities

    # determines the type of an activity by recalculating the type probabilities
    def update_types(self):
        """
    	Updates the types of a given activity
        :return: does not return anythhing
        """
        for act in self.all_activities:
            act.calc_probs()
            probsum = cumsum(np.array([act.pi, act.pm, act.pd]))
            z = np.random.rand()
            if z < probsum[0]:
                act.type = 'init'
            elif z < probsum[1]:
                act.type = 'mature'
            else:
                act.type = 'decline'

    def delete_declining(self):
        """
        Loop through all activities, and if the state == decline delete the activity
        """
        [self.delete_activity(act) for act in self.all_activities if act.type == 'decline']

    def get_grow_candidates(self, act):
        """
        Get object which can spawn a new activity, check if there are empty neighbours, and determine with a random
        chance if the candidate can grow
        :param act: object which can spawn a new activity
        :return: list with candidates which might spawn
        """
        grow_candidates = [[candidate, act] for candidate in act.field_neighbors if isinstance(self.grid[candidate], Empty)]
        grow_probs = [calc_grow_prob(np.array(candidate_pos[0]), np.array(act.pos), self.dist_decay) for candidate_pos in grow_candidates]
        return [grow_candidates[i] for i in range(len(grow_candidates)) if np.random.rand() < grow_probs[i]]


    def determine_activity(self, grow_candidates):
        """
        Loop through list with candidates, check what the initial node for activity had, check the neighbours of the
        candidate and determine what the new activity is going to be
        :param grow_candidates: list with candidates
        :return: list with the candidates and their new activity
        """
        real_list = []
        for candidate in grow_candidates:
            z = np.random.rand()
            new_node_neighborhood = self.grid[candidate[0]].neighborhood
            # check what the activity is, and what the new activity will be
            if isinstance(candidate[1], Housing):
                probsum = cumsum(self.inMatrix[0])
                if z < probsum[0]:
                    if self.check_density_activity(new_node_neighborhood, self.housing_threshold):
                        candidate[1] = Housing
                        real_list += [candidate]

                elif z < probsum[1]:
                    if self.check_density_activity(new_node_neighborhood, self.industry_threshold):
                        candidate[1] = Industry
                        real_list += [candidate]

                else:
                    if self.check_density_activity(new_node_neighborhood, self.store_threshold):
                        candidate[1] = Stores
                        real_list += [candidate]

            elif isinstance(candidate[1], Industry):
                if self.check_density_activity(new_node_neighborhood, self.industry_threshold):
                    candidate[1] = Industry
                    real_list += [candidate]
            else:
                if self.check_density_activity(new_node_neighborhood, self.store_threshold):
                    candidate[1] = Stores
                    real_list += [candidate]
        return real_list

    def street_density_check(self, new_node_neighborhood, thresholds):
        """
        Check if the constraints for the building of a street are met
        :param new_node_neighborhood: neighbourhood of the potential new street
        :param thresholds: Parameters taken into account
        :return: True or False
        """
        housing, industry, stores, streets = self.neighbourhood_density(new_node_neighborhood)
        if sum([housing, industry, stores]) >= thresholds['activity'] and streets == 0:
            return True
        return False

    def neighbourhood_density(self, new_node_neighborhood):
        """
        Loop through neighbours and count per activity how many there are, then divide it through amount of neighbours
        :param new_node_neighborhood: neighbourhood of the potential new activity
        :return: ratio of housing, industry, stores and streets
        """
        length_neighbours = len(new_node_neighborhood)
        housing = sum(self.grid[neighbor].value == 1 for neighbor in new_node_neighborhood)/length_neighbours
        industry= sum(self.grid[neighbor].value == 2 for neighbor in new_node_neighborhood)/length_neighbours
        stores = sum(self.grid[neighbor].value == 3 for neighbor in new_node_neighborhood)/length_neighbours
        streets = sum(self.grid[neighbor].value == 4 or self.grid[neighbor].value == 5 for neighbor in new_node_neighborhood)/length_neighbours
        return housing, industry, stores, streets

    def check_density_activity(self, new_node_neighborhood, thresholds):
        """
        Check if the constraints for adding a new activity are met
        :param new_node_neighborhood: neighbourhood of the potential new activity
        :param thresholds: Parameters taken into account
        :return: True or False
        """
        housing, industry, stores, streets = self.neighbourhood_density(new_node_neighborhood)
        if (housing <= thresholds['housing'] and industry <= thresholds['industry'] and stores <= thresholds['stores'] \
                and streets >= thresholds['streets']):
            return True
        return False

    def street_check(self, new_street_nodes):
        """
        Loop through list to see which new street nodes can be build
        :param new_street_nodes: list with all new street nodes
        :return: added nodes
        """
        new_nodes = []
        for node in new_street_nodes:
            new_node_neighborhood = self.grid[node[0]].neighborhood
            if self.street_density_check(new_node_neighborhood, self.street_thresholds):
                new_nodes += [node]
        return new_nodes

    def vancant_cells(self, init_pos, second_pos):
        """
        See if the cells between two given points are empty
        :param init_pos: first position
        :param second_pos: second position
        :return: True or False
        """
        inter_cells = list(bresenham(init_pos[0], init_pos[1], second_pos[0], second_pos[1]))
        for cell in inter_cells[1:-1]:
            if self.grid[cell].value != 0:
                return False
        return True

    def initiate_streets(self):
        """
        Do all checks and make a street
        """
        new_nodes = []
        for node in self.street_nodes:
            new_nodes += [self.street_check(self.get_grow_candidates(node))]
        for i in range(len(self.street_nodes)):
            init_node = self.street_nodes[i]
            for second_node in new_nodes[i]:
                if self.vancant_cells(init_node.pos, second_node[0]):
                    self.lay_street(init_node.pos, second_node[0])
                    break

    def initiate_activity(self):
        """
        Go through all activities that are in the init state, then check if they can spawn a new cell
        """
        init_sites = [act for act in self.all_activities if act.type == 'init']
        new_positions = []
        new_type = []
        for act in init_sites:
            new_variable = self.determine_activity(self.get_grow_candidates(act))
            for var in new_variable:
                if var[0] not in new_positions:
                    new_positions.append(var[0])
                    new_type.append(var[1])
        [self.add_activity(new_positions[i], new_type[i]) for i in range(len(new_positions))]

    def get_neighbors(self, pos, radius):
        """
        Take all neighbours in the given range from the given position on the grid
        :param pos: given position to check
        :param radius: given radius to use
        :return: list of neighbours
        """
        neighbors = []
        row, col = pos
        for i in range(col-radius, col+radius+1):
            for j in range(row-radius, row+radius+1):
                if not (0 <= i < self.n and 0 <= j < self.n):
                    continue
                if pos == (j,i):
                    continue
                neighbors += [(j,i)]
        return neighbors

    def step(self):
        """
        Take time step, make new streets and activities, update all the lists
        """
        self.t += 1
        self.update_types()
        self.initiate_streets()
        self.initiate_activity()
        self.delete_declining()
        self.activities[0] += [sum([1 for act in self.all_activities if isinstance(act, Housing)])]
        self.activities[1] += [sum([1 for act in self.all_activities if isinstance(act, Industry)])]
        self.activities[2] += [sum([1 for act in self.all_activities if isinstance(act, Stores)])]

    def plot_growth(self):
        """
        Plot graph with activities
        """
        _, axes = plt.subplots(4, figsize=(16,16))
        labels = ['Housing', 'Industry', 'Stores', 'Streets']
        for i, act in enumerate(self.activities):
            sns.lineplot(range(len(act)), act, ax=axes[i])
            axes[i].set_xlabel(labels[i])
            axes[i].set_ylabel('Time')
        plt.show()

class Empty(object):
    """
    Empty/vacant cell object, holds the 2 neighborhoods of a cell as variable.
    """
    def __init__(self, pos, neighborhood, field_neighbors, n):
        self.value = 0
        self.pos = pos
        self.neighborhood = neighborhood
        self.field_neighbors = field_neighbors
        self.n = n

class Activity(object):
    """
    Activity object, general class that holds all general variables needed for
    for each type of activity.
    """
    def __init__(self, pos, empty_cell, city, init_decay, mature_decay, value):
        self.pos = pos
        self.city = city
        self.init_t = city.t
        self.init_decay = init_decay
        self.mature_decay = mature_decay
        self.value = value
        self.type = 'new'
        self.neighborhood = empty_cell.neighborhood
        self.field_neighbors = empty_cell.field_neighbors
        self.calc_probs()

    def calc_probs(self):
        self.pi, self.pm, self.pd = fast_calc_probs(self.init_decay, self.mature_decay, self.city.t, self.init_t)

class Housing(Activity):
    """
    Inheritance class of Activity
    """
    def __init__(self, pos, empty_cell, city, init_decay=1/30, mature_decay=1/30):
        super().__init__(pos, empty_cell, city, init_decay, mature_decay, value=1)

class Industry(Activity):
    """
    Inheritance class of Activity
    """
    def __init__(self, pos, empty_cell, city,init_decay=1/15, mature_decay=1/30):
        super().__init__(pos, empty_cell, city,  init_decay, mature_decay, value=2)

class Stores(Activity):
    """
    Inheritance class of Activity
    """
    def __init__(self, pos, empty_cell, city, init_decay=1/10, mature_decay=1/30):
        super().__init__(pos, empty_cell, city, init_decay, mature_decay, value=3)


class StreetNode(object):
    """
    Street node object, which can spawn new street nodes
    Holds street neighborhoods
    """
    def __init__(self, pos, empty_cell, street_field, city):
        self.pos = pos
        self.neighborhood = empty_cell.neighborhood
        self.field_neighbors = street_field
        self.value = 4

class StreetRoute(object):
    """
    Street object, cannot spawn anything, is an object that occupies
    cell as street on the grid.
    """
    def __init__(self, pos, empty_cell, city):
        self.pos = pos
        self.city = city
        self.neighborhood = empty_cell.neighborhood
        self.field_neighbors = empty_cell.field_neighbors
        self.value = 5

class Runner(object):
    """
    Roel denk ik
    """
    def __init__(self, args, iterations=50):
        self.args = args
        self.iterations = iterations
        self.cities = []

    def iterate_city(self, arguments, seed):
        # Runs the model for specified number of iterations with given variable
        np.random.seed(seed)
        print('s', seed)
        print('a', arguments)
        city = City(**arguments)
        for _ in range(self.iterations):
            city.step()
        return city

    def run_experiment(self):
        seeds = [np.random.randint(0, 1000000) for _ in range(len(self.args))]
        argument_list = [(arg, seed) for arg, seed in zip(self.args, seeds)]
        pool = mp.Pool(os.cpu_count())
        result_cities = pool.starmap(self.iterate_city, argument_list)
        pool.close()
        print(result_cities)
        self.cities = result_cities

    def clean_memory(self):
        del self
        gc.collect()

    def plot_grid(self, city_num, city_n=100):
        fig, ax = plt.subplots()
        cmap = sns.color_palette(["black", "forestgreen", "gold", "navy", "grey"])
        activity_grid = np.array([obj.value for row in self.cities[city_num].grid for obj in row]).reshape(city_n,city_n)
        sns.heatmap(activity_grid, cmap=cmap, ax=ax)
        return fig


# calculates the probability for a activity to be of some type
@jit(nopython=True)
def fast_calc_probs(init_decay, mature_decay, t, init_t):
    pi = np.exp(-init_decay*(t-init_t))
    pm = (1-pi)*np.exp(-mature_decay*(t-init_t))
    return pi,pm, 1-pi-pm


# get the probability an activity is starting at candidate position
# function is outside the class to use faster jit function
@jit(nopython=True)
def calc_grow_prob(candidate_pos, act_pos, dist_decay):
    return np.exp(-dist_decay*euclidean_dist(act_pos, candidate_pos))

# jit function to compute the euclidean distance of two points
@jit(nopython=True)
def euclidean_dist(pos1, pos2):
    return np.sqrt(np.sum((pos1-pos2)**2))

# Jit function to fastly compute the cumalative sum of a array
@jit(nopython=True)
def cumsum(l):
    return np.cumsum(l)

if __name__ ==  "__main__":
    pass
