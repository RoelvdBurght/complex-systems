import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from time import time
# import copy
from bresenham import bresenham
from itertools import combinations_with_replacement

from numba import int32, float32, types, typed, jit    # import the types
# from numba.experimental import jitclass


# creates a grid that of nxn with a single unspecified activity in the middle
class City(object):
    def __init__(self, n=10, n_radius=1, field_radius=2, street_field_radius=3, n_init=4, n_house_inits=2, init_decay=1/30, mature_decay=1/20, dist_decay=0.9, \
                                                                    street_thresholds={'housing':0.1, 'industry':0.6, 'stores':0.7}, \
                                                                    industry_threshold={'housing':0.2, 'industry':1, 'stores':0.4, 'streets':0.15},
                                                                    store_threshold={'housing':0.2, 'industry':0.2, 'stores':1, 'streets':0},
                                                                    housing_threshold={'housing':0.8, 'industry':0.15, 'stores':0.25, 'streets':0.05}):

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

        # self.activity_threshold = activity_threshold
        self.industry_threshold = industry_threshold
        self.housing_threshold = housing_threshold
        self.store_threshold = store_threshold
        self.all_activities = []

        # self.mature_decay
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
        grid = np.empty((self.n, self.n), dtype=object)
        for i in range(self.n):
            for j in range(self.n):
                neighbors = self.get_neighbors((i,j), self.n_radius)
                field_neighbors = self.get_neighbors((i,j), self.field_radius)
                cell = Empty((i,j), neighbors, field_neighbors, self.n)
                grid[(i,j)] = cell
        return grid

    def init_streets(self, n_init):
        i=0
        while i < n_init:
            init_pos = (np.random.randint(0,self.n), np.random.randint(0,self.n))
            try:
                empty_cell = self.grid[init_pos]
                self.grid[init_pos] = StreetNode(init_pos, empty_cell, self.get_neighbors((init_pos[0],init_pos[1]), self.street_field_radius), self)
                self.street_nodes += [self.grid[init_pos]]
                del empty_cell
                second_pos = (init_pos[0]+np.random.randint(-5,6), init_pos[1])
                third_pos = (init_pos[0], init_pos[1]+np.random.randint(-5,6))

        self.lay_street(init_pos, second_pos)
        self.lay_street(init_pos, third_pos)

    def lay_street(self, init_pos, second_pos):
        empty_cell = self.grid[second_pos]
        self.grid[second_pos] = StreetNode(second_pos, empty_cell, self.get_neighbors((second_pos[0],second_pos[1]), self.street_field_radius), self)
        self.street_nodes += [self.grid[second_pos]]
        inter_cells = list(bresenham(init_pos[0], init_pos[1], second_pos[0], second_pos[1]))
        for cell in inter_cells[1:-1]:
            self.grid[cell] = StreetRoute(cell, self.grid[cell], self)
        del empty_cell

    def init_activity(self, n_house_inits):
        for streetNode in self.street_nodes:
            # print('street_node', streetNode.pos)
            i = 0
            while i < n_house_inits:
                # try:
                pos = (streetNode.pos[0]+np.random.randint(-2,3),streetNode.pos[1]+np.random.randint(-2,3))
                try:
                    if self.grid[pos].value != 0:
                        continue
                except:
                    continue
                clas = self.types[0]
                self.add_activity(tuple(pos), clas)
                i += 1

    # adds a activity at specified position
    def add_activity(self, pos, clas):
        empty_cell = self.grid[pos]
        self.grid[pos] = clas(pos, empty_cell, self, self.init_decay, self.mature_decay)
        del empty_cell
        self.all_activities += [self.grid[pos]]

    # deletes activity at specified position
    def delete_activity(self, act):
        self.grid[act.pos] = Empty(act.pos, act.neighborhood, act.field_neighbors, self.n)
        self.all_activities.remove(act)
        del act

    # return a list of all acitivity objects on the city grid
    def get_all_activities(self):
        activities = []
        for pos in combinations_with_replacement(range(self.n), 2):
            if self.grid[pos]:
                activities.append(self.grid[pos])
        return activities

    # determines the type of an activity by recalculating the type probabilities
    def update_types(self):
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

    # deletes all activities that are declining
    def delete_declining(self):
        [self.delete_activity(act) for act in self.all_activities if act.type == 'decline']

    # get the candidate positions in the field of an initiate state
    def get_grow_candidates(self, act):
        grow_candidates = [[candidate,act] for candidate in act.field_neighbors if isinstance(self.grid[candidate], Empty)]
        grow_probs = [calc_grow_prob(np.array(candidate_pos[0]), np.array(act.pos), self.dist_decay) for candidate_pos in grow_candidates]
        return [grow_candidates[i] for i in range(len(grow_candidates)) if np.random.rand() < grow_probs[i]]

    # checks the activity in the neighborhood of a candidate, when above threshold activity shall be made 
    # def activity_check(self, candidate):
    #     candidate_neighborhood = self.grid[candidate].neighborhood
    #     activity = 0
    #     for pos in candidate_neighborhood:
    #         if isinstance(self.grid[pos], Activity):
    #             activity += 1
    #     return self.activity_threshold <= activity/len(candidate_neighborhood)


    # function to implement;
    def determine_activity(self, grow_candidates):
        real_list = []
        for candidate in grow_candidates:
            z = np.random.rand()
            new_node_neighborhood = self.grid[candidate[0]].field_neighbors
            if isinstance(candidate[1], Housing):
                # inMatrix = kans van huis naar huis, industry, store
                # ook kijken naar meer huizen in de buurt == grotere kans op huis
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
        housing, industry, stores, streets = self.neighbourhood_density(new_node_neighborhood)
        if (housing > thresholds['housing'] or industry > thresholds['industry'] or stores > thresholds['stores']) and (streets == 0):
            return True
        return False

    def neighbourhood_density(self, new_node_neighborhood):
        housing = sum(self.grid[neighbor].value == 1 for neighbor in new_node_neighborhood)/len(new_node_neighborhood)
        industry= sum(self.grid[neighbor].value == 2 for neighbor in new_node_neighborhood)/len(new_node_neighborhood)
        stores = sum(self.grid[neighbor].value == 3 for neighbor in new_node_neighborhood)/len(new_node_neighborhood)
        streets = sum(self.grid[neighbor].value == 4 or self.grid[neighbor].value == 5 for neighbor in new_node_neighborhood)/len(new_node_neighborhood)
        return housing, industry, stores, streets

    def check_density_activity(self, new_node_neighborhood, thresholds):
        housing, industry, stores, streets = self.neighbourhood_density(new_node_neighborhood)
        if (housing < thresholds['housing'] and industry < thresholds['industry'] and stores < thresholds['stores'] \
                and streets > thresholds['streets']):
            return True
        return False

    def street_check(self, new_street_nodes):
        new_nodes = []
        for node in new_street_nodes:
            new_node_neighborhood = self.grid[node[0]].neighborhood
            if self.street_density_check(new_node_neighborhood, self.street_thresholds):
                new_nodes += [node]
        return new_nodes

    def vancant_cells(self, init_pos, second_pos):
        inter_cells = list(bresenham(init_pos[0], init_pos[1], second_pos[0], second_pos[1]))
        for cell in inter_cells[1:-1]:
            if self.grid[cell].value != 0:
                return False
        return True

    def initiate_streets(self):
        new_nodes = []
        for node in self.street_nodes:
            new_street_nodes = self.get_grow_candidates(node)
            new_nodes += [self.street_check(new_street_nodes)]
        for i in range(len(self.street_nodes)):
            init_node = self.street_nodes[i]
            for second_node in new_nodes[i]:
                if self.vancant_cells(init_node.pos, second_node[0]):
                    self.lay_street(init_node.pos, second_node[0])
                    break


    # initiate possible new cells
    def initiate_activity(self):
        init_sites = [act for act in self.all_activities if act.type == 'init']
        new_positions = []
        new_type = []
        for act in init_sites:
            grow_candidates = self.get_grow_candidates(act)
            # grow_candidates = [candidate for candidate in grow_candidates if self.activity_check(candidate[0])]
            new_variable = self.determine_activity(grow_candidates)

            for var in new_variable:
                if var[0] not in new_positions:
                    new_positions.append(var[0])
                    new_type.append(var[1])
        [self.add_activity(new_positions[i], new_type[i]) for i in range(len(new_positions))]

    # get the neigbors cell given neigborhood
    def get_neighbors(self, pos, radius):
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

    # time step
    def step(self):
        self.t += 1
        self.update_types()
        self.initiate_streets()
        self.initiate_activity()
        self.delete_declining()
        self.activities[0] += [sum([1 for act in self.all_activities if isinstance(act, Housing)])]
        self.activities[1] += [sum([1 for act in self.all_activities if isinstance(act, Industry)])]
        self.activities[2] += [sum([1 for act in self.all_activities if isinstance(act, Stores)])]
        # self.history += [copy.copy(self.grid)]

    def plot_growth(self):
        fig, axes = plt.subplots(4, figsize=(16,16))
        labels = ['Housing', 'Industry', 'Stores', 'Streets']
        for i, act in enumerate(self.activities):
            sns.lineplot(range(len(act)), act, ax=axes[i])
            axes[i].set_xlabel(labels[i])
            axes[i].set_ylabel('Time')
        plt.show()

class Empty(object):
    def __init__(self, pos, neighborhood, field_neighbors, n):
        self.value = 0
        self.pos = pos
        self.neighborhood = neighborhood
        self.field_neighbors = field_neighbors
        self.n = n
        # self.init_t = t

class Activity(object):
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
        # self.pi = np.exp(-self.decay*(self.city.t-self.init_t))
        # self.pm = (1-self.pi)*np.exp(-self.decay*(self.city.t-self.init_t))
        # self.pd = 1 - self.pm - self.pi

class Housing(Activity):
    def __init__(self, pos, empty_cell, city, init_decay=1/30, mature_decay=1/30):
        super().__init__(pos, empty_cell, city, init_decay, mature_decay, value=1)

    def check_activity(self, candidate):
        candidate_field = self.city.grid[candidate].field
        street_node_density = 0
        for pos in candidate_field:
            for neighbor_pos in self.city.grid[pos].neighborhood:
                if isinstance(self.city.grid[neighbor_pos], StreetNode):
                    street_node_density += 1

        return street_node_density < self.street_node_threshold

class Industry(Activity):
    def __init__(self, pos, empty_cell, city,init_decay=1/15, mature_decay=1/30):
        super().__init__(pos, empty_cell, city,  init_decay, mature_decay, value=2)

    #check activity, check all houses/stores/streets in the neighbourhood, if < threshold its okay, if > threshold not
    # def check_activity(self, candidate):
    #     candidate_field = self.city.grid[candidate].field
    #     node_density = 0
    #     house_density = 0
    #     stores_density = 0
    #     for pos in candidate_field:
    #         for neighbour in self.city.grid[pos].neighbourhood:
    #             if isinstance(self.city.grid[neighbour], StreetNode):
    #                 node_density += 1
    #             elif isinstance(self.city.grid[neighbour], Housing):
    #                 house_density += 1
    #             elif isinstance(self.city.grid[neighbour], Stores):
    #                 stores_density += 1
    #
    #     if stores_density / len(candidate_field) < 0.4 and house_density / len(candidate_field) < 0.4 and \
    #         node_density > 2:
    #         return True
    #     return False

class Stores(Activity):
    def __init__(self, pos, empty_cell, city, init_decay=1/10, mature_decay=1/30):
        super().__init__(pos, empty_cell, city, init_decay, mature_decay, value=3)

    # def check_activity(self, candidate):
    #     print(candidate)
    #     print(City.grid[candidate].field)
    #     candidate_field = self.grid[candidate].field
    #     node_density = 0
    #     house_density = 0
    #     industry_density = 0
    #     for pos in candidate_field:
    #         for neighbour in self.grid[pos].neighbourhood:
    #             if isinstance(self.grid[neighbour], StreetNode):
    #                 node_density += 1
    #             elif isinstance(self.grid[neighbour], Housing):
    #                 house_density += 1
    #             elif isinstance(self.grid[neighbour], Industry):
    #                 industry_density += 1
    #
    #     if industry_density / len(candidate_field) < 0.4 and house_density / len(candidate_field) < 0.4 and \
    #         node_density > 2:
    #         return True
    #     return False

class StreetNode(object):
    def __init__(self, pos, empty_cell, street_field, city):
        # super().__init__(pos, empty_cell, city, value=4, decay=0)
        self.pos = pos
        self.neighborhood = empty_cell.neighborhood
        self.field_neighbors = street_field
        self.value = 4

class StreetRoute(object):
    def __init__(self, pos, empty_cell, city):
        self.pos = pos
        self.city = city
        self.neighborhood = empty_cell.neighborhood
        self.field_neighbors = empty_cell.field_neighbors
        self.value = 5

class Runner(object):
    def __init__(self, change_var, vars, iterations=250):
        self.change_var = change_var
        self.vars = vars
        self.iterations = iterations
        self.results = []

    def run_experiment(self):
        for v in self.vars:
            city = City(**{self.change_var:v})
            for i in range(len(self.iterations)):
                city.step()
            # self.results.append(self.city.activities)


    # def plot_results(self):
    #     fig, axes = plt.subplots(4, figsize=(16, 16))
    #     labels = ['Housing', 'Industry', 'Stores', 'Streets']
    #     for i, act in enumerate(self.results):
    #         sns.lineplot(range(len(act)), act, ax=axes[i])
    #         axes[i].set_xlabel(labels[i])
    #         axes[i].set_ylabel('Time')
    #     plt.show()

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

@jit(nopython=True)
def euclidean_dist(pos1, pos2):
    return np.sqrt(np.sum((pos1-pos2)**2))

@jit(nopython=True)
def cumsum(l):
    return np.cumsum(l)

if __name__ ==  "__main__":
# #     t = time()
    np.random.seed(90)
    city = City(n=50)
    # city.initiate_streets()
