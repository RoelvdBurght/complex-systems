import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from time import time
import copy

from itertools import combinations_with_replacement

from numba import int32, float32, types, typed, jit    # import the types
# from numba.experimental import jitclass


# creates a grid that of nxn with a single unspecified activity in the middle
class City(object):
    def __init__(self, n=10, n_radius=1, field_radius=2, dist_decay=0.8, activity_threshold=1/8):
        self.n = n
        self.t = 0
        self.n_radius = n_radius
        self.field_radius = field_radius
        self.dist_decay = dist_decay
        self.activity_threshold = activity_threshold
        self.all_activities = []
        self.grid = self.initialise_grid()
        self.add_activity((n//2,n//2))
        self.add_activity((0,0))
        # self.add_activity((n//2+1,n//2+1))    
        self.history = [self.grid]

    def initialise_grid(self):
        return np.array([[Empty((i,j), self.n_radius, self.field_radius, self.n) for i in range(self.n)] for j in range(self.n)])

    # adds a activity at specified position
    def add_activity(self, pos):
        empty_cell = self.grid[pos]
        self.grid[pos] = Activity(pos, empty_cell, self)
        del empty_cell
        self.all_activities += [self.grid[pos]]
        

    # deletes activity at specified position
    def delete_activity(self, act):
        self.grid[act.pos] = Empty(act.pos, self.n_radius, self.field_radius, self.n)
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
            probsum = np.cumsum([act.pi, act.pm, act.pd])
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

    # get the probability an activity is starting at candidate position
#     def grow_prob(self, candidate_pos, act_pos):
#         return np.exp(-self.dist_decay*np.linalg.norm(act_pos - candidate_pos))
    
    # get the candidate positions in the field of an initiate state
    def get_grow_candidates(self, act):
        grow_candidates = [candidate for candidate in act.field_neighbors if isinstance(self.grid[candidate],Empty)]
        # print(grow_candidates)
        grow_probs = [grow_prob(np.array(candidate_pos), np.array(act.pos), self.dist_decay) for candidate_pos in grow_candidates]
        return [grow_candidates[i] for i in range(len(grow_candidates)) if np.random.rand() < grow_probs[i]]

    # checks the activity in the neighborhood of a candidate, when above threshold activity shall be made 
    def candidate_check(self, candidate):
        # print(candidate)
        candidate_neighborhood = self.grid[candidate].neighborhood
        # print(candidate_neighborhood)
        activity = 0
        for pos in candidate_neighborhood:
            if isinstance(self.grid[pos], Activity):
                # print(pos)
                activity += 1
        # print(activity)
        return self.activity_threshold <= activity/((2*self.n_radius+1)**2-1)

    # function to implement;
    def determine_activity(self):
        pass

    # initiate possible new cells
    def initiate_new(self):
        init_sites = [act for act in self.all_activities if act.type == 'init']
        new_activities = []
        for act in init_sites:
            grow_candidates = self.get_grow_candidates(act)
            new_activities += [candidate for candidate in grow_candidates if self.candidate_check(candidate)]
            # print(new_activities)
            # print('--------------------------------')
            # self.determine_activity()
        [self.add_activity(act_pos) for act_pos in set(list(new_activities))]

        # 1 determine if cell is canididate for growing
        # 2 check neighborhood of cell

    # time step
    def step(self):
        self.t += 1
        self.update_types()
        self.initiate_new()
        self.delete_declining()
        self.history += [copy.copy(self.grid)]

class Empty(object):
    def __init__(self, pos, n_radius, field_radius, n):
        self.value = 0
        self.pos = pos
        self.n = n
        self.neighborhood = self.get_neighbors(n_radius)
        self.field_neighbors = self.get_neighbors(field_radius)

    # get the neigbors cell given neigborhood
    def get_neighbors(self, radius):
        neighbors = []
        row, col = self.pos
        for i in range(col-radius, col+radius+1):
            for j in range(row-radius, row+radius+1):
                if not (0 <= i < self.n and 0 <= j < self.n and (i,j) != self.pos):
                    continue
                neighbors += [(i,j)]
        return neighbors

class Activity(object):
    def __init__(self, pos, empty_cell, city):
        self.value = 1
        self.decay = 1/30
        self.pos = pos
        self.city = city
        self.init_t = city.t
        self.type = 'new'
        self.neighborhood = empty_cell.neighborhood
        self.field_neighbors = empty_cell.field_neighbors
        self.calc_probs()
        
    def calc_probs(self):
        self.pi = np.exp(-self.decay*(self.city.t-self.init_t))
        self.pm = (1-self.pi)*np.exp(-self.decay*(self.city.t-self.init_t))
        self.pd = 1 - self.pm - self.pi
        
# get the probability an activity is starting at candidate position
# function is outside the class to use jit function
# gives huge model speedup
@jit(nopython=True)
def grow_prob(candidate_pos, act_pos, dist_decay):
    return np.exp(-dist_decay*euclidean_dist(act_pos, candidate_pos))

@jit(nopython=True)
def euclidean_dist(pos1, pos2):
    return np.sqrt(np.sum((pos1-pos2)**2))


# if __name__ ==  "__main__":
# #     t = time()
#     C = City(n=10)
#     for _ in range(10):
#         C.step()
#     print(C.grid)   

# #     print(time()-t)
