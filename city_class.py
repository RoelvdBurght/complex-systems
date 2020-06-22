import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from itertools import combinations_with_replacement

from numba import int32, float32, types, typed, jit    # import the types
# from numba.experimental import jitclass


# creates a grid that of nxn with a single unspecified activity in the middle
class City(object):
    def __init__(self, n=100, n_radius=1, field_radius=2, dist_decay=0.8, activity_threshold=1/8):
        self.grid = np.zeros(shape=(n,n), dtype=object)
        self.n = n
        self.t = 0
        self.n_radius = n_radius
        self.field_radius = field_radius
        self.dist_decay = dist_decay
        self.activity_threshold = activity_threshold
        self.all_activities = []
        self.add_activity((n//2,n//2))
    
    # adds a activity at specified position
    def add_activity(self, pos):
        self.grid[pos] = Activity(pos, self)
        self.all_activities += [self.grid[pos]]
        

    # deletes activity at specified position
    def delete_activity(self, act):
        self.grid[act.pos] = None
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

    # get the neigbors cell given neigborhood
    def get_neighbors(self, site, radius):
        neighbors = []
        if isinstance(site, Activity):
            row, col = site.pos[0], site.pos[1]
        else:
            row, col  = site
        for r in range(1,radius+1):
            for i, j in ((row - r, col), (row + r, col), (row, col - r),
                (row, col + r), (row - r, col - r), (row - r, col + r),
                (row + r, col - r), (row + r, col + r)):
                if not (0 <= i < self.n and 0 <= j < self.n):
                    continue                
                if not self.grid[i][j]:
                    neighbors += [(i,j)]
        return neighbors

    # get the probability an activity is starting at candidate position
#     def grow_prob(self, candidate_pos, act_pos):
#         return np.exp(-self.dist_decay*np.linalg.norm(act_pos - candidate_pos))
    
    # get the candidate positions in the field of an initiate state
    def get_grow_candidates(self, act):
        grow_candidates = self.get_neighbors(act, self.field_radius)
        grow_probs = [grow_prob(np.array(candidate_pos), np.array(act.pos), self.dist_decay) for candidate_pos in grow_candidates]
        return [grow_candidates[i] for i in range(len(grow_candidates)) if np.random.rand() < grow_probs[i]]

    # checks the activity in the neighborhood of a candidate, when above threshold activity shall be made 
    def candidate_check(self, candidate):
        candidate_neighborhood = self.get_neighbors(candidate, self.n_radius)
        activity = 0
        for pos in candidate_neighborhood:
            if not self.grid[pos]:
                activity += 1
        return self.activity_threshold < activity/(2*self.n_radius+1)**2
        
    # function to implement;
    def determine_activity(self):
        pass
    
    # initiate possible new cells
    def initiate_new(self):
        init_sites = [act for act in self.all_activities if act.type == 'init']
#         print(len(init_sites))
        for act in init_sites:
            grow_candidates = self.get_grow_candidates(act)
            new_activities = [candidate for candidate in grow_candidates if self.candidate_check(candidate)]
            self.determine_activity()
            [self.add_activity(act_pos) for act_pos in new_activities]

        # 1 determine if cell is canididate for growing
        # 2 check neighborhood of cell

    # time step
    def step(self):
        self.delete_declining()
        self.t += 1
        self.update_types()
        self.initiate_new()

class Activity():
    def __init__(self, pos, city):
        self.decay = 1/30
        self.pos = pos
        self.city = city
        self.init_t = city.t
        self.type = 'new'
        self.calc_probs()
        
    def calc_probs(self):
        self.pi = np.exp(-self.decay*(self.city.t-self.init_t))
        self.pm = (1-self.pi)*np.exp(-self.decay*(self.city.t-self.init_t))
        self.pd = 1 - self.pm - self.pi


# get the probability an activity is starting at candidate position
# this give huge speed up...
@jit(nopython=True)
def grow_prob(candidate_pos, act_pos, dist_decay):
    return np.exp(-dist_decay*euclidean_dist(act_pos, candidate_pos))

@jit(nopython=True)
def euclidean_dist(pos1, pos2):
    return np.sqrt(np.sum((pos1-pos2)**2))