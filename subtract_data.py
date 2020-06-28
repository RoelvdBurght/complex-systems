
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def in_nested_list(my_list, item):
    """
    Determines if an item is in my_list, even if nested in a lower-level list.
    """
    if item in my_list:
        return True
    else:
        return any(in_nested_list(sublist, item) for sublist in my_list if isinstance(sublist, list))

def get_clusters(grid, activity):
    activities_grid = np.argwhere(grid==activity).tolist()
    clusters = []

    for start_pos in activities_grid[:]:
        if in_nested_list(clusters, tuple(start_pos)):
            continue
        list_neighbors = [neighbor for neighbor in get_neighbors(start_pos, radius=1, n=grid.shape[0]) if grid[neighbor] == activity]

        for neighbor in list_neighbors:
            neigbor_neighbors = get_neighbors(tuple(neighbor), radius=1,  n=grid.shape[0])
            for n in neigbor_neighbors:
                if grid[n] == activity and n not in list_neighbors:
                    list_neighbors += [n]
        clusters += [list_neighbors]
    return clusters

# get the neigbors cell given neigborhood
def get_neighbors(pos, radius, n):
        neighbors = []
        row, col = pos
        for i in range(col-radius, col+radius+1):
            for j in range(row-radius, row+radius+1):
                if not (0 <= i < n and 0 <= j < n):
                    continue
                if pos == (j,i):
                    continue
                neighbors += [(j,i)]
        return neighbors

def paths(main_path):
    all_files = []
    all_paths = [main_path.format(i) for i in np.linspace(0,7/8, 8)]
    for path in all_paths:
        all_files +=[[path + f for f in listdir(path) if isfile(join(path, f))]]
    return all_files

files = ['results/stores_thresholds2/housing_activity_0.625/run_{}.p'.format(i) for i in range(10)]
aves_house = np.zeros(10)
aves_industry = np.zeros(1)
aves_store = np.zeros(1)
for file in files:
    res = pickle.load(open(file, 'rb'))
    # sns.heatmap(res.activity_grid, cmap=['black', 'forestgreen', 'yellow', 'navy', 'grey'])
    # plt.show()

    aves_house += res.housing_over_time[-1]/10
    aves_industry += res.industry_over_time[-1]/10
    aves_store += res.stores_over_time[-1]/10

plt.plot(aves_house, label='housing', c='C2')
plt.plot(aves_industry, label='industry', c='C8')
plt.plot(aves_store, label='stores', c='C10')


plt.legend()
plt.show()
# aves = np.zeros(1000)
# stds = []
# for threshold_folder in all_files[:]:
#     cluster_size = []
#     for file in threshold_folder:
#         res = pickle.load(open(file, 'rb'))
#         aves += np.array(res.housing_over_time)/10
# plt.plot(aves)

# np.savetxt('data/industry_thres_stores_activities/housing_clusters_size.txt',aves)
# np.savetxt('data/industry_thres_stores_activities/housing_clusters_size_var.txt',np.array(stds)**2)

# x = np.linspace(0,7/8,8)
# plt.plot(aves)
# plt.show()
