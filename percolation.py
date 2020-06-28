import pickle
import os
import city_model as cc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
city = cc.City()

# Check if percolation occurs
def find_perco(grid):
    """
    check if the activity is street or street nodes, then loop through neighbours and check activity as well
    dd to a list, loop through whole list and append when new neighbours with street or street nodes activity are
    found. If Y-axis == 99 it means we got through the other side
    :return: List with all streets, boolean whether the system percolated
    """
    list_neighbours = []
    is_percolation = False
    poslist = []
    for i in range(100):
        if grid[(0, i)] == 4 or grid[(0, i)] == 5:
            poslist.append((0, i))
    for i in poslist:
        neighbourhood = city.get_neighbors(pos=i, radius=1)
        for neighbour in neighbourhood:
            if grid[neighbour] == 4 or grid[neighbour] == 5:
                list_neighbours.append(neighbour)
        for position in list_neighbours:
            new_neighbourhood = city.get_neighbors(pos=position, radius=1)
            for neighbour in new_neighbourhood:
                if neighbour[0] == 99:
                    is_percolation = True
                if grid[neighbour] == 4 or grid[neighbour] == 5:
                    if neighbour not in list_neighbours:
                        list_neighbours.append(neighbour)
    return list_neighbours, is_percolation

def percolation_grid(list_neighbours):
    """
    Make empty list, grid sized. Loop over the list of streets, put 1 on the grid to show the streets
    """
    gridz = np.zeros((100, 100))
    for pos in list_neighbours:
        gridz[pos] = 1
    plt.figure(dpi=120)
    sns.heatmap(gridz)
    plt.show()

def plotperculationone():
    """
    Plot the graph for the ratio of percolation per threshold value
    """
    last_list = []
    x = [0, '1/8', '2/8', '3/8', '4/8', '5/8', '6/8', '7/8']
    # create an index for each tick position
    xi = list(range(len(x)))
    for i in range(len(total_list)):
        counter = 0
        for j in total_list[i]:
            if j == True:
                counter += 1
        perc = counter / len(total_list[i])
        last_list.append(perc)
    plt.figure(dpi=120)
    plt.plot(last_list)
    plt.xlabel('p')
    plt.ylabel(r'P$\infty$(p)')
    plt.xticks(xi, x)
    plt.title('Ratio of percolation')
    plt.savefig('results/images/perculation.pdf')
    plt.show()

def plotperculationtwo():
    """
    Check if there is a power law happening in our system
    """
    x = abs(np.array([0,1/8,2/8,3/8,4/8,5/8,6/8,7/8])-3/8)
    last_list = []
    for i in range(len(total_list)):
        counter = 0
        for j in total_list[i]:
            if j == True:
                counter += 1
        perc = counter / len(total_list[i])
        last_list.append(abs(perc))
    plt.figure(dpi=120)
    plt.loglog(x,last_list)
    plt.xlabel('p')
    plt.ylabel(r'P$\infty$(p)')
    plt.title('Ratio of percolation')
    plt.savefig('results/images/perculation2.pdf')
    plt.show()

# go through all folders and check per parameter per folder if percolation happens, save this to a list
temp = 'results/street_params/'
paths = [temp  + f for f in os.listdir(temp)]
total_list = []
for path in paths:
    print(path)
    files = [path + '/' + f for f in os.listdir(path)]
    partial_list = []
    for i in range(10):
        res = pickle.load(open(files[i], 'rb'))
        list_neighbours, is_percolation = find_perco(res.activity_grid)
        print('Is there a percolation for',i , is_percolation)
        partial_list.append(is_percolation)
    total_list.append(partial_list)

with open('results/lists/wiebe.txt', 'w') as f:
    for item in total_list:
        f.write("%s\n" % item)

plotperculationtwo()
plotperculationone()