#!/usr/bin/python3
import copy
from sys import maxsize

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import random
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from random import random, randint, sample


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario


''' <summary>
	This is the entry point for the default solver
	which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''


def defaultRandomTour(self, time_allowance=60.0):
    results = {}
    cities = self._scenario.getCities()
    ncities = len(cities)
    foundTour = False
    count = 0
    bssf = None
    start_time = time.time()
    while not foundTour and time.time() - start_time < time_allowance:
        # create a random permutation
        perm = np.random.permutation(ncities)
        route = []
        # Now build the route using the random permutation
        for i in range(ncities):
            route.append(cities[perm[i]])
        bssf = TSPSolution(route)
        count += 1
        if bssf.cost < np.inf:
            # Found a valid route
            foundTour = True
    end_time = time.time()
    results['cost'] = bssf.cost if foundTour else math.inf
    results['time'] = end_time - start_time
    results['count'] = count
    results['soln'] = bssf
    results['max'] = None
    results['total'] = None
    results['pruned'] = None
    return results


''' <summary>
    This is the entry point for the greedy solver, which you must implement for 
    the group project (but it is probably a good idea to just do it for the branch-and
    bound project as a way to get your feet wet).  Note this could be used to find your
    initial BSSF.
    </summary>
    <returns>results dictionary for GUI that contains three ints: cost of best solution, 
    time spent to find best solution, total number of solutions found, the best
    solution found, and three null values for fields not used for this 
    algorithm</returns> 
'''


# Time complexity of O(n^2) because we loop through every city to find the cost to every other city.
# Space complexity of O(n) just storing an array of cities
def greedy(self, time_allowance=60.0):
    results = {}
    cities = self._scenario.getCities()
    foundTour = False
    count = 0
    ncities = len(cities)
    bssf = None
    start_time = time.time()

    # Initially choose a random city to start at.
    # While the cities aren't added to the route
    while not foundTour:
        count += 1
        cities_list_copy = cities.copy()
        startIndex = random.randint(0, ncities - 1)
        route = [cities_list_copy.pop(startIndex)]
        while len(cities_list_copy) > 0:
            least_cost_index = 0
            least_cost = route[-1].costTo(cities_list_copy[0])
            for i in range(len(cities_list_copy)):
                if least_cost > route[-1].costTo(cities_list_copy[i]):
                    least_cost_index = i
                    least_cost = route[-1].costTo(cities_list_copy[i])
            route.append(cities_list_copy.pop(least_cost_index))
        foundTour = not (route[-1].costTo(route[0]) == np.inf)

        bssf = TSPSolution(route)
    end_time = time.time()
    results['cost'] = bssf.cost if foundTour else math.inf
    results['time'] = end_time - start_time
    results['count'] = count
    results['soln'] = bssf
    results['max'] = None
    results['total'] = None
    results['pruned'] = None
    return results


''' <summary>
    This is the entry point for the branch-and-bound algorithm that you will implement
    </summary>
    <returns>results dictionary for GUI that contains three ints: cost of best solution, 
    time spent to find best solution, total number solutions found during search (does
    not include the initial BSSF), the best solution found, and three more ints: 
    max queue size, total number of states created, and number of pruned states.</returns> 
'''


# Time complexity is O(n!n^3) because you could possilby have every state gnereate another n amount of states O(n!). We then have to expand each state O(n), and then reduce the matrix O(n^2)
# Space complexity is O(n!n^2) because each state only reuqires n^2 space * n! states you could possibly have
def branchAndBound(self, time_allowance=60.0):
    results = {}
    cities = self._scenario.getCities()
    max_length = 0
    count = 0
    bssf_cost = np.inf
    bssf = None
    num_pruned = 0
    child_count = 0
    ncities = len(cities)
    priority_queue = []
    start_time = time.time()

    # Time Complexity: O(n^3) on average will return on 1st but worst case it goes thorugh every single state until it gets to the end.
    # Space Complexity: O(n) space gets preserved from the greedy algorithm and only stores the path of the solution of n cities
    while bssf_cost == np.inf:
        greedy_cost = self.greedy()
        bssf_cost = greedy_cost['cost']
        bssf = greedy_cost['soln']
    child_count += 1
    count += 1

    # Time Complexity: O(n^2) to initialize the nxn matrix
    matrix = self.initializeMatrix(cities)
    start_state = StateClass()
    start_state.matrix = matrix
    lower_bound, start_state.matrix = self.reduceMatrix(start_state.availableCitiesToVisit(cities), start_state.matrix,
                                                        start_state.columns, start_state.rows)
    start_state.lower_bound = lower_bound
    start_index = random.randint(0, ncities - 1)
    start_state.path.append(cities[start_index])
    start_state.city_index = start_index

    # Push the first state on the queue
    heapq.heappush(priority_queue, start_state)

    # Time Complexity: O(n!) because worst case every state generates n more states
    # Space Complexity: O(n!)
    while len(priority_queue) > 0 and time.time() - start_time < time_allowance:
        # get the max length of the priority queue
        temp_queue_len = len(priority_queue)
        if max_length < temp_queue_len:
            max_length = temp_queue_len

        current_state = heapq.heappop(priority_queue)
        # Prune if the lower bound is greater than the upper bound
        if current_state.lower_bound > bssf_cost:
            num_pruned += 1
            continue

        available_state = current_state.availableCitiesToVisit(cities)
        # Time Complexity: O(n^3) because we go through each available state and reduce the cost matrix which takes O(n^2) time
        # Space Complexity: O(n^2) stores the matrix of the cost from each city to each city and priority queue of states
        # This loops through and expands the current state
        for i in available_state:
            child_count += 1
            if matrix[current_state.city_index][i] == np.inf:
                num_pruned += 1
                continue
            new_state = StateClass()
            new_state.parent = current_state
            new_state.city_index = i
            new_state.matrix = copy.deepcopy(current_state.matrix)
            for k in current_state.path:
                new_state.path.append(k)
            new_state.path.append(cities[i])
            # deep copy each element one by one to eventually get the reduced cost matrix of the current state
            new_state.depth = copy.deepcopy(current_state.depth)
            new_state.depth += 1
            new_state.lower_bound = copy.deepcopy(current_state.lower_bound)
            new_state.lower_bound += new_state.matrix[current_state.city_index][i]
            new_state.columns = copy.deepcopy(current_state.columns)
            new_state.rows = copy.deepcopy(current_state.rows)
            # Infinity the row and column of the cities we are searching
            # i (column) is the city we are going to and current city index is the city we are coming from (row)
            for j in range(ncities):
                if new_state.matrix[j][i] != np.inf:
                    new_state.matrix[j][i] = np.inf
            new_state.columns.append(i)
            for j in range(ncities):
                if new_state.matrix[current_state.city_index][j] != np.inf:
                    new_state.matrix[current_state.city_index][j] = np.inf
            new_state.rows.append(current_state.city_index)

            # O(n^2) time because we are caling reduceMatrix()
            reduced_matrix_bound, new_state.matrix = self.reduceMatrix(new_state.availableCitiesToVisit(cities),
                                                                       new_state.matrix, new_state.columns,
                                                                       new_state.rows)
            new_state.lower_bound += reduced_matrix_bound

            # Check if solution is complete by reaching the end of our cities, add to queue, or prune
            if new_state.lower_bound < bssf_cost:
                if new_state.depth == ncities and cities[new_state.city_index].costTo(
                        cities[start_state.city_index]) != np.inf:
                    # Gets the potential solution
                    solution = TSPSolution(new_state.path)
                    count += 1
                    if solution.cost < bssf.cost:
                        bssf = TSPSolution(new_state.path)
                        bssf_cost = bssf.cost
                else:
                    heapq.heappush(priority_queue, new_state)
            else:
                num_pruned += 1

    end_time = time.time()
    results['cost'] = bssf.cost
    results['time'] = end_time - start_time
    results['count'] = count
    results['soln'] = bssf
    results['max'] = max_length
    results['total'] = child_count
    results['pruned'] = num_pruned
    return results


# Time Complexity: O(n^2) to go through the list of cities twice to get the cost of each city to each city
# Space Complexity: o(n^2) has to store an nxn matrix
def initializeMatrix(self, listOfCities):
    adjacencyMatrix = [[np.inf for i in range(len(listOfCities))] for j in range(len(listOfCities))]
    for i in range(len(listOfCities)):
        for j in range(len(listOfCities)):
            if i != j:
                adjacencyMatrix[i][j] = listOfCities[i].costTo(listOfCities[j])
    return adjacencyMatrix


# Time and Space complexity is O(n^2) because it stores a matrix of n cities x n cities. Has to loop through each row and column to get the smallest distance and reduce the matrix.
def reduceMatrix(self, cities, adjacency_matrix, columns, rows):
    lower_bound = 0
    length = len(adjacency_matrix)
    # Find the smallest cost of each row
    # Time Complexity: O(n^2) because we go through each city in each row
    for i in range(len(cities)):
        # if the current row is already all infinities then don't search for the smallest distance
        if i in rows:
            continue
        min_cost = min(adjacency_matrix[i][:])
        if min_cost < np.inf:
            lower_bound += min_cost
            # O(n)
            adjacency_matrix[i] = list(map(lambda row: row - min_cost, adjacency_matrix[i]))
        else:
            lower_bound = np.inf
            return lower_bound, adjacency_matrix
    # Find the smallest distance of each column
    # Time Complexity: O(n^2) because you go through each city in each row
    for j in range(len(cities)):
        # if the column is already all infinities then don't search for the smallest distance
        if j in columns:
            continue
        # O(n)
        min_cost = min(list(map(lambda col: col[j], adjacency_matrix)))
        if min_cost == 0:
            continue
        if min_cost < np.inf:
            lower_bound += min_cost
            for i in range(length):
                adjacency_matrix[i][j] -= min_cost
        else:
            lower_bound = np.inf
    return lower_bound, adjacency_matrix


''' <summary>
    This is the entry point for the algorithm you'll write for your group project.
    </summary>
    <returns>results dictionary for GUI that contains three ints: cost of best solution, 
    time spent to find best solution, total number of solutions found during search, the 
    best solution found.  You may use the other three field however you like.
    algorithm</returns> 
'''


def fancy(self, time_allowance=60.0):
    results = {}
    cities = self._scenario.getCities()
    foundTour = False
    count = 0
    ncities = len(cities)
    bssf = None
    start_time = time.time()

    # Create adjacency matrix
    matrix = self.initializeMatrix(cities)

    end_time = time.time()
    results['cost'] = bssf.cost if foundTour else math.inf
    results['time'] = end_time - start_time
    results['count'] = count
    results['soln'] = bssf
    results['max'] = None
    results['total'] = None
    results['pruned'] = None
    return results


def evolve(pop, tourn_size, mut_rate):
    new_generation = Population([])
    pop_size = len(pop.individuals)
    elitism_num = pop_size // 2

    # Elitism
    for _ in range(elitism_num):
        fittest = pop.get_fittest()
        new_generation.add(fittest)
        pop.rmv(fittest)

    # Crossover
    for _ in range(elitism_num, pop_size):
        parent_1 = selection(new_generation, tourn_size)
        parent_2 = selection(new_generation, tourn_size)
        child = crossover(parent_1, parent_2)
        new_generation.add(child)

    # Mutation
    for i in range(elitism_num, pop_size):
        mutate(new_generation.individuals[i], mut_rate)

    return new_generation


def crossover(parent_1, parent_2):
    def fill_with_parent1_genes(child, parent, genes_n):
        start_at = randint(0, len(parent.genes) - genes_n - 1)
        finish_at = start_at + genes_n
        for i in range(start_at, finish_at):
            child.genes[i] = parent_1.genes[i]

    def fill_with_parent2_genes(child, parent):
        j = 0
        for i in range(0, len(parent.genes)):
            if child.genes[i] == None:
                while parent.genes[j] in child.genes:
                    j += 1
                child.genes[i] = parent.genes[j]
                j += 1

    genes_n = len(parent_1.genes)
    child = Individual([None for _ in range(genes_n)])
    fill_with_parent1_genes(child, parent_1, genes_n // 2)
    fill_with_parent2_genes(child, parent_2)

    return child


def mutate(individual, rate):
    for _ in range(len(individual.genes)):
        if random() < rate:
            sel_genes = sample(individual.genes, 2)
            individual.swap(sel_genes[0], sel_genes[1])


def selection(population, competitors_n):
    return Population(sample(population.individuals, competitors_n)).get_fittest()


def run_ga(genes, pop_size, n_gen, tourn_size, mut_rate, verbose=1):
    population = Population.gen_individuals(pop_size, genes)
    history = {'cost': [population.get_fittest().travel_cost]}
    counter, generations, min_cost = 0, 0, maxsize

    if verbose:
        print("-- TSP-GA -- Initiating evolution...")

    start_time = time()
    while counter < n_gen:
        population = evolve(population, tourn_size, mut_rate)
        cost = population.get_fittest().travel_cost

        if cost < min_cost:
            counter, min_cost = 0, cost
        else:
            counter += 1

        generations += 1
        history['cost'].append(cost)

    total_time = round(time() - start_time, 6)

    if verbose:
        print("-- TSP-GA -- Evolution finished after {} generations in {} s".format(generations, total_time))
        print("-- TSP-GA -- Minimum travelling cost {} KM".format(min_cost))

    history['generations'] = generations
    history['total_time'] = total_time
    history['route'] = population.get_fittest()

    return history


# A single route that is a possible solution to TSP
class Individual:
    def __init__(self, genes):
        assert (len(genes) > 3)
        self.genes = genes
        self.__reset_params()

    def swap(self, gene_1, gene_2):
        self.genes[0]
        a, b = self.genes.index(gene_1), self.genes.index(gene_2)
        self.genes[b], self.genes[a] = self.genes[a], self.genes[b]
        self.__reset_params()

    def add(self, gene):
        self.genes.append(gene)
        self.__reset_params()

    @property
    def fitness(self):
        if self.__fitness == 0:
            self.__fitness = 1 / self.travel_cost  # Normalize travel cost
        return self.__fitness

    @property
    def travel_cost(self):  # Get total travelling cost
        if self.__travel_cost == 0:
            for i in range(len(self.genes)):
                origin = self.genes[i]
                if i == len(self.genes) - 1:
                    dest = self.genes[0]
                else:
                    dest = self.genes[i + 1]

                self.__travel_cost += origin.get_distance_to(dest)

        return self.__travel_cost

    def __reset_params(self):
        self.__travel_cost = 0
        self.__fitness = 0


class Population:  # Population of individuals
    def __init__(self, individuals):
        self.individuals = individuals

    @staticmethod
    def gen_individuals(sz, genes):
        individuals = []
        for _ in range(sz):
            individuals.append(Individual(sample(genes, len(genes))))
        return Population(individuals)

    def add(self, route):
        self.individuals.append(route)

    def rmv(self, route):
        self.individuals.remove(route)

    def get_fittest(self):
        fittest = self.individuals[0]
        for route in self.individuals:
            if route.fitness > fittest.fitness:
                fittest = route

        return fittest


# Holds all the data of a state for the Branch and Bound algorithm
class StateClass:
    def __init__(self):
        self.lower_bound = np.inf
        self.depth = 1
        self.parent = None
        self.matrix = [[]]
        self.city_index = None
        self.path = []
        self.columns = []
        self.rows = []

    # Compares two states by the ratio of the lower bound * 2 to depth
    # Time and Space complexity are both O(1) because multiplication, division, and comparison are all constant time operators and nothing is stored
    def __lt__(self, other):
        return (2 * self.lower_bound) / self.depth < (2 * other.lower_bound) / self.depth

    # Space Complexity: O(n) because it stores an array of n cities.
    # Time Complexity: O(n^2) because we go through our list of cities and then go through the cities in the path to check if that city is availble to vist
    def availableCitiesToVisit(self, listOfCities):
        available_cities = []
        for i in range(len(listOfCities)):
            if listOfCities[i] not in self.path:
                available_cities.append(i)
        return available_cities