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
import numpy as np



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
    	This is the entry point for the default solver
    	which just finds a valid random tour.  Note this could be used to find your
    		initial BSSF.
    		</summary>
    		<returns>results dictionary for GUI that contains three ints: cost of solution, 
    		time spent to find solution, number of permutations tried during search, the 
    		solution found, and three null values for fields not used for this 
    		algorithm</returns> 
    	'''

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
        n_population = 5
        n_iter = 2000
        selectivity = 0.15
        p_cross = 0.5
        p_mut = 0.1
        print_interval = 100
        return_history = False
        verbose = False
        # Create adjacency matrix
        matrix = self.initializeMatrix(cities)
        start_time = time.time()
        pop = self.init_population(matrix, 40)
        best = pop.best
        score = float("inf")
        history = []
        popDifference = 1
        while popDifference != 0:
            pop.select(n_population * selectivity)
            history.append(pop.cost)
            if pop.cost < score:
                best = pop.best
                score = pop.cost
            children = pop.mutate(p_cross, p_mut)
            pop = Population(children, pop.matrix)
            if len(history) > 2:
                if history[-1] - history[-2] == 0:
                    break

        end_time = time.time()
        best = TSPSolution(best)
        results['cost'] = score
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = best
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    #Use the default tour to generate a subset of potential solutions (a population)
    def init_population(self, adjacency_mat, n_population):
        return Population(
            np.asarray([self.greedy(60)['soln'].route for _ in range(n_population)]),
            adjacency_mat
        )

class Population():
    def __init__(self, solutions, adjacency_matrix):
        self.cost = 0
        self.best = None
        self.matrix = adjacency_matrix
        self.potentialSolutions = solutions
        self.parents = []


    def fitness(self, chromosome):
        sum = 0
        for i in range(len(chromosome) - 1):
            sum += chromosome[i].costTo(chromosome[i+1])
        return sum

    #evaluate the fitness of each member of the population of potential solutions
    def evaluate(self):
        distances = np.asarray(
            [self.fitness(chromosome) for chromosome in self.potentialSolutions]
        )
        #the cost of the best solution is the solution with the best distance cost
        self.cost = np.min(distances)
        #return the solution with the minimum distance, finding the index to the solution with the minimum cost
        self.best = self.potentialSolutions[distances.tolist().index(self.cost)]
        #append the best solution to the next generation, to mutate from
        self.parents.append(self.best)
        #return a variant of the max distance minus all distances involved
        if False in (distances[0] == distances):
            distances = np.max(distances) - distances
        #return will be an accurate evaluation of the fitness of an individual compared to the population
        return distances / np.sum(distances)

#use random probability to select the parents of the next population
    """Given the evaluation scores of each potential solution, use random numbers to search for the next parents
        If the value of the index in the probability vector form the evaluation is higher, we append the parent 
        at that index. Repeat this process until we have k parents.
    """
    def select(self, k=4):
        fitness = self.evaluate()
        while len(self.parents) < k:
            index = np.random.randint(0, len(fitness))
            if fitness[index] > np.random.rand():
                self.parents.append(self.potentialSolutions[index])
        self.parents = np.asarray(self.parents)

    """A function to swap two random genes(cities) in a solution. This function can be dangerous due to the
        fact that not every city as a path to every other city. Hence, we must use this function in concurrence with
         a crossover function, making sure that the swapping of these two cities does not greatly increase the cost 
         of the current solution
    """
    def swap(self, chromosome):
        a, b = np.random.choice(len(chromosome), 2)
        #swap two random cities in the solution
        chromosome[a], chromosome[b] = (
            chromosome[b],
            chromosome[a],
        )
        return chromosome


    def crossover(self, p_cross=0.1):
        children = []
        count, size = self.parents.shape
        #generate an equal number of children as the prior population
        for _ in range(len(self.potentialSolutions)):
            #if the random number is greater than the probability of a crossover, then immediately
            #append a random parent to the solution
            if np.random.rand() > p_cross:
                children.append(
                    list(self.parents[np.random.randint(count, size=1)[0]])
                )
            #otherwise, pick two random parents and splice them together to generate a child
            else:
                #get two random parents
                mom, dad = self.parents[
                                   np.random.randint(count, size=2), :
                                   ]
                #need to sample without replacement, can't select the same indexes twice
                index = np.random.choice(range(size), size=2, replace=False)
                start, end = min(index), max(index)
                #initiate a child solution of all type Nones
                child = [None] * size
                #from two random indexes, generate a child's solution
                for i in range(start, end + 1, 1):
                    child[i] = mom[i]
                #at all remaining None positions, append an element from the other parent to the new child
                currentPosition = 0
                for i in range(size):
                    if child[i] is None:
                        #make sure to not append a city that has already been appended to the solution
                        while dad[currentPosition] in child:
                            currentPosition += 1
                        child[i] = dad[currentPosition]
                children.append(child)
        return children

    """Mix both the splicing and swap functions together to create a new list of potential solutions"""
    def mutate(self, p_cross=0.1, p_mut=0.1):
        potentialSolutions = []
        children = self.crossover(p_cross)

        for child in children:
            if np.random.rand() < p_mut:
                potentialSolutions.append(self.swap(child))
            else:
                potentialSolutions.append(child)
        return potentialSolutions


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