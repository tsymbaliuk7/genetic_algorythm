import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint, rand


class GeneticOptimization:
    def __init__(self, func, rng, accuracy, max_generation, crossover_rate):
        self.function = func
        self.range = rng
        self.accuracy = accuracy
        self.max_generation = max_generation
        self.bits_num = 32
        self.population_num = 100
        self.population = [list(randint(0, 2, self.bits_num)) for i in range(self.population_num)]
        self.crossover_rate = crossover_rate
        self.mutation_rate = 1.0/float(self.bits_num)

    def genetic_optimization(self, option):
        comparing = lambda a, b: a < b
        if option == 'max':
            comparing = lambda a, b: a > b
        print('GeneticOptimization searching for:', option)
        x_best, y_best = 0, self.function(self.decode(self.population[0]))
        for generation in range(self.max_generation):
            decoded_population = [self.decode(pop) for pop in self.population]
            Yi = [self.function(x) for x in decoded_population]
            for i in range(self.__len__()):
                if comparing(Yi[i], y_best):
                    x_best, y_best = self.population[i], Yi[i]
            if generation % 10 == 0:
                print('{:5} Best: f_{}({}) = {}'.format(generation, option, self.decode(x_best), y_best))
            selected_population = [self.selection(Yi, competitors_num=3, comparing=comparing) for _ in range(self.__len__())]
            new_population = []
            for i in range(0, self.population_num, 2):
                parents = (selected_population[i], selected_population[i + 1])
                for child in self.crossover(*parents):
                    self.mutate(child)
                    new_population.append(child)
            self.population = new_population
        print('Best: f_{}({}) = {}'.format(option, self.decode(x_best), y_best))
        print('-' * 20 + '\n' * 2)
        return self.decode(x_best), y_best

    def selection(self, yi, competitors_num, comparing):
        selected = randint(len(self.population))
        for i in randint(0, len(self.population), competitors_num - 1):
            if comparing(yi[i], yi[selected]):
                selected = i
        return self.population[selected]

    def crossover(self, parent1, parent2):
        child1, child2 = parent1.copy(), parent2.copy()
        if rand() < self.crossover_rate:
            point = randint(1, len(parent1) - 2)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
        return [child1, child2]

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if rand() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]

    def decode(self, bit_string):
        largest = 2 ** self.bits_num
        substring = bit_string[0: self.bits_num]
        chars = ''.join([str(s) for s in substring])
        integer = int(chars, 2)
        value = self.range[0] + (integer / largest) * (self.range[1] - self.range[0])
        return value

    def show_func(self, extremum=None):
        x = np.linspace(*self.range, 1000)
        plt.title('Function:')
        plt.plot(x, list(map(self.function, x)))
        if extremum:
            plt.plot(extremum[0], extremum[1], 'bo')
            plt.plot(extremum[2], extremum[3], 'bo')
        plt.show()

    def __len__(self):
        return self.population_num
