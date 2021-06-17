from math import cos, sin
from GeneticOptimization import GeneticOptimization

if __name__ == '__main__':
    rng = (0, 5)
    y = lambda x:  x ** (1 / 2) * sin(10 * x)
    genetic = GeneticOptimization(y, rng, 0.001, 100, 0.9)
    genetic.show_func()

    x_min, y_min = genetic.genetic_optimization('min')
    x_max, y_max = genetic.genetic_optimization('max')

    genetic.show_func(extremum=(x_min, y_min, x_max, y_max))
