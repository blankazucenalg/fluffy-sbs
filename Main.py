import logging
from random import random, randint, gauss

import pandas as pd
import numpy as np

logging.basicConfig(filename='logging.log', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

earthquakes_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

Y_train = earthquakes_df["Magnitude"]
Y_test = test_df["Magnitude"]

# drop labels from training set, given that label is the dependant feature
# and we want to predict label given predictors or independent features
X_train = earthquakes_df.drop("Magnitude", axis=1)
X_train = X_train.drop("Dangerous", axis=1)

# Test set (droping rows with NULL values)
# make a copy of PassengerId
X_test = test_df.drop("Magnitude", axis=1).dropna().copy()
X_test = X_test.drop("Dangerous", axis=1).dropna().copy()

X_train = X_train.as_matrix()
Y_train = Y_train.as_matrix()
X_test = X_test.as_matrix()
Y_test = Y_test.as_matrix()


# lin = LinearRegression() #initialize regressor
class GeneticAlgorithm:
    def __init__(self):
        pass

    def fit(self, X_train, Y_train, population_size, epsilon=0.1, epochs=3):
        self.population_size = population_size
        self.chromosome_length = len(X_train[0])
        self.population = np.zeros([population_size, self.chromosome_length])
        for i in range(population_size):
            for j in range(self.chromosome_length):
                self.population[i][j] = random() *10
        for t in range(len(X_train)*epochs):
            self.value = Y_train[t%len(X_train)]
            self.train = X_train[t%len(X_train)]
            fitness = self.population_fitness() #Error existente reales y pob
            while fitness[0] > epsilon:
                new_population = self.generate_new_population(fitness)
                self.population = (self.choose_best(new_population))[0:self.population_size]
                fitness = self.population_fitness()
            logging.info("**" * 60 + "\n")
            logging.info("At train index %s" % (str(t%len(X_train))))
            logging.info("%s %s \nError= %s" % (self.train, self.population[0], abs(fitness[0])))
            logging.info("**" * 60 + "\n")

    def population_fitness(self):
        fitness = []
        for single in self.population:
            fitness.append(self.fitness_calculation(single))
        return fitness

    def choose_best(self, population):
        return sorted(population, key=self.fitness_calculation, reverse=False)

    def generate_new_population(self, fitness):
        new_population = []
        for i in range(0, int(self.population_size / 2)):
            position_one = self.choose_single(fitness)
            position_two = self.choose_single(fitness)
            #
            father = np.array(self.population[position_one])
            mother = np.array(self.population[position_two])
            #
            son_one, son_two = self.one_point_crosses(father, mother)
            #
            self.mutate(son_one)
            self.mutate(son_two)
            #
            new_population.append(son_one)
            new_population.append(son_two)
        new_population = np.concatenate((self.population,new_population))

        return new_population

    def choose_single(self, fitness):
        fitness_sum = sum(fitness)
        fitness_average = fitness_sum / self.population_size

        expected_values = []
        for single_fitness in fitness:
            expected_value = single_fitness / fitness_average
            expected_values.append(expected_value)

        random_number = random() * self.population_size  # same as random.random()*T
        partial_sum = 0.0
        index = 0
        for expected_value in expected_values:
            partial_sum = partial_sum + expected_value
            if partial_sum >= random_number:
                return index
            index = index + 1

    def one_point_crosses(self, father, mother):
        cross_point = randint(1, self.chromosome_length - 1)
        #
        left_side_father = father[0:cross_point]
        right_side_father = father[cross_point:self.chromosome_length]
        #
        left_side_mother = mother[0:cross_point]
        right_side_mother = mother[cross_point:self.chromosome_length]
        #
        son_one = np.concatenate((left_side_father, right_side_mother))
        son_two = np.concatenate((left_side_mother, right_side_father))

        return son_one, son_two

    def mutate(self, single):
        mutation_point = randint(0, self.chromosome_length - 1)
        single[mutation_point] += gauss(0, 1)

    def fitness_calculation(self, single):
        return abs(self.value - single * np.matrix(self.train).transpose())

    def predict(self, X_test):
        return [(self.population[0] * np.matrix(test).transpose()) for test in X_test]

    def score(self,X_train, Y_train):
        results = self.predict(X_train)
        error = abs(Y_train - results)
        return 0.5

ga = GeneticAlgorithm()
ga.fit(X_train, Y_train, 10, epochs=3)  # fit training data

preds = ga.predict(X_train)  # make prediction on X test set

results = open('results,csv', 'w')
results.write("Magnitude\n")
for p in preds:
    results.write(p.__str__()+"\n")
results.close()

