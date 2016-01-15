import logging
import numpy as np
import math
from random import random, uniform


def tansig(val):
    y = np.array([[((2 / (1 + math.e ** (-2 * x))) - 1) for x in k] for k in val])
    return y


def purelin(x):
    return x


def deltasig(a1):
    return np.array([[(1 - x) * (1 + x) for x in k] for k in a1])


def deltalin():
    return 1


class Net:
    logging.basicConfig(filename='logger.log', level=logging.DEBUG)

    def __init__(self, input_length, neurons, target_length):
        # Initialize the weights with random values
        self.weights1 = np.zeros([neurons, input_length])
        self.weights1.fill(1)
        self.threshold1 = np.zeros([neurons, 1])
        self.weights2 = np.zeros([target_length, neurons])
        self.threshold2 = np.zeros([1, 1])

    def auto_learning_speed(self):
        self.alpha = 0.2
        d = self.eigenvalues(self.compute_R(self.training_patterns, 0))

    def learn(self, training_patterns, training_targets, desired_error):
        self.training_patterns = training_patterns
        self.training_targets = training_targets
        self.desired_error = desired_error
        self.auto_learning_speed()
        generacion = 0
        zeros = 0

        while zeros < len(self.training_targets):
            for p in range(len(training_patterns)):
                a1 = tansig(np.array(np.matrix(self.weights1) * training_patterns[p]) + self.threshold1)
                a2 = purelin(np.array(np.matrix(self.weights2) * a1) + self.threshold2)
                error = self.training_targets[p] - a2
                if (abs(error) > self.desired_error).all():
                    self.update_weights(error, training_patterns[p], a1)
                    zeros = 0
                else:
                    zeros += 1

                logging.info("-----------------------------------------------------------")
                logging.info('---                     Iteration %s                    ---',
                             generacion * len(training_patterns) + p + 1)
                logging.info("-----------------------------------------------------------")
                logging.info(' W1 = %s', self.weights1)
                logging.info(' b1 = %s', self.threshold1)
                logging.info(' W2 = %s', self.weights2)
                logging.info(' b2 = %s', self.threshold2)
                logging.info(' Error: %s', error.__str__())
            generacion += 1

        logging.info("-----------------------------------------------------------")
        logging.info('---           The ADALINE network was trained           ---')
        logging.info("-----------------------------------------------------------")
        logging.info(" alpha = %s", self.alpha)
        logging.info(' desired error = %s', self.desired_error)
        logging.info(' W1 = %s', self.weights1)
        logging.info(' b1 = %s', self.threshold1)
        logging.info(' W2 = %s', self.weights2)
        logging.info(' b2 = %s', self.threshold2)
        logging.info(' Error: %s', error.__str__())
        logging.info(' The network was trained in %s generations.', generacion.__str__())

    def get_target(self, input):
        a1 = tansig(np.array(np.matrix(self.weights1) * input) + self.threshold1)
        a2 = purelin(np.array(np.matrix(self.weights2) * a1) + self.threshold2)
        return a2

    def update_weights(self, error, input, a1):
        S2 = -2 * deltalin() * error
        d = deltasig(a1) * np.identity(len(a1))
        S1 = np.array(np.matrix(d) * self.weights2.transpose()) * S2
        self.weights2 -= self.alpha * S2 * a1.transpose()
        self.threshold2 -= self.alpha * S2
        self.weights1 -= self.alpha * S1 * input.transpose()
        self.threshold1 -= self.alpha * S1

    def classify(self, patterns):
        return [self.get_target(p) for p in patterns]

    def eigenvalues(self, md):  # P = |A - lambda*I|
        # Function linalg.eig(a) from Numpy library.
        comp = 0
        e = np.linalg.eig(md)
        # print "eigenvalues = ", e[0]
        e_elem = e[0]
        for i in range(len(e_elem)):
            if (e_elem[i] > comp):
                comp = e_elem[i]
            else:
                comp = comp

        alpha_interval = 1 / comp
        # print "alpha_interval = 0 to", alpha_interval
        return round(uniform(0.0, alpha_interval / 2), 4)

    def compute_R(self, training_patterns, p):
        mult_ma = np.zeros((3, 3))
        mult_mb = np.zeros((3, 3))
        mult_mca = np.zeros((3, 3))
        mult_mc = np.zeros((3, 3))
        # print "p", p
        while p < len(training_patterns):
            trans_p = training_patterns[p]
            i = 0

            while i < 3:
                for j in range(0, 3):
                    elem = trans_p[i] * trans_p[j]
                    mult_ma[i][j] = elem * 0.5  # E = 0.5, from R = E[PP^T]
                i += 1

            p += 1

            if (p < len(training_patterns)):
                trans_p = training_patterns[p]
                i = 0
                while i < 3:
                    for j in range(0, 3):
                        elem = trans_p[i] * trans_p[j]
                        mult_mb[i][j] = elem * 0.5  # E = 0.5, from R = E[PP^T]
                    i += 1
                mult_mct = mult_ma + mult_mb
                mult_mc = mult_mca + mult_mct
                for i in range(3):
                    for j in range(3):
                        mult_mca[i][j] = mult_mc[i][j]
                p += 1

            elif (p == len(training_patterns)):
                mult_mct = mult_ma
                mult_mc = mult_mca + mult_mct
                break
        return mult_mc
