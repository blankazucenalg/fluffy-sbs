import Tkinter, time, logging, sys
import numpy as np
from NeuralNetwork import Net

class Main:
    def __init__(self):
        logging.basicConfig(filename='logger.log', level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler())

        self.training_patterns = []
        self.training_targets = []
        self.desired_error = 0.1
        self.neurons = 3
        logging.info("-----------------------------------------------------------")
        logging.info("         PREDICCION DE SISMOS (BACKPROPAGATION)")
        logging.info("-----------------------------------------------------------")

        # Creating GUI
        top = Tkinter.Tk()
        top.wm_title("ADALINE Example")
        Tkinter.Button(top, text="Entrenar red", command=self.train_the_net).pack()
        Tkinter.Button(top, text="Clasificar patrones", command=self.classify_patterns).pack()
        self.label = Tkinter.StringVar()
        Tkinter.Label(top, textvariable=self.label).pack()
        top.mainloop()

    def train_the_net(self):
        start_time = time.time()
        # Reading the inputs from a file
        logging.info("---               Training patterns                     ---")
        logging.info("-----------------------------------------------------------")
        logging.info(" Pattern\t\tTarget")
        for line in open('inputs.csv', 'r'):
            self.training_patterns.append(np.matrix(line.split('|')[0]))
            self.training_targets.append(np.matrix(line.split('|')[1]))
            logging.info("%s", line.replace('\n', '').replace('|', '\t\t'))

        # Creating & training the network
        input_length = len(self.training_patterns[0])
        target_length = len(self.training_targets[0])
        self.net = Net(input_length, self.neurons, target_length)
        self.net.learn(self.training_patterns, self.training_targets, self.desired_error)
        # If alpha is not set, it would be calculated by the correlation matrix
        # You can also add a weights matrix and a threshold value to the network and add them to the learn() function. Eg
        # net.learn(self.training_patterns, self.training_targets, self.desired_error, alpha=0.4, weights=w, threshold=b)


        '''
        # Now the ADALINE is trained and we can get the results and save them in a file
        f = open('weights.py', 'w')
        f.write('W1 = ' + self.net.weights1.__str__() + '\n')
        f.write('b1 = ' + self.net.threshold1.__str__())
        f.write('W2 = ' + self.net.weights2.__str__() + '\n')
        f.write('b2 = ' + self.net.threshold2.__str__())
        f.close()
        '''

        # Showing weights and threshold in GUi
        self.label.set('Red entrenada')
        # Plotting the inputs and targets
        logging.info("---    Training finished in %s seconds    ---", (time.time() - start_time))
        logging.info("-----------------------------------------------------------")

    def classify_patterns(self):
        start_time = time.time()
        # Once the network is trained, classify some patterns
        self.patterns = []
        self.targets = []
        for line in open('test.csv', 'r'):
            self.patterns.append(np.matrix(line.split('|')[0]))
        self.targets = self.net.classify(self.patterns)

        # Now we can get the results
        logging.info("-----------------------------------------------------------")
        logging.info("---           Pattern classification results           ---")
        logging.info("-----------------------------------------------------------")
        logging.info(" Pattern \t\t Target")
        f = open('results.txt', 'w')
        tar = [[int(round(q,0)) for q in p] for p in self.targets]
        for p, t in zip(self.patterns, tar):
            f.write(p.__str__().replace('[', '').replace(']', '').replace('\n', ';') + "|" + t.__str__().replace('[','').replace(']','').replace(',',';') + "\n")
            logging.info(p.__str__().replace('[', '').replace(']', '').replace('\n', ';') + "\t" + t.__str__().replace('[','').replace(']','').replace(',',';'))
        f.close()
        logging.info("---    Classifying finished in %s seconds    ---", (time.time() - start_time))
        logging.info("-----------------------------------------------------------")
        # Plotting the inputs and targets


Main()
