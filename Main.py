import Tkinter, time, logging, sys
import numpy as np
from NeuralNetwork import Net
from scipy import *
from matplotlib import pyplot as plt
from RBF import RBF
import Tkinter
import logging
import numpy as np
from copy import copy

__author__ = 'azu'

logging.basicConfig(filename='logger.log', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())


def get_patterns(file):
    p = []
    for line in open(file, 'r'):
        p.append(np.matrix(line.split('|')[0]))
    return np.array(p)


def get_targets(file):
    p = []
    for line in open(file, 'r'):
        p.append(np.matrix(line.split('|')[1]))
    return np.array(p)


# Get data
w = np.matrix([[0.7071, -0.7071], [0.7071, 0.7071], [-1, 0]])
patterns = get_patterns('inputs.csv')
targets = get_targets('inputs.csv')
test = get_patterns('test.csv')
neurons = 3
epochs = 300

'''
#Creating GUI
top = Tkinter.Tk()
top.wm_title("Red neuronal competitiva")
Tkinter.Button(top, text="Mostrar parametros iniciales", command=plot_init_data).pack()
Tkinter.Button(top, text="Mostrar parametros finales", command=plot_final_data).pack()
label = Tkinter.StringVar()
var = "Numero de neuronas: "+neurons.__str__()+"\nEpocas: "+epochs.__str__()
label.set(var)
Tkinter.Label(top, textvariable=label).pack()
top.mainloop()

logging.info("Parametros iniciales: %s",w)
logging.info("Parametros finales: %s",net.w)
'''

# ----- 1D Example ------------------------------------------------
n = 100

x = mgrid[-1:1:complex(0,n)].reshape(n, 1)
# set y and add random noise
y = sin(3*(x+0.5)**3 - 1)
# y += random.normal(0, 0.1, y.shape)

# rbf regression
rbf = RBF(1, 10, 1)
rbf.train(patterns, targets)
z = rbf.test(test)

# plot original data
plt.figure(figsize=(12, 8))
plt.plot(x, y, 'k-')

# plot learned model
plt.plot(x, z, 'r-', linewidth=2)

# plot rbfs
plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')

for c in rbf.centers:
    # RF prediction lines
    cx = arange(c-0.7, c+0.7, 0.01)
    cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
    plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

plt.xlim(-1.2, 1.2)
plt.show()
