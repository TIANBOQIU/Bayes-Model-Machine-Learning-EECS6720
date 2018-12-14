import csv
from random import random
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from scipy.stats import norm
from scipy.special import digamma
from scipy.special import gamma
from scipy import log
from scipy import e
from scipy import pi
import matplotlib.pyplot as plt
from numpy import sinc

# load files
# -read x
def read_x_csv(filename):
    infile = open(filename,"rb")
    dataset = csv.reader(infile)
    x = []
    for row in dataset:
        xi = []
        for i in row:
            xi.append(float(i))
        x.append(xi)
    return x
# -read y or z
def read_y_csv(filename):
    infile = open(filename,"rb")
    dataset = csv.reader(infile)
    y = []
    for row in dataset:
        y.append(float(row[0]))
    return y

# -read X_set1 y_set1 z_set1
x = read_x_csv("X_set1.csv")
x = np.array(x)
y = read_y_csv("y_set1.csv")
y = np.array(y)
z = read_y_csv("z_set1.csv")
z = np.array(z)

N = len(x)
# d = 101
d = len(x[0])


infile = open('mu1.txt','rb')
mu1_raw = infile.readline().split(' ')[:-1]
mu1 = np.ones(d)
for i in range(0,d):
    mu1[i] = float(mu1_raw[i])

y1 = []
for i in range(0,N):
    xi = x[i]
    y1.append(xi.dot(mu1))

ax0 = plt.subplot(111)
# plot yi_hat vs zi as a solid line
ax0.plot(z,y1,'b',label='yi_hat')
# plot yi vs zi
ax0.scatter(z,y,c='g',s=5,alpha=0.6,label='yi')
# plot 10*sinc(zi) vs zi
ax0.plot(z,10*sinc(z),'r',label='10*sinc(zi)')
ax0.set_xlabel('zi')
plt.legend()
plt.show()
