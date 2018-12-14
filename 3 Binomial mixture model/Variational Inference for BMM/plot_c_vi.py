import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def read_csv(filename):
    infile = open(filename,"rb")
    dataset = csv.reader(infile)
    x = []
    for row in dataset:
        x.append(int(row[0]))
    return x

x = read_csv("x.csv")
N = len(x)

K = 50
infile = open('VI_phi_K=50.txt','r')
phi_raw = infile.readline()
phi = np.fromstring(phi_raw, dtype='float64', sep=' ')
phi = phi.reshape(N,K)
print phi
print sum(phi[0])

def max_i(p):
    max = 0
    for i in range(1, K):
        if p[i] > p[max]:
            max = i
    return max

#print max_i(phi[1997])

count = np.zeros(21*K).reshape(21,K)
for i in range(0,N):
    xi = x[i]
    index = max_i(phi[i])
    count[xi][index] += 1

cluster = []
for i in range(0,21):
    cluster.append(max_i(count[i]))

xx = range(0,21)
print cluster
ax = plt.figure().gca()
ax.plot(xx, cluster, 'o')
ax.set_xlabel('x')
ax.set_ylabel('cluster index')
ax.set_title('K=50')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig("HW4_P2_c_K=50.png")
plt.show()
