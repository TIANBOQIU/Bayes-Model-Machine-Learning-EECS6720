import csv
from scipy.stats import binom
from numpy import log
import numpy as np

def read_csv(filename):
    infile = open(filename,"rb")
    dataset = csv.reader(infile)
    x = []
    for row in dataset:
        x.append(int(row[0]))
    return x

x = read_csv("x.csv")
N = len(x)
K = 15
#pi = np.array([0.2,0.3,0.5])
pi = np.array(range(1,K+1), dtype='float64')*2
pi = pi / sum(pi)
#theta = np.ones(K) / 2
#theta = np.array([0.7,0.2,0.1])
theta = np.array(range(1,K+1), dtype='float64')
theta = theta / sum(theta)
phi = np.ones(N*K).reshape(N,K)

def phi_i_j_un(i, j):
    global theta, pi
    return binom.pmf(x[i], 20, theta[j]) * pi[j]

def phi_i_j(i, j):
    z_i = 0.0
    for k in range(0,K):
        z_i += phi_i_j_un(i, k)
    return phi_i_j_un(i,j) / z_i

def update_phi():
    global pi, theta, phi
    for i in range(0,N):
        for j in range(0,K):
            phi[i][j] = phi_i_j(i, j)

def update_theta():
    global phi, theta
    phi_t = np.transpose(phi)
    for j in range(0, K):
        n_j = sum(phi_t[j])
        t = 0.0
        for i in range(0,N):
            t += x[i] * phi[i][j]
        theta[j] = t / n_j / 20

def update_pi():
    global phi, pi
    phi_t = np.transpose(phi)
    for j in range(0,K):
        n_j = sum(phi_t[j])
        pi[j] = n_j / N

def update():
    global phi, pi, theta
    update_phi()
    update_theta()
    update_pi()

def objective_function():
    global phi, theta, pi
    t1 = 0.0
    t2 = 0.0
    for i in range(0, N):
        for j in range(0, K):
            t1 += phi[i][j] * (log(binom.pmf(x[i],20,theta[j])) + log(pi[j]))
            t2 += phi[i][j] * log(phi[i][j])
    return t1 - t2

results = []

def update_n():
    global phi, pi, theta
    for i in range(50):
        update()
        #print theta
        #print pi
        r = objective_function()
        results.append(r)
        print i+1, "th", r



print objective_function()
update_n()

out = open("EM_l.txt",'w')
for r in results:
    out.write('{} '.format(r))
out2 = open("EM_phi.txt",'w')
for i in range(0,N):
    for j in range(0,K):
        out2.write('{} '.format(phi[i][j]))
print phi

plot_b = True

if plot_b:
    y = results
    x = range(1,51)
    import matplotlib.pyplot as plt
    fig =plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y,"r")
    ax.set_ylabel("L(theta,pi)")
    ax.set_xlabel("iter")
    ax.set_title("log marginal likelihood")
    fig.savefig('EM_of_K=15.png')
    plt.show()
