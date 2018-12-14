# maybe some thing wrong with initialization
# sample ci should be two case, maybe something wrong
import csv
from scipy.stats import binom
import numpy as np
from scipy.special import digamma
from scipy.special import gamma
from scipy.special import loggamma
from scipy import log
from scipy import e
from scipy import pi
from scipy import exp
from scipy.misc import comb
from scipy.stats import dirichlet
from scipy.stats import beta
from random import randint
from copy import copy, deepcopy
from scipy import stats

def read_csv(filename):
    infile = open(filename,"rb")
    dataset = csv.reader(infile)
    x = []
    for row in dataset:
        x.append(int(row[0]))
    return x

x = read_csv("x.csv")
N = len(x)

K = 30
alpha = 0.75
a = 0.5
b = 0.5


class Cluster:
    def __init__(self,index):
        self.index = index
        self.member = []
        self.a1 = 0.5
        self.b1 = 0.5
    def size(self):
        return len(self.member)



clusters = {}
for i in range(0, K):
    clusters[i] = Cluster(i)
c = []
for i in range(0, N):
    ci = randint(0, K-1)
    clusters[ci].member.append(x[i])
    c.append(ci)

def print_cluster(clusters):
    for j in clusters:
        print "index", j
        print clusters[j].member
#print_cluster(clusters)

p_new = np.zeros(21)
for i in range(0, 21):
    p_new[i] = alpha*gamma(a+i)*gamma(b+20-i)/(alpha+N-1)/gamma(a+b+20)
#print p_new

def sample_c_i(i):
    global c, clusters
    #print_cluster(clusters)
    xi = x[i]
    ci = c[i]
    phi_i_j = {}
    clusters_cp = deepcopy(clusters)
    clusters_cp[ci].member.remove(xi)

    for j in clusters_cp:
        n_j_i = clusters_cp[j].size()
        if n_j_i > 0:
            theta_j = beta.rvs(clusters_cp[j].a1,clusters_cp[j].b1)
            phi_i_j[j] = binom.pmf(xi,20,theta_j) * n_j_i / (alpha + N - 1)

    j_new = max(clusters) + 1
    phi_i_j[j_new] = p_new[xi]
    # generate j1
    c_phi = phi_i_j.keys()
    p_phi = phi_i_j.values() / sum(phi_i_j.values())
    #j1 = stats.rv_discrete(name='custm',values=(c_phi,p_phi))
    j1 = np.random.choice(c_phi,p = p_phi)
    #j1 = max(phi_i_j, key = phi_i_j.get)
    #print "debug:x",i,"ci",ci, "j1",j1
    if j1 == j_new:
        c[i] = j1
        clusters[j1] = Cluster(j1)
        clusters[j1].member.append(xi)
        clusters[ci].member.remove(xi)
        if clusters[ci].size() == 0:
            del clusters[ci]
    else:
        c[i] = j1
        clusters[j1].member.append(xi)
        #print xi, ci,  clusters[ci].member
        #print "ci",ci,"member",clusters[ci].member,"xi",xi
        clusters[ci].member.remove(xi)
        if clusters[ci].size() == 0:
            del clusters[ci]

def sample_c():
    for i in range(0, N):
        sample_c_i(i)

def sample_theta():
    global clusters
    for j in clusters:
        if clusters[j].size() > 0:
            sum_j_member = sum(clusters[j].member)
            clusters[j].a1 = a + sum_j_member
            clusters[j].b1 = b + 20 * len(clusters[j].member) - sum_j_member

iter = 1000
out = open('p3b.txt','w')
out2 = open('p3c.txt','w')
def update_n():
    for i in range(0, iter):
        sample_c()
        sample_theta()
        print ""
        print i+1, "th"
        print clusters.keys()
        print "len",len(clusters)
        for j in clusters:
            size_j = clusters[j].size()
            out.write('{} '.format(size_j))
        out.write('\n')
        out2.write('{} '.format(len(clusters)))




print "initial"
print clusters.keys()
print "len",len(clusters)
#print_cluster(clusters)
update_n()
#print_cluster(clusters)
#for i in range(0, N):
#    xi = x[i]
#    ci = c[i]
#    clusters[ci].member.remove(xi)
#print_cluster(clusters)
