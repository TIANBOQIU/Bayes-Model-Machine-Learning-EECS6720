import csv
from random import random
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import slogdet
from scipy.stats import norm
from scipy.special import digamma
from scipy.special import gamma
from scipy.special import loggamma
from scipy import log
from scipy import e
from scipy import pi

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
# N = 100
N = len(x)
# d = 101
d = len(x[0])
# priors' settings
a0 = 1e-16
b0 = 1e-16
e0 = 1
f0 = 1



# initialization
e1 = 10
f1 = 5
# for alpha_i , i = 1 to d
a1 = np.ones(d)
b1 = np.ones(d)

sigma1 = np.identity(d)
mu1 = np.ones(d)
#print sigma1
#print len(sigma1), len(sigma1[0])

#def update_e1_f1():
def E_q_gamma_ln(a, b):
    return digamma(a) - log(b)

def E_q_gamma(a, b):
    return float(a) / b

def Entropy_gamma(a, b):
    # use loggamma to prevent overflow
    return a - log(b) + loggamma(a) + (1-a) * digamma(a)

def E_q_norm_SE(yi, xi, mu1, sigma1):
    return (yi - xi.dot(mu1)) ** 2 + xi.dot(sigma1).dot(xi)

def E_q_norm_WTW(mu1, sigma1):
    return np.trace(sigma1) + mu1.dot(mu1)

def Entropy_norm(sigma1):
    sign, logdet = slogdet(sigma1)
    return 0.5 * (d*log(2*pi*e) + logdet)

#x0 = x[0]
#y0 = y[0]
#print E_q_norm_SE(y0,x0,mu1,sigma1)

def VI_objective(a1,b1,e1,f1,sigma1,mu1):
    sum_1 = 0.0
    for i in range(0, N):
        xi = x[i]
        yi = y[i]
        sum_1 += 0.5 * E_q_gamma_ln(e1,f1) - 0.5 * E_q_gamma(e1,f1)*E_q_norm_SE(yi,xi,mu1,sigma1) - 0.5 * log(2*pi)
    sum_2 = 0.0
    sum_3 = 0.0
    sum_6 = 0.0
    p = 0.0
    for i in range(0, d):
        ai = a1[i]
        bi = b1[i]
        sum_2 += E_q_gamma_ln(ai,bi)
        p += E_q_gamma(ai,bi) * (sigma1[i][i] + mu1[i]**2)

        sum_3 += log(b0**a0 / gamma(a0)) + (a0 - 1) * E_q_gamma_ln(ai,bi) - b0 * E_q_gamma(ai,bi)
        sum_6 += Entropy_gamma(ai,bi)
    sum_2 = 0.5 * sum_2 -0.5*d*log(2*pi) - 0.5 * p
    sum_4 = log(f0**e0 / gamma(e0)) + (e0 - 1) * E_q_gamma_ln(e1,f1) - f0 * E_q_gamma(e1,f1)
    sum_5 = Entropy_norm(sigma1) + Entropy_gamma(e1,f1) + sum_6
    #sum_5 = Entropy_gamma(e1,f1) + sum_6
    #print sum_1, sum_2, sum_3, sum_4, sum_5, sum_6
    #print Entropy_norm(sigma1)
    #print Entropy_gamma(e1,f1)
    #print "------"
    return sum_1 + sum_2 + sum_3 + sum_4 + sum_5
    #print sum_1, sum_2, sum_3, sum_4, sum_5, sum_6
    #print Entropy_norm(sigma1), Entropy_gamma(e1,f1)

#print VI_objective(a1,b1,e1,f1,sigma1,mu1)



def VI_update():
    global e1
    global f1
    global a1
    global b1
    global sigma1
    global mu1
    global e0
    global f0
    global a0
    global b0
    e1 = 0.5 * N + e0
    p1 = 0.0
    for i in range(0,N):
        xi = x[i]
        yi = y[i]
        p1 += E_q_norm_SE(yi,xi,mu1,sigma1)
    f1 = f0 + 0.5 * p1

    for i in range(0,d):
        a1[i] = a0 + 0.5
        b1[i] = b0 + 0.5 * (sigma1[i][i] + mu1[i]**2)
        #print update_b1_i(i),E_q_norm_WTW(mu1,sigma1)

    p2 = np.identity(d)
    for i in range(0,d):
        p2[i][i] = E_q_gamma(a1[i],b1[i])

    p3 = np.zeros((d,d))
    p4 = np.zeros(d)
    for i in range(0,N):
        p3 += np.outer(x[i],x[i])
        p4 += y[i] * x[i]
    #
    #print p2, "p3",E_q_gamma(e1,f1), det(inv(E_q_gamma(e1,f1) * p3))
    sigma1 = inv(p2 + E_q_gamma(e1,f1) * p3)
    mu1 = sigma1.dot(E_q_gamma(e1,f1) * p4)


results = []

#print Entropy_gamma(e1,f1)
#print e1, f1
for i in range(0,500):
    #print VI_objective(a1,b1,e1,f1,sigma1,mu1)
    #print sigma1
    VI_update()
    #print e1, f1
    r = VI_objective(a1,b1,e1,f1,sigma1,mu1)
    results.append(r)
    #print det(sigma1)
    #print i,'th',det(sigma1), log(det(sigma1))
    print r

# save e1,f1,a1,b1,sigma1,mu1
## P2 (b)
out_alpha = open("alpha.txt","w")
for ai in a1:
    out_alpha.write("{} ".format(ai))
out_alpha.write('\n')
for bi in b1:
    out_alpha.write("{} ".format(bi))

## P2(c)
print "1/E_q[lambda]=", 1.0 / E_q_gamma(e1,f1)
out = open("p2c.txt",'w')
out.write('{}'.format(1.0 / E_q_gamma(e1,f1)))

out = open("mu1.txt",'w')
for i in mu1:
    out.write('{} '.format(i))

# plots for dataset1
plot_b = True

if plot_b:
    y = results
    x = range(1,501)
    import matplotlib.pyplot as plt
    fig =plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y,"r")
    ax.set_ylabel("L(a1,b1,e1,f1,mu1,sigma1)")
    ax.set_xlabel("iter")
    ax.set_title("VI objective function")
    plt.show()
