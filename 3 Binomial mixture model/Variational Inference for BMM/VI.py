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

phi = np.ones(N*K).reshape(N,K)
# q(pi) = Dir(alpha1, ..., alphaK)
alpha = np.array(range(1,K+1),dtype='float64')
alpha = alpha / sum(alpha)
# q(theta_j) = Beta(a1_j, b1_j) for j = 1, 2, .. K
#a = np.array(range(1,K+1),dtype='float64')
#a = alpha / sum(alpha)
a = np.ones(K, dtype = 'float64')
#b = np.array(range(1,K+1),dtype='float64')
#b = alpha / sum(alpha)
b = np.ones(K, dtype = 'float64')
alpha_p = np.ones(K, dtype = 'float64') / 10
a_p = 0.5
b_p = 0.5

def E_beta_ln(a1,b1):
    return digamma(a1) - digamma(a1+b1)
def E_beta_ln_m(a1,b1):
    return digamma(b1) - digamma(a1+b1)
def E_Dir_ln_j(alpha, j):
    alpha_j = alpha[j]
    return digamma(alpha[j]) - digamma(sum(alpha))

def phi_i_j_un(i, j):
    global a, b
    global alpha
    global x
    t1 = x[i] * E_beta_ln(a[j],b[j])
    t2 = (20 - x[i]) * E_beta_ln_m(a[j],b[j])
    t3 = E_Dir_ln_j(alpha, j)
    return exp(t1 + t2 + t3)

def phi_i_j(i, j):
    z_i = 0.0
    for k in range(0,K):
        z_i += phi_i_j_un(i, k)
    return phi_i_j_un(i,j) / z_i

def objective_function():
    global a, b, phi, alpha
    sum1 = 0.0
    sum6 = 0.0
    for i in range(0,N):
        xi = x[i]
        for j in range(0,K):
            sum1 += phi[i][j] * (log(comb(20,xi)) + xi * E_beta_ln(a[j],b[j]) + (20-xi) * E_beta_ln_m(a[j],b[j]) + E_Dir_ln_j(alpha, j))
            sum6 += phi[i][j] * log(phi[i][j])
    sum2 = loggamma(sum(alpha_p)) - sum(loggamma(alpha_p))
    sum3 = 0.0
    #sum4 = Entropy_Dir(alpha)
    sum4 = dirichlet.entropy(alpha)
    sum5 = 0.0
    for j in range(0, K):
        sum2 +=  (alpha_p[0]-1) * E_Dir_ln_j(alpha,j)
        sum3 += loggamma(a_p+b_p) - loggamma(a_p) - loggamma(b_p) + (a_p-1)*E_beta_ln(a[j],b[j]) + (b_p-1) * E_beta_ln_m(a[j],b[j])
        sum5 += beta.entropy(a[j],b[j])


    return sum1 + sum2 + sum3 + sum4 + sum5 - sum6

#print objective_function()
def update_phi():
    global phi, a, b, alpha
    for i in range(0, N):
        for j in range(0, K):
            phi[i][j] = phi_i_j_un(i, j)
    for i in range(0, N):
        phi[i] = phi[i] / sum(phi[i])

def update_alpha():
    global alpha, phi, a, b
    phi_t = np.transpose(phi)
    for j in range(0, K):
        alpha[j] = alpha_p[j] + sum(phi_t[j])

def update_a_b():
    global a, b, alpha, phi, x
    phi_t = np.transpose(phi)
    for j in range(0, K):
        sum_phi_xi = sum(phi_t[j] * x)
        a[j] = a_p + sum_phi_xi
        b[j] = b_p + 20 * sum(phi_t[j]) - sum_phi_xi

def update():
    global phi, a, b, alpha
    update_phi()
    update_alpha()
    update_a_b()

#print objective_function()
#update()
#print objective_function()


def print_debug():
    global alpha, a, b, phi, a_p, b_p, alpha_p
    print "alpha", alpha
    print "a", a
    print "b", b
    print "phi[0]",phi[0]

def max_i(p):
    max = 0
    for i in range(1, K):
        if p[i] > p[max]:
            max = i
    return max

print objective_function()
print a_p, b_p, alpha_p
print_debug()

results = []
iter = 1000
def update_n():
    global iter
    for i in range(0, iter):
        update()
        #print_debug()
        r = objective_function()
        print i+1, "th", r
        results.append(r)

update_n()

out = open("VI_K=50.txt",'w')
for r in results:
    out.write('{} '.format(r))
out2 = open("VI_phi_K=50.txt",'w')
for i in range(0,N):
    for j in range(0,K):
        out2.write('{} '.format(phi[i][j]))


count = np.zeros(21*K).reshape(21,K)
for i in range(0,N):
    xi = x[i]
    index = max_i(phi[i])
    count[xi][index] += 1
cluster = []
for i in range(0,21):
    cluster.append(max_i(count[i]))
print cluster
print count


plot_b = True

if plot_b:
    y = results
    x = range(1,iter+1)
    import matplotlib.pyplot as plt
    fig =plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y,"r")
    ax.set_ylabel("L")
    ax.set_xlabel("iter")
    ax.set_title("VI_objective_function")
    fig.savefig('VI_of_K=50.png')
    plt.show()
