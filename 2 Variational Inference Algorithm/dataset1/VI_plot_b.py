import matplotlib.pyplot as plt

def read(filename):
    infile = open(filename,'rb')
    a1 = infile.readline()
    b1 = infile.readline()
    a1 = a1.split(' ')[:-1]
    b1 = b1.split(' ')[:-1]
    return a1, b1

def E_q_gamma(a, b):
    return float(a) / b


a1, b1 = read('alpha.txt')
print a1
print b1
print len(a1), len(b1)
d = len(a1)
x = range(1,d+1)
y = []
for i in range(0,d):
    y.append(1.0 / E_q_gamma(float(a1[i]),float(b1[i])))


fig =plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x,y,s=5,alpha=0.8)
ax.set_ylabel('1/E_q[alpha_k]')
ax.set_xlabel("k")
plt.show()
