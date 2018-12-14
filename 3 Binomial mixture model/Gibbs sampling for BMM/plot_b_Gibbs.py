import matplotlib.pyplot as plt
import numpy as np

def clean(line):
    line = line.split(' ')[:-1]
    ret = []
    for i in line:
        ret.append(int(i))
    return ret

infile = open('p3b.txt','rb')

results = []

line = infile.readline()
while line:
    results.append(clean(line))
    line = infile.readline()

iter = len(results)

def get_six(line):
    line.sort(reverse=True)
    return line[:6]

results_six = []
for result in results:
    results_six.append(get_six(result))


r = np.array(results_six)

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []

print len(r), len(r[0])
for i in range(0,len(r)):
    try:
        x1.append(r[i][0])
    except:
        x1.append(0)
    try:
        x2.append(r[i][1])
    except:
        x2.append(0)
    try:
        x3.append(r[i][2])
    except:
        x3.append(0)
    try:
        x4.append(r[i][3])
    except:
        x4.append(0)
    try:
        x5.append(r[i][4])
    except:
        x5.append(0)
    try:
        x6.append(r[i][5])
    except:
        x6.append(0)


#print len(x1), len(x2), len(x3), len(x4), len(x5), len(x6)
#print x6[303], x6[304]

x = range(1,1001)
ax0 = plt.subplot(111)
#ax0.plot(x,x1,'-.',c='b',alpha=0.7)
#ax0.plot(x,x2,'-.',c='r',alpha=0.7)
#ax0.plot(x,x3,'-.',c='g',alpha=0.7)
#ax0.plot(x,x4,'-.',c='c',alpha=0.7)
#ax0.plot(x,x5,'-.',c='m',alpha=0.7)
#ax0.plot(x,x6,'-.',c='y',alpha=0.7)

ax0.plot(x,x1,c='b',alpha=0.7)
ax0.plot(x,x2,c='r',alpha=0.7)
ax0.plot(x,x3,c='g',alpha=0.7)
ax0.plot(x,x4,c='c',alpha=0.7)
ax0.plot(x,x5,c='m',alpha=0.7)
ax0.plot(x,x6,c='y',alpha=0.7)

ax0.set_xlabel('iteration')
ax0.set_ylabel("number of observations")
ax0.set_title("number of observations of the largest 6 clusters")
#plt.legend()
plt.savefig("HW4_P3_b.png")
plt.show()
