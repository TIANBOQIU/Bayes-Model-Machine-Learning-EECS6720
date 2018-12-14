import matplotlib.pyplot as plt

infile = open('p3c.txt','rb')
line = infile.readline()

def clean(line):
    line = line.split(' ')[:-1]
    ret = []
    for i in line:
        ret.append(int(i))
    return ret

x1 = clean(line)

#print len(x1)

x = range(1,1001)

ax0 = plt.subplot(111)
ax0.plot(x,x1,c='b',alpha=0.7)

ax0.set_xlabel('iteration')
ax0.set_ylabel("number of clusters")
ax0.set_title("number of clusters that contain data")

#plt.legend()
plt.savefig("HW4_P3_c.png")
plt.show()
