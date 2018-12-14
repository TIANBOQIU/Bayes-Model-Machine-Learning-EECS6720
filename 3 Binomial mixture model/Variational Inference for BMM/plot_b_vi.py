import matplotlib.pyplot as plt

infile1 = open('VI_K=3.txt','rb')
r1_raw = infile1.readline().split(' ')[:-1]
r1 = []
for r in r1_raw:
    r1.append(float(r))
r1 = r1[1:]
infile2 = open('VI_K=15.txt','rb')
r2_raw = infile2.readline().split(' ')[:-1]
r2 = []
for r in r2_raw:
    r2.append(float(r))
r2 = r2[1:]
infile3 = open('VI_K=50.txt','rb')
r3_raw = infile3.readline().split(' ')[:-1]
r3 = []
for r in r3_raw:
    r3.append(float(r))
r3 = r3[1:]

x = range(2,1001)

ax0 = plt.subplot(111)
# plot yi_hat vs zi as a solid line



ax0.plot(x,r1,'b',label='K=3')
ax0.plot(x,r2,'r',label='K=15')
ax0.plot(x,r3,'g',label='K=50')
ax0.set_xlabel('iteration')
ax0.set_ylabel("L")
ax0.set_title("VI objective function")
plt.legend()
plt.savefig("HW4_P2_b.png")
plt.show()
