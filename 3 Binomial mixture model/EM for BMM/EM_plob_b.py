import matplotlib.pyplot as plt

infile1 = open('EM_K3.txt','rb')
r1_raw = infile1.readline().split(' ')[:-1]
r1 = []
for r in r1_raw:
    r1.append(float(r))
r1 = r1[1:]
infile2 = open('EM_K9.txt','rb')
r2_raw = infile2.readline().split(' ')[:-1]
r2 = []
for r in r2_raw:
    r2.append(float(r))
r2 = r2[1:]
infile3 = open('EM_K15.txt','rb')
r3_raw = infile3.readline().split(' ')[:-1]
r3 = []
for r in r3_raw:
    r3.append(float(r))
r3 = r3[1:]

x = range(2,51)

ax0 = plt.subplot(111)
# plot yi_hat vs zi as a solid line
ax0.plot(x,r1,'b',label='K=3')
ax0.plot(x,r2,'r',label='K=9')
ax0.plot(x,r3,'g',label='K=15')
ax0.set_xlabel('iteration')
ax0.set_ylabel("L(theta,pi)")
ax0.set_title("log marginal likelihood")
plt.legend()
plt.savefig("HW4_P1_b.png")
plt.show()
