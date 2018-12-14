def read_result(filename):
    infile = open(filename,"rb")
    data = infile.read().split(' ')
    data = data[:-1]
    for i in range(0,len(data)):
        data[i] = float(data[i])
    return data

ret_a = read_result("result.txt")

#print ret_a

# plot ln(P(R,U,V)) for iter 2 to 100
y = ret_a[2:]
x = range(2,101)
import matplotlib.pyplot as plt
fig =plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,y,"r")
#ax.set_xlim([2,101])
ax.set_ylabel("ln(P(R,U,V))")
ax.set_xlabel("iter")
ax.set_title("Iteration 2 to 100")
plt.show()
