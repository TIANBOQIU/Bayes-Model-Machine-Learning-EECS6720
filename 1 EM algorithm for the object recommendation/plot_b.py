def read_result(filename):
    infile = open(filename,"rb")
    data = infile.read().split(' ')
    data = data[:-1]
    for i in range(0,len(data)):
        data[i] = float(data[i])
    return data

# we use different seed to generate randoms
# run the model separately and get these five results
ret_1 = read_result("result2.txt")
ret_2 = read_result("result22.txt")
ret_3 = read_result("result111.txt")
ret_4 = read_result("result333.txt")
ret_5 = read_result("result456.txt")
#print ret_a

# plot ln(P(R,U,V)) for iter 20 to 100
y1 = ret_1[20:]
y2 = ret_2[20:]
y3 = ret_3[20:]
y4 = ret_4[20:]
y5 = ret_5[20:]
x = range(20,101)
import matplotlib.pyplot as plt
fig =plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,y1,"r",x,y2,"g",x,y3,"b",x,y4,"y",x,y5,"m")
#ax.set_xlim([2,101])
ax.set_ylabel("ln(P(R,U,V))")
ax.set_xlabel("iter")
ax.set_title("5 different random starting points")
plt.show()
