import numpy as np
import csv
import scipy
from scipy.stats import norm
from numpy.linalg import inv
from math import *

users = {}
movies = {}

def read_data_csv(filename):
    infile = open(filename,"rb")
    entries = csv.reader(infile)
    for e in entries:
        user_id = int(e[0])
        movie_id = int(e[1])
        rating = int(e[2])
        if not users.has_key(user_id):
            users[user_id] = ({movie_id:rating})
        else:
            users[user_id][movie_id] = rating

        if not movies.has_key(movie_id):
            movies[movie_id] = ({user_id:rating})
        else:
            movies[movie_id][user_id] = rating


def test_1():
    read_data_csv("ratings.csv")
    print len(users), len(movies)
def test_2():
    #read_data_csv("ratings.csv")
    print len(users), len(movies)
    print users[196]
    print movies[1680]
# test_1() -> len(users):943 len(movies):1676
read_data_csv("ratings.csv")
#test_2()
N = 943
M = 1682 # six movies's ratings never show in any user
d = 5
U = np.array(norm.rvs(0,0.1,N*d,random_state=1))
U = U.reshape(N,d)
V = np.array(norm.rvs(0,0.1,M*d,random_state=2))
V = V.reshape(M,d)
#print U[942],len(U)
#print V[1681], len(V)

def E_phi(user_id, movie_id):
    rating = users[user_id][movie_id]
    utv = U[user_id-1].dot(V[movie_id-1])
    if rating==1:
        return utv + norm.pdf(-utv,0,1) / (1-norm.cdf(-utv,0,1))
    elif rating == -1:
        return utv + (-norm.pdf(-utv,0,1) / norm.cdf(-utv,0,1))

#print E_phi(196,242)
#sum_uut = np.zeros((d,d))
#sum_ueq = np.zeros(d)
def update_V():
    #global sum_uut
    #global sum_ueq
    for j in range(0,M):
        movie_id = j+1
        if movies.has_key(movie_id):
            sum_uut = np.zeros((d,d))
            sum_ueq = np.zeros(d)
            for user_id in movies[movie_id]:
                u = U[user_id-1]
                sum_uut += np.outer(u,u)
                sum_ueq += u * E_phi(user_id,movie_id)
            V[j] = inv(np.identity(d)+sum_uut).dot(sum_ueq)

def update_U():
    #global sum_vvt
    #global sum_veq
    for i in range(0,N):
        user_id = i+1
        if users.has_key(user_id):
            sum_vvt = np.zeros((d,d))
            sum_veq = np.zeros(d)
            for movie_id in users[user_id]:
                v = V[movie_id-1]
                sum_vvt += np.outer(v,v)
                sum_veq += v * E_phi(user_id,movie_id)
            U[i] = inv(np.identity(d)+sum_vvt).dot(sum_veq)

def log_joint():
    sum_utu = 0.0
    for u in U:
        utu = u.dot(u)
        sum_utu += utu
    sum_vtv = 0.0
    for v in V:
        vtv = v.dot(v)
        sum_vtv += vtv
    sum = 0.0
    for user_id in users:
        u = U[user_id-1]
        for movie_id in users[user_id]:
            v = V[movie_id-1]
            rating = users[user_id][movie_id]
            utv = u.dot(v)
            if rating == 1:
                sum += log(norm.cdf(utv,0,1))
            elif rating == -1:
                sum += log(1-norm.cdf(utv,0,1))
    return sum_utu + sum_vtv + sum - ((M+N)*d/2.0)*log(2*pi)

def update():
    update_V()
    update_U()

def train():
    result = []
    iter = 100
    ret0 = log_joint()
    #result.append(ret0)
    print "initial", "ln(P(R,U,V))={}".format(ret0)
    for i in range(0,iter):
        update()
        ret =  log_joint()
        result.append(ret)
        print "iter", i+1, "ln(P(R,U,V))={}".format(ret)
        #update()

    out_result = open("result.txt","w")
    out_result.write("{} ".format(ret0))
    for ret in result:
        out_result.write("{} ".format(ret))

    out_U = open("U.txt","w")
    for row in U:
        for val in row:
            out_U.write("{} ".format(val))
    out_V = open("V.txt","w")
    for row in V:
        for val in row:
            out_V.write("{} ".format(val))


# load updated U and V from latest iter
def load():
    global U
    global V
    in_u = open("U.txt","rb")
    in_v = open("V.txt","rb")
    U = np.fromfile(in_u,dtype="float64",sep=' ')
    V = np.fromfile(in_v,dtype="float64",sep=' ')
    U = U.reshape(N,d)
    V = V.reshape(M,d)

#load()
#print log_joint()

def read_data_csv(filename,users,movies):
    infile = open(filename,"rb")
    entries = csv.reader(infile)
    for e in entries:
        user_id = int(e[0])
        movie_id = int(e[1])
        rating = int(e[2])
        if not users.has_key(user_id):
            users[user_id] = ({movie_id:rating})
        else:
            users[user_id][movie_id] = rating

        if not movies.has_key(movie_id):
            movies[movie_id] = ({user_id:rating})
        else:
            movies[movie_id][user_id] = rating


def test():
    users_test = {}
    movies_test = {}
    read_data_csv("ratings_test.csv",users_test,movies_test)
    #print users_test[617]
    #print movies_test[590]
    likes = 0
    dislikes = 0
    # right likes and dislikes
    predict_likes = 0
    predict_dislikes = 0
    for user_id in users_test:
        u = U[user_id-1]
        for movie_id in users_test[user_id]:
            v = V[movie_id-1]
            rating = users_test[user_id][movie_id]
            utv =u.dot(v)
            p_like = norm.cdf(utv)
            predict = 0
            if p_like > 0.5:
                predict = 1

            else:
                predict = -1
            if rating == 1:
                likes += 1
            elif rating == -1:
                dislikes += 1
            if rating == 1 and predict == 1:
                predict_likes +=1
            if rating == -1 and predict == -1:
                predict_dislikes +=1

    #print "like\tdislikes"
    print "-----Confusion Matrix-----"
    print "actual\t\tlikes\tdislikes"
    print "predict"
    print "likes\t\t{}\t{}".format(predict_likes,dislikes-predict_dislikes)
    print "dislikes\t{}\t{}".format(likes-predict_likes,predict_dislikes)


load()
test()
