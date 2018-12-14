# a naive bayes classifier to classify spam / ham (non-spam)
from __future__ import division
import csv
from math import *



# read train test data and label
infile = open("X_train.csv", "rb")
train_data_csv = csv.reader(infile)
infile2 = open("X_test.csv", "rb")
test_data_csv = csv.reader(infile2)
infile3 = open("label_train.csv", "rb")
train_label_csv = csv.reader(infile3)
infile4 = open("label_test.csv", "rb")
test_label_csv = csv.reader(infile4)

output = open('filterResult.txt', 'w')
#for row in train_data_csv:
#    print row
# a list of list (54d)
train_data = []
test_data = []
# a list of ints
train_label = []
test_label = []
for row in train_data_csv:
    train_data.append(row)

for row in test_data_csv:
    test_data.append(row)

for row in train_label_csv:
    train_label.append(int(row[0]))

for row in test_label_csv:
    test_label.append(int(row[0]))

#print train_data[0][53]
#print test_data[0]
#print train_label[1630], train_label[1631]
#print test_label[181], test_label[182]

# now split into train_spam train_ham
print len(train_data), "dimension: ", len(train_data[0])
train_data_demension = len(train_data[0])
train_data_N = len(train_data)
test_data_N = len(test_data)
print len(train_label)

train_spam = []
train_spam_N = 0
train_ham = []
train_ham_N = 0
for i in range (0, len(train_label)):
    if train_label[i] == 1:
        train_spam.append(train_data[i])
        train_spam_N += 1
    else:
        train_ham.append(train_data[i])
        train_ham_N += 1

#print "train_spam: ", len(train_spam), "size: ",train_spam_N
#print "train_ham: ", len(train_ham), "size: ", train_ham_N

# compute sigmas_spam sigmas_ham
sigmas_spam = []
sigmas_ham = []
for i in range(0, train_data_demension):
    sigmas_spam.append(0)
    sigmas_ham.append(0)
#print len(sigmas_spam), len(sigmas_ham)
for i in range(0, len(train_spam)):
    for j in range(0, train_data_demension):
        sigmas_spam[j] += int(train_spam[i][j])
        sigmas_ham[j] += int(train_ham[i][j])

print sigmas_spam
print sigmas_ham

#print factorial(3)

# compute prior p(y* = spam|y) p(y* = ham|y)
p_prior_spam = (1 + train_spam_N) / (train_data_N + 2)
p_prior_ham = (1 + train_ham_N) / (train_data_N + 2)

#print "p_prior_spam: ", p_prior_spam, "p_prior_ham: ", p_prior_ham


#p_likelihoods_spam = []
#for p(x*|spam)
#test_data1 = test_data[0]

label_test_spam = 0
label_test_ham = 0
for label in test_label:
    if label == 1:
        label_test_spam += 1
    elif label == 0:
        label_test_ham += 1

spam_as_spam = 0
ham_as_ham = 0

# wrong classification indexes
true_spam_but_ham = []
true_ham_but_spam = []
ambiguious_predictions = []

# for index in range(0, test_data_N)
for index in range(0, test_data_N):
    p_likelihoods_spam = []
    test_data1 = test_data[index]
    right_class = test_label[index]
    predict_class = 0
    for i in range(0, train_data_demension):
        k = int(test_data1[i])
        sigma = sigmas_spam[i]

        #print "sigma", sigma
        # due to overflow problem
        #p1 = factorial(k + sigma) / (factorial(k) * factorial (sigma))
        #p1 = 1L
        log_p1 = 0
        #for i in range(1, k+1):
        for t in range(1, k+1):  #if k == 0
            #p1 *= (t + sigma) / t
            #log_p1 *= log((1 + sigma/t), 10)
            log_p1 += log((1 + sigma/t), 10)
            #print t, "th", log_p1
        #p1 /= factorial(k)
        #print i, "the p1",p1
        #print "log_p1: ", log_p1

        p2 = ((train_spam_N + 1)/(train_spam_N + 2)) ** (sigma + 1)
        #print "p2: ", p2
        #p3 = (1.0 / (sigma+2)) ** k
        # p3 = (sigma+2) ** k
        p3 = (train_spam_N + 2) ** k
        #print "p3: ", p3
        # in case of overflow e.g. test_data[96] k = 101 sigma = 3719 will cause p1 / p3
        #(which use L->float)can not convert into float
        #p = p1  * p2 / p3
        #if p1 > 1e+200:
        p = (10 ** (log_p1 - log(p3, 10)) ) * p2
        #else:
        #    p = p1  * p2 / p3
        # print i,"th:", p1, p2, p3, p
        p_likelihoods_spam.append(p)

    #print p_likelihoods_spam

    #print p_likelihoods_spam
    def is_less_than_1(p):
        for val in p:
            if val >= 1:
                return False
        return True
    #print "checking: ", is_less_than_1(p_likelihoods_spam)
    # compute product of 54 dimension
    p_likelihood_spam = 1.0
    for val in p_likelihoods_spam:
        p_likelihood_spam *= val
    #print "p_likelihood_spam (product)", p_likelihood_spam


    p_likelihoods_ham = []
    #for p(x*|ham)
    #test_data1 = test_data[0]
    for i in range(0, train_data_demension):
        k = int(test_data1[i])
        sigma = sigmas_ham[i]
        #p1 = factorial(k + sigma) / (factorial(k) * factorial (sigma))
        log_p1 = 0
        for t in range(1, k+1):
            #p1 *= (t + sigma) / t
            #p1 *= 1 +sigma/t case 358 overflow, get a inf
            log_p1 += log((1 +sigma/t), 10)
        #p1 /= factorial(k)
        p2 = ((train_ham_N + 1)/(train_ham_N + 2)) ** (sigma + 1)
        #p3 = (1.0 / (sigma+2)) ** k
        #p3 = (sigma+2) ** k
        p3 = (train_ham_N + 2) ** k
        #p =p1 * p2 / p3
        #if p1 > 1e+50:
        #print "p1",p1,"p2",p2,"p3",p3
        #p = (10 ** (log(p1, 10) - log(p3, 10)) ) * p2
        p = (10 ** (log_p1 - log(p3, 10)) ) * p2
        #else:
        #    p =p1 * p2 / p3
        #print i,"th:", p1, p2, p3, p
        p_likelihoods_ham.append(p)

    #print "p_likelihoods_spam:", p_likelihoods_spam

    #print "p_likelihoods_ham", p_likelihoods_ham
    #print "checking: ", is_less_than_1(p_likelihoods_ham)
    # compute product of 54 dimension

    # debug 922
    p_likelihood_ham = 1.0
    for val in p_likelihoods_ham:
        p_likelihood_ham *= val
    #log_p_likelihood_ham = 0
    #for val in p_likelihoods_ham:
    #    log_p_likelihood_ham += log(val, 10)
    #p_likelihood_ham = 10 ** log_p_likelihood_ham
    #print "p_likelihood_ham (product)", p_likelihood_ham


    #print "p_prior_spam: ", p_prior_spam, "p_prior_ham: ", p_prior_ham

    if p_likelihood_spam == 0:
        p_predict_is_spam = 0
    else:
        p_predict_is_spam = p_likelihood_spam * p_prior_spam / (p_likelihood_spam * p_prior_spam + p_likelihood_ham*p_prior_ham)
    #print "p_likelihood_spam", p_likelihood_spam
    #p11 = p_likelihood_ham / p_likelihood_spam
    #p12 = p_prior_ham / p_prior_spam
    #p_predict_is_spam = 1.0 / (1.0 + p11 * p12)

    if p_likelihood_ham == 0:
        p_predict_is_ham = 0
    else:
        p_predict_is_ham = p_likelihood_ham * p_prior_ham / (p_likelihood_spam * p_prior_spam + p_likelihood_ham*p_prior_ham)
    #p21 = p_likelihood_spam / p_likelihood_ham
    #p22 = p_prior_spam / p_prior_ham
    #p_predict_is_ham = 1.0 / (1.0 + p21 * p22)
    print p_likelihood_spam, p_prior_spam
    print p_likelihood_ham, p_prior_ham
    print index,"th p_predict_is_spam: ", p_predict_is_spam
    print index,"th p_predict_is_ham: ", p_predict_is_ham
    output.write('{},"th p_predict_is_spam: ", {}\n'.format(index, p_predict_is_spam))
    output.write('{},"th p_predict_is_ham: ", {}\n'.format(index, p_predict_is_ham))




    if p_predict_is_spam > p_predict_is_ham:
        predict_class = 1
    else:
        predict_class = 0

    if right_class == 1 and predict_class == 1:
        spam_as_spam += 1
    #else:
    elif right_class == 1 and predict_class == 0:
        true_spam_but_ham.append(index)
    if right_class == 0 and predict_class == 0:
        ham_as_ham += 1
    #else:
    elif right_class == 0 and predict_class == 1:
        true_ham_but_spam.append(index)

    if abs(p_predict_is_spam - p_predict_is_ham) < 0.1:
        ambiguious_predictions.append(index)

print "true_spam_but_ham", true_spam_but_ham, "len", len(true_spam_but_ham)
print "true_ham_but_spam", true_ham_but_spam, "len", len(true_ham_but_spam)
print "ambiguious_predictions", ambiguious_predictions


print "-----RESULT------"
print "spam_as_spam", spam_as_spam, "in total", label_test_spam, "precision", spam_as_spam / label_test_spam
print "ham_as_ham", ham_as_ham,"in total", label_test_ham, "precision", ham_as_ham / label_test_ham
print "-----Confusion Matrix-----"
print "actual\t\tspam\t\tnon-spam"
print "predict\t"
print "spam\t\t",spam_as_spam,'\t\t', label_test_ham - ham_as_ham
print "non-spam\t",label_test_spam - spam_as_spam, '\t\t', ham_as_ham

# for question P4(c)
out_sigmas = open('sigmas_spam_ham', 'w')
out_sigmas.write('{}\n{}'.format(sigmas_spam,sigmas_ham))

# compute the expectation of lambda1 (spam)
expected_lambda1 = []
expected_lambda0 = []
for val in sigmas_spam:
    expected_lambda1.append((1+val)/ (2+train_spam_N))
for val in sigmas_ham:
    expected_lambda0.append((1+val)/ (2+train_ham_N))
print "-----Expectations-----"
print "E(lambda1)=", expected_lambda1, "dimension=", len(expected_lambda1)
print "E(lambda0)=", expected_lambda0, "dimension=", len(expected_lambda0)
out_expectaions = open('expectations', 'w')
out_expectaions.write('{}\n{}'.format(expected_lambda1, expected_lambda0))

out_expectations.write('{}\n'.format(test_data[true_spam_but_ham[1]]))
out_expectations.write('{}\n'.format(test_data[true_spam_but_ham[2]]))
out_expectations.write('{}\n'.format(test_data[true_spam_but_ham[3]]))
lines = [line.rstrip('\n') for line in open('README')]
print "vocabularies:", lines
