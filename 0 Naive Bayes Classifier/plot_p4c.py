from numpy.random import rand
import matplotlib
#matplotlib.use('gtkagg')
import matplotlib.pyplot as plt




E_lambda1= [1.6209430496019597, 1.7483159828536436, 4.343539497856705, 1.8022045315370483, 5.447642375995101, 1.8395590936925903, 2.8499693815064298, 2.30006123698714, 1.8058787507654623, 3.652173913043478, 1.3214941824862216, 5.838334353949786, 1.5248009797917943, 0.8775260257195346, 1.1586037966932028, 5.358848744641763, 2.9571341090018373, 3.358236374770361, 22.936313533374157, 2.0930802204531536, 14.049601959583589, 2.2780159216166567, 2.5988977342314756, 2.238211879975505, 0.16166564605021433, 0.0863441518677281, 0.015921616656460504, 0.21187997550520515, 0.01224739742804654, 0.065523576240049, 0.015309246785058175, 0.006736068585425597, 0.16105327617881202, 0.0226576852418861, 0.08205756276791182, 0.3453766074709124, 0.45866503368034295, 0.05878750765462339, 0.12063686466625842, 0.4304960195958359, 0.001224739742804654, 0.032455603184323334, 0.10777709736680956, 0.06307409675443969, 1.3919167176974894, 0.1702388242498469, 0.018983466013472138, 0.028169014084507043, 0.3086344151867728, 1.426821800367422, 0.11818738518064911, 5.461726883037355, 2.080220453153705, 0.9589712186160441]
E_lambda0 = [0.5209080047789725, 2.1286340103544403, 1.2588610115491836, 0.010752688172043012, 1.4289127837514934, 0.25806451612903225, 0.0991636798088411, 0.2831541218637993, 0.34727200318598167, 1.0653126244524094, 0.17244125846276384, 3.902031063321386, 0.38908801274392674, 0.3508562325766627, 0.07805655117483075, 0.4022301871764237, 0.4221425726802071, 0.6045400238948626, 6.675428116288331, 0.039824771007566706, 2.8092393468737553, 0.001991238550378335, 0.06332138590203107, 0.053365193150139385, 7.565909996017523, 3.481879729191557, 8.655117483074472, 1.6200716845878136, 1.4129828753484668, 1.3663878932696136, 0.9044205495818399, 0.6475507765830346, 1.1636798088410991, 0.6547192353643967, 1.4229390681003584, 1.157706093189964, 1.5205097570688968, 0.09159697331740342, 0.942652329749104, 0.6742333731581044, 0.49143767423337315, 1.7212266029470331, 0.596973317403425, 0.8195937873357229, 2.3058542413381122, 0.7837514934289128, 0.07487056949422541, 0.41656710473914776, 0.27598566308243727, 1.2735961768219832, 0.22819593787335724, 0.5009956192751892, 0.12226204699322979, 0.20987654320987653]

lines = ['make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses', 'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000', 'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet', '857', 'data', '415', '85', 'technology', '1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table', 'conference', ';', '(', '[', '!', '$', '#']

ax0 = plt.subplot(111)
#ax1 = plt.subplot(122)


ax0.scatter(lines, E_lambda1, color = 'r', alpha = 0.5, label='E(lambda1)')

ax0.scatter(lines, E_lambda0, color = 'g', alpha = 0.5, label='E(lambda0)')

true_spam_but_ham1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,0,0,0,0,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,0,0,0,0,0,0,0,19,0,0]
true_spam_but_ham2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,0,0]
true_spam_but_ham3 = [0,0,0,0,11,21,11,0,0,0,0,0,0,0,0,0,0,0,32,0,11,0,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0]

#ax0.scatter(lines, true_spam_but_ham1, color = 'b', alpha = 1, label='true_spam_but_non-spam')
#ax0.scatter(lines, true_spam_but_ham2, color = 'b', alpha = 1, label='true_spam_but_non-spam')
ax0.scatter(lines, true_spam_but_ham3, color = 'b', alpha = 1, label='true_spam_but_non-spam')

#plt.title('true_spam_but_non-spam\n(p_spam = 3.46610905196e-91)')
#plt.title('true_spam_but_non-spam\n(p_spam = 0.00431017925322)')
plt.title('true_spam_but_non-spam\n(p_spam = 5.18945652963e-20)')
plt.ylabel('times')
plt.legend()
plt.xticks(rotation = 90)
plt.show()

#print len(E_lambda1), len(E_lambda0)