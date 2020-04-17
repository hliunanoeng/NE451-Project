import math
import random
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt

class BadRandomNumberGenerator(object):
    def __init__(self, a=1277, c=0, m=131072):
        self.a=a
        self.c=c
        self.m=m
        self.S=1
    def getNumber(self):
        self.S=((self.a*self.S)+self.c)%self.m
        return (self.S/self.m)

#Problem 4-10
#(b)

# graph_y=[0]
# l=[]
# n_accepted=0
# for i in range(1,(10**8)+1):
#     x=2*random.random()-1
#     y=2*random.random()-1
#     r2=(x**2)+(y**2)
#     if r2<=1:
#         n_accepted+=1
#         accepted=n_accepted/i
#         E_N2=(accepted-math.pi/4)**2
#         l.append(E_N2)
#     if i in [10**3,10**4,10**5,10**6,10**7,10**8]:
#         y=(sum(l)+((i/10)*graph_y[-1]))/i
#         print(y)
#         graph_y.append(y)
#         l=[]
#
# del graph_y[0]
# plt.style.use('ggplot')
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.loglog([10**3,10**4,10**5,10**6,10**7,10**8],graph_y)
#
# ax1.set_xlabel("N")
# ax1.set_ylabel(r"<$\bar E_{N}^2$>")
# plt.show()

#(c)

# graph_y=[0]
# l=[]
# n_accepted=0
# num=BadRandomNumberGenerator()
# for i in range(1,(10**8)+1):
#     x=2*num.getNumber()-1
#     y=2*num.getNumber()-1
#
#     r2=(x**2)+(y**2)
#     if r2<=1:
#         n_accepted+=1
#         accepted=n_accepted/i
#         E_N2=(accepted-math.pi/4)**2
#         l.append(E_N2)
#     if i in [10**3,10**4,10**5,10**6,10**7,10**8]:
#         y=(sum(l)+((i/10)*graph_y[-1]))/i
#         print(y)
#         graph_y.append(y)
#         l=[]
#
# del graph_y[0]
# plt.style.use('ggplot')
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.loglog([10**3,10**4,10**5,10**6,10**7,10**8],graph_y)
#
# ax1.set_xlabel("N")
# ax1.set_ylabel(r"<$\bar E_{N}^2$>")
# plt.show()


#Problem 4-12

# delta=3
# x0=0
# N=10**6
# data=[x0]
# num_reject=0
#
# def weightDistribution(x):
#     return 1/(math.cosh(x))
#
# def normalizedWeightDistribution(x):
#     return 1 / (math.cosh(x)*math.pi)
#
# for i in range(N-1):
#     new_x=data[-1]+delta*(1-(random.random()*2))
#     w=weightDistribution(data[-1])
#     new_w=weightDistribution(new_x)
#
#     if i==0.01*N:
#         acceptance_rate=(1-(num_reject/(0.01*N)))
#         print(acceptance_rate)
#         if acceptance_rate > 0.4 or acceptance_rate < 0.6:
#             warnings.warn("The accpetance rate is not around 50%, please adjust step size.")
#     if (new_w/w)>=1:
#         data.append(new_x)
#     else:
#         r=random.random()
#         if (new_w/w)>r:
#             data.append(new_x)
#         else:
#             data.append(data[-1])
#             num_reject+=1
#
# normalized=[normalizedWeightDistribution(i) for i in np.linspace(-3.5,3.5,71)]
#
# plt.style.use('ggplot')
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.hist(data,bins=np.linspace(-3.5,3.5,71),density=True, label=r"Normalized Histogram ($\Delta_{bin}$ = 0.1)")
# ax1.plot(np.linspace(-3.5,3.5,71), normalized,label=r"Normalized Weight Distribution 1/ch(x)")
# ax1.set_xlabel("x")
# ax1.set_ylabel("Density")
# ax1.legend()
# plt.show()

#Problem 4-13

def weightDistribution(x):
    return 1/(math.cosh(x))

def f(x):
    return x**2

def Metropolis(N, weight, function, delta=0.5, x0=0, test_tolerance=0.1,limit=[10**(-4),10**4]):

    data=[x0]
    num_reject=0
    for i in range(1, N + 1):
        new_x=data[-1]+delta*(1-(random.random()*2))
        w=weight(data[-1])
        new_w=weight(new_x)

        if int(test_tolerance*N)>=10:
            if i==int(test_tolerance*N):
                acceptance_rate=(1-(num_reject/(test_tolerance*N)))
                # print(acceptance_rate)
                if acceptance_rate < 0.4 or acceptance_rate > 0.6:
                    warnings.warn("The accpetance rate is not around 50%, please adjust step size.")
        else:
            acceptance_rate = None

        if new_x<limit[0] or new_x>limit[1]:
            data.append(data[-1])
            num_reject+=1

        if (new_w/w)>=1:
            data.append(new_x)
        else:
            r=random.random()
            if (new_w/w)>r:
                data.append(new_x)
            else:
                data.append(data[-1])
                num_reject+=1

    f = [function(i) for i in data]
    I = sum(f)/float(len(f))

    if acceptance_rate is None:
        print("Using ", str(N), " sample points, given step size ", str(delta), ", the integral evaluates to be ", str(I), "; \nan acceptance rate is not calculated as the total number of sample points collected is too small (<10).")
    else:
        print("Using ", str(N), " sample points, given step size ", str(delta), ", the integral evaluates to be ", str(I), " with an acceptance rate of ", str(acceptance_rate*100), "%.")

    return I, acceptance_rate


I = Metropolis(N=2**12, weight=weightDistribution, function=f)

# l=np.array([7,8,9,10,11,12,13,14,15,16])
# d=dict()
# for i in l:
#     d[i]=[]
#     N=2**i
#     while len(d[i])<20:
#         result,rate = Metropolis(N,weightDistribution,function=f)
#         if abs(rate - 0.5) < 0.1:
#             d[i].append(result)
#
# with open('Metropolis Method.pickle','wb') as handle:
#     pickle.dump(d,handle)

d = pickle.load(open('Metropolis Method.pickle', 'rb'))

avg=[]
std=[]

l=[7,8,9,10,11,12,13,14,15,16]
N=[]
for i in l:
    N.append(2**i)

for i in l:
    avg.append(np.mean(d[i]))
    std.append(np.std(d[i]))

plt.style.use('ggplot')
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.errorbar(N,avg,yerr=std)
ax1.set_xscale('log',basex=2)
ax1.set_xlabel("N")
ax1.set_ylabel("Simulated Value (Metropolis)")
plt.show()


#Problem 4-14

# def weightDistribution(r_sqrt):
#     return (math.e**(-1*r_sqrt))
#
# def normalizedWeightDistribution(r_sqrt):
#     return (math.e**(-1*r_sqrt))/(math.pi**0.5)
#
# def distributionGenerator(D,delta,N=10**7):
#     x0=np.array([0]*D)
#     data=[x0]
#     num_reject=0
#
#     for i in range(N-1):
#         new_x=np.array([ i+delta*(1-(random.random()*2))for i in data[-1]])
#         new_r_sqrt=(new_x**2).sum()
#         w=weightDistribution((np.array(data[-1])**2).sum())
#         new_w=weightDistribution(new_r_sqrt)
#
#         if i==0.01*N:
#             acceptance_rate=(1-(num_reject/(0.01*N)))
#             print(acceptance_rate)
#             if acceptance_rate < 0.4 or acceptance_rate > 0.6:
#                 warnings.warn("The accpetance rate is not around 50%, please adjust step size.")
#         if (new_w/w)>=1:
#             data.append(new_x)
#         else:
#             r=random.random()
#             if (new_w/w)>r:
#                 data.append(new_x)
#             else:
#                 data.append(data[-1])
#                 num_reject+=1
#     return data

# D=1
# data=distributionGenerator(D, delta=2)
# data=[ (i**2).sum() for i in data]
# with open('D1.pickle','wb') as handle:
#     pickle.dump(data,handle)

# D=2
# data=distributionGenerator(D, delta=1)
# data=[ (i**2).sum() for i in data]
# with open('D2.pickle','wb') as handle:
#     pickle.dump(data,handle)

# D=3
# data=distributionGenerator(D, delta=1)
# data=[ (i**2).sum() for i in data]
# with open('D3.pickle','wb') as handle:
#     pickle.dump(data,handle)

# D=4
# data=distributionGenerator(D, delta=1)
# data=[ (i**2).sum() for i in data]
# with open('D4.pickle','wb') as handle:
#     pickle.dump(data,handle)

# D=5
# data=distributionGenerator(D, delta=0.75)
# data=[ (i**2).sum() for i in data]
# with open('D5.pickle','wb') as handle:
#     pickle.dump(data,handle)

# D1 = pickle.load(open('D1.pickle', 'rb'))
# D2 = pickle.load(open('D2.pickle', 'rb'))
# D3 = pickle.load(open('D3.pickle', 'rb'))
# D4 = pickle.load(open('D4.pickle', 'rb'))
# D5 = pickle.load(open('D5.pickle', 'rb'))
#
# plt.style.use('ggplot')
# fig = plt.figure()
# ax1 = fig.add_subplot(321)
# ax2 = fig.add_subplot(322)
# ax3 = fig.add_subplot(323)
# ax4 = fig.add_subplot(324)
# ax5 = fig.add_subplot(325)
#
# weights=np.ones(len(D1))/len(D1)
# ax1.hist(D1,bins=np.linspace(0,5,51),density=True, label=r"Normalized Histogram for D = 1 ($\Delta_{bin}$ = 0.1)")
# ax1.set_xlabel("r")
# ax1.set_ylabel("Density")
# ax1.legend()
#
# ax2.hist(D2,bins=np.linspace(0,5,51),density=True, label=r"Normalized Histogram for D = 2 ($\Delta_{bin}$ = 0.1)")
# ax2.set_xlabel("r")
# ax2.set_ylabel("Density")
# ax2.set_ylim(0,1)
# ax2.legend()
#
# ax3.hist(D3,bins=np.linspace(0,5,51),density=True, label=r"Normalized Histogram for D = 3 ($\Delta_{bin}$ = 0.1)")
# ax3.set_xlabel("r")
# ax3.set_ylabel("Density")
# ax3.set_ylim(0,1)
# ax3.legend()
#
# ax4.hist(D4,bins=np.linspace(0,5,51),density=True, label=r"Normalized Histogram for D = 4 ($\Delta_{bin}$ = 0.1)")
# ax4.set_xlabel("r")
# ax4.set_ylabel("Density")
# ax4.set_ylim(0,1)
# ax4.legend()
#
# ax5.hist(D5,bins=np.linspace(0,5,51),density=True, label=r"Normalized Histogram for D = 5 ($\Delta_{bin}$ = 0.1)")
# ax5.set_xlabel("r")
# ax5.set_ylabel("Density")
# ax5.set_ylim(0,1)
# ax5.legend()
#
# plt.show()
