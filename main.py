"""
  Data source:
  Statistics Canada. Table 36-10-0103-01 Gross domestic product, income-based, quarterly (x 1,000,000). DOI: https://doi.org/10.25318/3610010301-eng
"""

"""
  python linear regression method derived from least squares example from Dr. Patrick Walls' Math 307 (included here)
"""

"""
  Python program to compute Canada's Solow residual (total factor productivity) based on linear regression of GDP, capital and labour

  By James Yu, 2020
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

"""
  We need to solve log(Y) = log(A) + alog(K) + (1-a)log(N)

  For every entry in the data, we compute log(Y) and alog(K)+ (1-a)log(N), solving for log(A)

  First, we determine all the values of log(A)
  Let Q1 1961 be time t = 0
"""

print("Starting...")

# unpack the GDP values
Y = []

with open("gdp.txt") as f:
  for line in f:
    Y.append(int(line.replace("\n", "")))

# unpack the labour values
N = []

with open("labour.txt") as f:
  for line in f:
    N.append(int(line.replace("\n", "")))

# compute log(A) for each period
logA = []

print("Computing log(A)...")
for i in range(len(Y)):
  # compute the capital share of output
  a = 1 - (N[i]/Y[i])
  print(f"Capital share t={i}: " + str(a))

  # compute capital
  K = Y[i] - N[i]

  # determine log(A)
  logA.append(math.log(Y[i]*1000000 - (a * math.log(K*1000000) + (1 - a) * math.log(N[i]*1000000))))

# do the least squares
print("Computing Regression...")
A = np.column_stack([np.ones(238), np.array(range(238))])
c = la.solve(A.T @ A, A.T @ logA)
print("Done")
print(f"c_0 = {c[0]}, c_1 = {c[1]}")
print(f"A(t) = {math.exp(c[0])} + {math.exp(c[1])}^t")

# so log(A_t) = c[0] + c[1] * t
# therefore A_t = math.exp(c[0] + c[1] * t)
# = e^c[0] + e^(c[1] * t)

ts = range(238)
logas = [math.exp(c[0]+c[1]*b) for b in ts]

fig, sub = plt.subplots(2, sharex=True)
fig.suptitle("Solow Residual")

sub[0].plot(ts,logas,'r',linewidth=2)
dubloga = [math.exp(b) for b in logA]
sub[0].scatter(ts,dubloga,alpha=0.5,lw=0,s=10)
sub[0].set(ylabel = "A")

sub[1].plot(ts,c[0]+c[1]*ts,'r',linewidth=2)
sub[1].scatter(ts,logA,alpha=0.5,lw=0,s=10)
sub[1].set(xlabel = "Time (0 = Q1 1961)", ylabel =  "log(A)")

plt.show()