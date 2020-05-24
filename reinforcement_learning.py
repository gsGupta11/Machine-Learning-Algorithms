

# **Reinforcement Learning**

**Data Preprocessing**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

"""**Algo 1 - Implementing UCB**"""

import math
N = len(dataset.values)
d = len(dataset.values[0])
ads_selected_ucb = []
no_of_selections = [0]*d
rewards_each_ad =[0]*d

for i in range(0,N):
  max_ucb = 0
  ad=0
  for j in range(0,d):
    if no_of_selections[j]>0:
      aver = rewards_each_ad[j]/no_of_selections[j]
      delta = (1.5*math.log(i+1)/no_of_selections[j])**0.5
      ucb = aver + delta

    else:
      ucb = 1e400
    if ucb>max_ucb:
      max_ucb=ucb
      ad=j
  ads_selected_ucb.append(ad)
  no_of_selections[ad]+=1
  rewards_each_ad[ad]+=dataset.values[i,ad]

"""**Algo 2 - Implementing Thomsons Sampling**"""

import random
N = len(dataset.values)
d = len(dataset.values[0])
ads_selected_thom = []
reward1=[0]*d
reward0=[0]*d
for i in range(0,N):
  ad = 0
  max_random = 0
  for j in range(0,d):
    random_gen=random.betavariate(reward1[j]+1,reward0[j]+1)
    if random_gen>max_random:
      max_random = random_gen
      ad=j
  ads_selected_thom.append(ad)
  if dataset.values[i,ad]==0:
    reward0[ad]+=1
  else:
    reward1[ad]+=1

"""**Histogram for UCB**"""

plt.hist(ads_selected_ucb)
plt.title("Histogram for representing selection of most optimal ad")
plt.xlabel("Ads")
plt.ylabel("No. of time each ad was selected")
plt.show()

"""**Histogram for Thomson Sampling**"""

plt.hist(ads_selected_thom)
plt.title("Histogram for representing selection of most optimal ad")
plt.xlabel("Ads")
plt.ylabel("No. of time each ad was selected")
plt.show()
