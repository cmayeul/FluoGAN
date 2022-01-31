#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

Ton = 2 #en nombre d'images
Toff = 4 #en nombre d'images
frames = 300
nemitters = 8
Ion = 1

signal = np.zeros(frames)
for i in range(nemitters) : 
    t = np.log(np.random.rand(100,2))
    t[:,0] *= -Ton
    t[:,1] *= -Toff
    cs = t.flatten().cumsum()
    t = 0
    k = 0
    while t < frames : 
        if cs[k] > int(t) + 1 :
            signal[int(t)] += Ion * (int(t) + 1 - t)
            j = int(t) + 1 
            while cs[k] > j + 1 and j < frames : 
                signal[j] += Ion
                j += 1
            if j < frames :
                signal[j] += Ion * (cs[k] - j)
        else : 
            signal[int(t)] += Ion * (cs[k] - t)
        t = cs[k+1]
        k += 2
    
plt.plot(np.arange(250), signal[50:])
plt.xlabel("images")
plt.ylabel("Nombre de photons émis")
plt.title("Fluctuations d'un groupe d'émetteurs modélisé selon SOFI-tools")
plt.show()