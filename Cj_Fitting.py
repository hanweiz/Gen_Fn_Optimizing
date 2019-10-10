import numpy as np
from scipy.optimize import leastsq
#from math import factorial
import matplotlib.pyplot as plt
from algopy import UTPM

data = np.loadtxt("Data/ALICE/ALICE_Cj_7000GeV_eta_3.4.txt")
x_data = data[:216,0]
y_data = data[:216,1]

def model(p, x):
     """1-component BD(NBD)"""
#     def BDNBD(p, x):
#         return np.log(((p[1] * (((1 - p[3]) / (1 - (p[3] * x))) ** p[2])) + (1. - p[1])) ** p[0])
     """1-component BD(NBD) X NBD"""
     def BDNBD(p, x):
        return np.log((((p[1] * (((1. - p[3]) / (1. - (p[3] * x))) ** p[2])) + (1. - p[1])) ** p[0]) * (((1. - p[5]) / (1. - (p[5] * x))) ** p[4]))
     """2-component BD(NBD) """
#     def BDNBD(p, x):
#        return np.log((p[8] * (((p[1] * (((1 - p[3]) / (1 - (p[3] * x))) ** p[2])) + (1. - p[1])) ** p[0])) + ((1. - p[8]) * ((p[5] * (((1. - p[7]) / (1. - (p[7] * x))) ** p[6]) + (1. - p[5])) ** p[4])))
     """2-component BD(NBD) X NBD"""
#    def BDNBD(p,x):
#        return np.log((p[12] * ((((p[1] * (((1. - p[3]) / (1. - (p[3] * x))) ** p[2])) + (1. - p[1])) ** p[0]) * (((1. - p[5]) / (1. - (p[5] * x))) ** p[4]))) + ((1. - p[12]) * ((((p[7] * (((1. - p[9]) / (1. - (p[9] * x))) ** p[8])) + (1. - p[7])) ** p[6]) * (((1. - p[11]) / (1. - (p[11] * x))) ** p[10]))))
    
     D = len(x)+1; P = 1
     xderv = UTPM(np.zeros((D,P)))
     xderv.data[0,0] = 0 #the value to evaluate Cj at corresponding to z=0
     xderv.data[1,0] = 1
     derv = BDNBD(p, xderv)
     comb = derv.data[:, 0]
     Cj = np.zeros(len(x))
     for i in range(len(comb)-1):
         Cj[i] = comb[i+1] * (i+1)     
     return Cj

p0 = [0.842873, 0.985068, 2.53949, 0.948632, 1.01767, 0.228849]

def errfn(p, x, y):
    return model(p, x) - y

pbestCj, covCj, infodictCj, mesgCj, ierCj = leastsq(errfn, p0, args=(x_data, y_data), ftol=1E-8, full_output=True)

plt.plot(x_data, y_data, 'b.', label='$C_j$ data')
plt.plot(x_data, model(pbestCj, x_data), 'k-', label='Cj_fit 1-comp_BD(NBD)$\\times$NBD')