# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:39:31 2019

@author: hanweiz
"""

import numpy as np
from scipy.optimize import leastsq
#from math import factorial
import matplotlib.pyplot as plt
from algopy import UTPM

data = np.loadtxt("Data/ALICE/ALICE_Cj_8000GeV_eta_3.4.txt")
multpdata = np.loadtxt("Data/ALICE/ALICE_Mult_8TeV_eta3.4.txt")
x_data = data[:213,0]
y_data = data[:213,1]
chg_part = multpdata[:,0]
multiplicity = multpdata[:,1]

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

def probconv(p, x):
    """1-component BD(NBD)"""
#    def BDNBD(p, x):
#        return ((p[1] * (((1 - p[3]) / (1 - (p[3] * x))) ** p[2])) + (1. - p[1])) ** p[0]
    """1-component BD(NBD) X NBD"""
    def BDNBD(p, x):
        return (((p[1] * (((1. - p[3]) / (1. - (p[3] * x))) ** p[2])) + (1. - p[1])) ** p[0]) * (((1. - p[5]) / (1. - (p[5] * x))) ** p[4])
    """2-component BD(NBD) """
#    def BDNBD(p, x):
#        return (p[8] * (((p[1] * (((1 - p[3]) / (1 - (p[3] * x))) ** p[2])) + (1. - p[1])) ** p[0])) + ((1. - p[8]) * ((p[5] * (((1. - p[7]) / (1. - (p[7] * x))) ** p[6]) + (1. - p[5])) ** p[4]))
    """2-component BD(NBD) X NBD"""
#    def BDNBD(p,x):
#        return (p[12] * ((((p[1] * (((1. - p[3]) / (1. - (p[3] * x))) ** p[2])) + (1. - p[1])) ** p[0]) * (((1. - p[5]) / (1. - (p[5] * x))) ** p[4]))) + ((1. - p[12]) * ((((p[7] * (((1. - p[9]) / (1. - (p[9] * x))) ** p[8])) + (1. - p[7])) ** p[6]) * (((1. - p[11]) / (1. - (p[11] * x))) ** p[10])))
    
    D = len(x); P = 1
    xderv = UTPM(np.zeros((D,P)))
    xderv.data[0,0] = 0 #the value to evaluate Cj at corresponding to z=0
    xderv.data[1,0] = 1
    derv = BDNBD(p, xderv)
    prob = derv.data[:, 0]
    return prob
    

"""1-component BD(NBD)"""
#pmultp = [0.540, 0.992, 2.298, 0.971]
#pCj = [0.8575, 0.98123, 2.59196, 0.94179]
#pmultp = [0.55608, 0.991598, 2.22349, 0.972442]
#pCj = [0.705393, 0.990005, 2.36823, 0.962828]

"""1-component BD(NBD) X NBD"""
#pmultp = [0.524227, 0.992543, 2.29104, 0.9771389, 0.891887, 0.361161]
#pCj = [0.842873, 0.985068, 2.53949, 0.948632, 1.01767, 0.228849]
#pCj1 = [0.847115, 0.984019, 2.55412, 0.946699, 1.0752, 0.217678]
pmultp = [0.541296, 0.991604, 2.21311, 0.972968, 0.860689, 0.356496]
pCj = [0.704847, 0.99001, 2.36823, 0.962836, 1.03356, 0.250412]

"""2-component BD(NBD)"""
#pmultp = [0.75129, 0.957666, 2.86246, 0.908163, 1.27775, 0.975339, 2.75544, 0.951601, 0.527693]
#pCj = [1., 0.861486, 9.70189, 0.614077, 0.612333, 0.993783, 2.22854, 0.971891, 0.10364]
#pmultp = [0.747569, 0.96031, 2.82844, 0.911347, 1.22787, 0.978399, 2.72543, 0.954777, 0.505032]
#pCj = [0.851076, 0.92235, 3.18746, 0.870279, 0.993161, 0.984275, 2.76002, 0.957805, 0.395386]



#def errfn(p, x, y):
#    return model(p, x) - y

#pbestCj, covCj, infodictCj, mesgCj, ierCj = leastsq(errfn, p0, args=(x_data, y_data), ftol=1E-8, full_output=True)

"""plotting code"""
#plt.title("ALICE 7TeV $|\eta| < 3.4$")
#plt.plot(x_data, y_data, 'b.', label='$C_j$ data')
#plt.plot(x_data, model(p0, x_data), 'r-', label='multp_fit')
#plt.plot(x_data, model(pbestCj, x_data), 'k-', label='Cj_fit 2-comp_BD(NBD)')   
#plt.legend(loc=3, fontsize=8)
#plt.ylabel("$\\langle N \\rangle \cdot C_j$")
#plt.xlabel("j")
#"""1-component BD(NBD) X NBD"""
##plt.figtext(0.25, 0.78, "$K1_{BD}=%.3f$, $p1_{BD}=%.3f$, $k1_{NBD}=%.3f$, $p1'_{NBD}=%.3f$, $k_{1NBD}=%.3f$, $p'_{1NBD}=%.3f$" % (p1[0], p1[1], p1[2], p1[3], p1[4], p1[5]), fontsize=8)
#"""2-component BD(NBD)"""
#plt.figtext(0.25, 0.78, "$K1_{BD}=%.3f$, $p1_{BD}=%.3f$, $k1_{NBD}=%.3f$, $p1'_{NBD}=%.3f$" % (pbestCj[0], pbestCj[1], pbestCj[2], pbestCj[3]), fontsize=8)
#plt.figtext(0.25, 0.73, "$K2_{BD}=%.3f$, $p2_{BD}=%.3f$, $k2_{NBD}=%.3f$, $p2'_{NBD}=%.3f$, $w=%.3f$" % (pbestCj[4], pbestCj[5], pbestCj[6], pbestCj[7], pbestCj[8]))

"""plotting multp and Cj subplots"""
fig, (multp, modcomb) = plt.subplots(2, sharex=False, sharey=False)
fig.suptitle("ALICE 8 TeV $|\eta| < 3.4$\tBD(NBD) $\\times$ NBD")

multp.set_yscale("log")
multp.set(xlabel="N", ylabel="P(N)")
multp.plot(chg_part, multiplicity, 'b.', label='Expt Data')
multp.plot(chg_part, probconv(pmultp, chg_part), 'r-', label="Multp_Fit")
multp.plot(chg_part, probconv(pCj, chg_part), 'k-', label="Cj_Fit")
multp.legend(loc="upper right", fontsize=7)

modcomb.set(xlabel='j', ylabel="$\\langle N \\rangle C_j$")
modcomb.plot(x_data, y_data, 'b.', label="Expt_Data")
modcomb.plot(x_data, model(pmultp, x_data), 'r-', label="Multp_Fit")
modcomb.plot(x_data, model(pCj, x_data), 'k-', label="Cj_Fit")
modcomb.legend(loc="lower left", fontsize=7)

plt.savefig("ALICE8_3.4_1-compBD(NBD)XNBD_subplots.pdf", format='pdf')