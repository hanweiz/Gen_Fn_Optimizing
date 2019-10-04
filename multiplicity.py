# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:05:50 2019

@author: hanweiz
"""

import numpy as np
from scipy.optimize import leastsq
#from math import factorial
import matplotlib.pyplot as plt
from algopy import UTPM

"""load data file using np.loadtxt. Change code according to structure of data file"""
multp = np.loadtxt("Data/ALICE/ALICE_Mult_7TeV_eta3.4_5.0.txt")
x_data = multp[:,0]
y_data = multp[:,1]
err = multp[:, 2]

""" Following errors for CMS Data only"""
#yerr=multp[:,2].T
#xerr=multp[:,4].T

"""now to write the generating function to differentiate
Here we use F_BD(G_NBD(z)) and output an array of probability
"""
#p0 = [0.727, 0.9893, 2.505, 0.9562, 2.2, 1.1]
def probconv(p, x):
    """1-component BD(NBD)"""
#    def BDNBD(p, x):
#        return ((p[1] * (((1 - p[3]) / (1 - (p[3] * x))) ** p[2])) + (1. - p[1])) ** p[0]
    """1-component BD(NBD) X NBD"""
#    def BDNBD(p, x):
#        return (((p[1] * (((1. - p[3]) / (1. - (p[3] * x))) ** p[2])) + (1. - p[1])) ** p[0]) * (((1. - p[5]) / (1. - (p[5] * x))) ** p[4])
    """2-component BD(NBD) """
    def BDNBD(p, x):
        return (p[8] * (((p[1] * (((1 - p[3]) / (1 - (p[3] * x))) ** p[2])) + (1. - p[1])) ** p[0])) + ((1. - p[8]) * ((p[5] * (((1. - p[7]) / (1. - (p[7] * x))) ** p[6]) + (1. - p[5])) ** p[4]))
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

def errfn(p, x, y, err):
    return ((probconv(p, x) - y) / err)

def chi2(p, x, y, err):
    return np.sum(((probconv(p, x) - y) / err) **2)

#p0 = [0.5399, 0.992, 2.298, 0.971]    #for 1-component BD(NBD)
#p0 = [0.5526, 0.990, 2.315, 0.974, 2., 0.9] #for 1-component BD(NBD) X NBD
#p0 = [0.5, 0.9, 2., 0.85, 0.5, 0.7, 2.6, 0.98, 0.5]    #for 2-component BD(NBD)
p0 = [0.793743, 0.952645, 2.89218, 0.918969, 1.392, 0.979348, 2.80535, 0.954537, 0.544544]
#p0 = [0.52, 0.95, 2.3, 0.95, 1., 0.4, 0.52, 0.95, 2.3, 0.95, 1., 0.4, 0.5] 
#p0 = [0.680585, 0.960456, 2.75574, 0.91669, 0.616862, 0.533712, 1., 0.997426, 3.46468, 0.952195, 0.00878733, 0.906642, 0.548319]    #for 2-component BD(NBD) X NBD
pbest, cov, infodict, mesg, ier = leastsq(errfn, p0, args=(x_data, y_data, err), ftol=1E-8, full_output=True)
chi2_value = chi2(pbest, x_data, y_data, err)

"""plotting code from henceforth"""

plt.yscale('log')
#plt.errorbar(x_data, y_data, yerr=err, fmt='b.', elinewidth=1, label='ALICE7TeV $3.4<|\eta|<5.0$', ms=3)
plt.plot(x_data, y_data, 'b.', label='$3.4 < |\eta| < 5.0$')
plt.plot(x_data, probconv(pbest, x_data), 'r-', label='2-comp$BD(NBD)$', linewidth=2)
plt.legend(loc='best')
plt.xlabel("N"); plt.ylabel("P(N)")
plt.title("ALICE 7 TeV $3.4 < |\eta| < 5.0$")

"""1-component chi2"""
#plt.figtext(0.13, 0.45, "$\chi^2/dof = %.3f/%d$" % (chi2_value, len(x_data)-len(p0)), fontsize=8)

"""2-component plots chi2 with weights"""
plt.figtext(0.13, 0.45, "$\chi^2/dof = %.3f/%d$, $w=%.3f\pm%1.1e$" % (chi2_value, len(x_data)-len(p0), pbest[-1], cov[-1,-1]), fontsize=8)

"""for 1-component BD(NBD)"""
plt.figtext(0.13, 0.40, "$K1_{BD}=%.3f \pm %1.1e$ , $p1_{BD}=%.3f \pm %1.1e$" % (pbest[0], cov[0,0], pbest[1], cov[1,1]), fontsize=8)
plt.figtext(0.13, 0.35, "$k1_{NBD}=%.3f\pm%1.1e$, $p1'=%.3f\pm%1.1e$" % (pbest[2], cov[2,2], pbest[3], cov[3,3]), fontsize=8)

"""for 1-component BD(NBD) X NBD"""
#plt.figtext(0.13, 0.30, "$k_{1NBD}=%.3f\pm%1.1e$, $p'_{1NBD}=%.3f\pm%1.1e$" % (pbest[4], cov[4,4], pbest[5], cov[5,5]), fontsize=8)

"""for 2-component BD(NBD)"""
plt.figtext(0.13, 0.30, "$K2_{BD}=%.3f \pm %1.1e$ , $p2_{BD}=%.3f \pm %1.1e$" % (pbest[4], cov[4,4], pbest[5], cov[5,5]), fontsize=8)
plt.figtext(0.13, 0.25, "$k2_{NBD}=%.3f\pm%1.1e$, $p2'=%.3f\pm%1.1e$" % (pbest[6], cov[6,6], pbest[7], cov[7,7]), fontsize=8)


"""for 2-component BD(NBD) X NBD"""
#plt.figtext(0.13, 0.25, "$K2_{BD}=%.3f \pm %1.1e$ , $p2_{BD}=%.3f \pm %1.1e$" % (pbest[6], cov[6,6], pbest[7], cov[7,7]), fontsize=8)
#plt.figtext(0.13, 0.20, "$k2_{NBD}=%.3f\pm%1.1e$, $p2'=%.3f\pm%1.1e$" % (pbest[8], cov[8,8], pbest[9], cov[9,9]), fontsize=8)
#plt.figtext(0.13, 0.15, "$k_{2NBD}=%.3f\pm%1.1e$, $p'_{2NBD}=%.3f\pm%1.1e$" % (pbest[10], cov[10,10], pbest[11], cov[11,11]), fontsize=8)




plt.savefig("ALICE7_3.4_5.0_2-compBD(NBD)_multiplicityfit.pdf", format='pdf')
#print("chi2 is ", chi2(pbest, x_data, multp[:,1]))