#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:03:19 2023

@author: djordan
"""

from SED_mc import fitstar, report
name = 'HD10400'
teff_med, teff_u_neg, teff_u_plus, Rstar_med, Rstar_u_neg, Rstar_u_plus = fitstar(name)
print ('Parameters for ' + name + ': ', teff_med, teff_u_neg, teff_u_plus, Rstar_med, Rstar_u_neg, Rstar_u_plus)
report(name)
print('Report for ' + name + 'is in the eports directory')