# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 16:04:42 2021

@author: RDCRLJTP
"""
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
from math import sin, cos, radians
import pyvista as pv
import matplotlib.pyplot as plt
from crrelGOSRT import SlabModel
import vtk
from distutils.version import StrictVersion

print(vtk.__version__)

if StrictVersion(vtk.__version__) <= StrictVersion('9.1.0'):
    print("?")
else:
    print("Okay!")
sys.exit()

def GetZenith(time,latitude,longitude,elevation,timeformat='%Y-%m-%d_%H:%M:%S'):
    from solarposition import sunposition as sunPos
    from datetime import datetime as DT
    """
        Function to compute Zenith and Azimuth angle (in degrees) From lat/lon/time/elevation

        Inputs: time (strptime, or string)


        Copyright (c) 2015 Samuel Bear Powell

        Note that this uses the publically available "sunposition"
        code written by Samuel Bear Powell and available from:
        https://github.com/s-bear/sun-position.git

        Use of this code is used in accordance with the MIT license, and is used
        solely for the purposes of computing solar zenith and azimuth angles as inputs to the
        RTM code.  Note that this code is included in the "main" directory as solarposition

        Ibrahim Reda, Afshin Andreas, Solar position algorithm for solar radiation applications,
        Solar Energy, Volume 76, Issue 5, 2004, Pages 577-589, ISSN 0038-092X,
        http://dx.doi.org/10.1016/j.solener.2003.12.003.
    """

    print("------------------------")
    print("  USING sunposition.py to estimate solar zenith and azimuth angle!" )
    print("  Returns azimuth angle and zenith angle in degrees! ")
    print("  Note that the azimuth angle here is 0 for the east direction!")
    print("------------------------")

    if isinstance(time,DT) == False:
        if isinstance(time,str) == False:
            print("Time must either be a string or a datetime!")
            sys.exit()
        time=DT.strptime(time,timeformat)

    az,zen = sunPos.sunpos(time,latitude,longitude,elevation)[:2] #discard RA, dec, H

    if np.cos(np.radians(zen)) <= 0:
        print("------------------------")
        print("Sun is Below Horizon at %.2f/%.2f at %s UTC"%(latitude,longitude,time))
        print("!!!You Cannot use these angles to set the incident radiation!!!")
        print(" Azimuth= %.2f/ Zenith = %.2f"%(az-90,zen))
        print("------------------------")

    return az-90,zen


Latitude = 43.8163
Longitude = -72.2740
Time = '02-12 15:35'
Elevation = 553
TimeFormat='%m-%d %H:%M'

Azimuth,Zenith=GetZenith(Time,Latitude,Longitude,Elevation,TimeFormat)

WaveLength=np.arange(400,1500,20)  # input for GetSpectralAlbedo function
Slab=SlabModel.SlabModel(namelist='namelist.txt')
Slab.Initialize()
Azi,Zenith=GetZenith(Time,Latitude,Longitude,Elevation,timeformat='%m-%d %H:%M')
#
Albedo,Absorption,Transmiss,transmissionDict=Slab.GetSpectralAlbedo(WaveLength,Zenith,Azi,nPhotons=1000)
plt.figure()
plt.plot(WaveLength,Albedo,lw='2',color='b',label="Photon Tracking Simple")
plt.show()
