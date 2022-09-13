"""
This is an example of how to run the 1D plane-parallel (slab) model to simulate the spectral albedo and
BRDF.
"""

import sys
import os
from matplotlib import pyplot as plt
import numpy as np
from crrelGOSRT import SlabModel
import vtk
from distutils.version import StrictVersion

### User input required ###
Latitude = 43.8163
Longitude = -72.2740
Time = '02-12 15:35'
Elevation = 553
TimeFormat='%m-%d %H:%M'

nPhotons=20000  # number of photons to use in slab model
###########################

# print(vtk.__version__)
#
# if StrictVersion(vtk.__version__) <= StrictVersion('9.1.0'):
#     print("?")
# else:
#     print("Okay!")
# sys.exit()

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

# Compute solar zenith and azimuth angle based on defined latitude, longitude, elevation, and time
Azimuth,Zenith=GetZenith(Time,Latitude,Longitude,Elevation,TimeFormat)


#%% Example 1 - Calculate spectral albedo, plot and save data to output file

WaveLength=np.arange(400,1600,20)  # choose wavelengths, input for GetSpectralAlbedo function
Slab=SlabModel.SlabModel(namelist='namelist.txt') # load namelist into model
Slab.Initialize() # initialize model (hardcodes namelist parameters into model and performs quality assurance checks)
# run model
Albedo,Absorption,Transmiss,transmissionDict=Slab.GetSpectralAlbedo(WaveLength,Zenith,Azimuth,nPhotons=nPhotons)
# plot albedo
plt.figure()
plt.plot(WaveLength,Albedo,lw='2',color='b',label="Photon Tracking Simple")
plt.show()
# write output to file
Slab.WriteSpectralToFile(os.getcwd()+'/25.0_NOEXT_092022.txt',
    nPhotons,Zenith,Azimuth,WaveLength,Albedo,Absorption,Transmiss,filename='34mm Feb12 Observations - 85% diffuse')


#%% Example 2 - Calculate BRDF and plot

wv=950  # define wavelength for which to calculate BRDF

# run model
BRDFArray,BRAziBins,BRZenBins,albedo,absorbed,transmiss = Slab.RunBRDF(wv,Zenith,Azimuth,nPhotons=nPhotons,binSize=7.5)

print("Wave Length = %i"%wv)
print("Albedo:",albedo/nPhotons)
print("Absorption:",absorbed/nPhotons)
print("Transmission:",transmiss/nPhotons)
print("Sum:",(albedo+absorbed+transmiss)/nPhotons)

dTheta=np.abs(np.radians(BRZenBins[1])-np.radians(BRZenBins[0]))
ZenZen=np.radians(np.tile(BRZenBins, (len(BRAziBins), 1)))

print(np.shape(ZenZen))
print(np.sum(BRDFArray*np.cos(ZenZen)*np.sin(ZenZen))*dTheta**2/nPhotons)
print(albedo/nPhotons)

# plot BRDF
figure,ax,Z=Slab.PlotBRDF(BRAziBins,BRZenBins,BRDFArray,levels=np.linspace(0,1.0,41),cmap='jet',norm=nPhotons,extend='both')
plt.colorbar(Z)
plt.title("BRDF $\lambda$ = %.1f nm | Zenith = %.1f"%(wv, Zenith))
