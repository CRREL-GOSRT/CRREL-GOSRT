## Continuous Medium Photon-Tracking Model ##
import sys
import numpy as np
from matplotlib import pyplot as plt
import glob as glob
from crrelGOSRT import SlabModel, Utilities
import pandas as pd


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

def addAlbedo(ax,diams=[0.15],labels=True):

    filepath='/Users/rdcrltwl/Desktop/SnowOpticsProject/Ice-refractive-index/'
    rfile=glob.glob(filepath+'*.csv')[0]

    data=pd.read_csv(rfile)

    wavl=np.linspace(0.5,2,2000)*1E-6 ## Convert nanometers to meters

    for d in diams:
        alb,abso=albedoPlot(wavl,data,d0=d)
        ax=plt.plot(wavl*1E9,alb,label="$D_0=%.1f$"%d)

    if labels == True:
        plt.ylabel("Albedo")
        plt.xlabel("$\lambda$ (nm)")
        plt.legend()
        plt.grid()
        plt.xlim(0.3,2)

def albedoPlot(wavl,data,Ko=9./7.,b=4.53,d0=0.15):
    wv=data.wave.values*1E-6
    kice=data.Im

    kice1=np.interp(np.log(wavl),np.log(wv),np.log(kice))

    kice1=np.exp(kice1)
    abso=4.*np.pi*kice1/wavl

    R=np.exp(-b*np.sqrt(abso*d0/1000.))
    return R,kice1

## Initialize Slab Model
Slab=SlabModel.SlabModel()

datetime='2022-03-02_16:00:00'
lat=43.72
lon=-72.27
elev=400.

Azi,Zenith = GetZenith(datetime,lat,lon,elev)

## Set Depth (in mm)
depth=900 ## depth in mm
## Define a range of WaveLengths to use in GetSpectralAlbedo()
WaveLength=np.arange(400,1600,25)

## Specify the number of photons to use.  Typically 10,000 is sufficient for accuracy, more should (in theory) be more accurate,
## but SIGNIFICANTLY increases Computational Expense
nPhotons=20000

dep=depth

## Use the Utilities module to get a nice visual of the visible spectrum to add to the figure (optional)
cols=Utilities.WaveLengthToColor(WaveLength, gamma=0.8)

## define the figure.
plt.figure(figsize=(10,10))
ax=plt.subplot(111)

## Output the data to this path.
OutputFilePath='/Folder/to/write/out/data/'

OutputFilePath='/Users/rdcrltwl/Desktop/Circles/Spectral/'
## Here is where you need to modify the script to make sure it matches your file paths.
#Slab.namelistDict['MasterPath']='YOUR/PATH/HERE/'

#Slab.namelistDict['MasterPath']='/Users/rdcrltwl/Desktop/Circles/Outputs/'
SnowTypes=['Compressed']#'Spheres_0_FINAL_MEDIUM.txt','Spheres_0_FINAL_PARTICLE.txt']#,'Spheres_1_PhaseOnly_big.txt']
#Slab.namelistDict['LayerTops']=[depth,0] ## Single layer model, top is depth, bottom is zero.
#Slab.namelistDict['Extinction']=1
#Slab.namelistDict['DiffuseFraction']=1.0



#Azi,Zenith=65.,60.
## loop over each snow type, and tun the spectral model.
ax=plt.subplot(111)
for sdx, s in enumerate(SnowTypes):
#    Slab.namelistDict['PropFileNames']=[s]
    Slab.Initialize() ## Initialize the model (i.e., encode the namelist dictionary into hard variables.)

    Albedo, Absorption,Transmiss,transmissionDict=Slab.GetSpectralAlbedo(WaveLength,Zenith,Azi,nPhotons=nPhotons) # RUN THE MODEL!

    ## WRITE OUTPUT TO FILE ##
    Slab.WriteSpectralToFile('%s/%s_%s_compressed_cm.txt'%(OutputFilePath,s,depth),
        nPhotons,Zenith,Azi,WaveLength,Albedo,Absorption,Transmiss,filename='%s_cm Single Layer Large'%dep)

    ## PLOT THE DATA! ##
    plt.plot(WaveLength,Albedo,lw='2',label="%s"%s)


## SHOW VISIBLE SPECTRUM!
for idx,i in enumerate(WaveLength):
    plt.scatter(i,0.2,color=cols[idx],marker='s',edgecolor='k')

addAlbedo(ax,[0.3,0.6,1.0],labels=False)

## Add some formatting to the graph.
plt.xlabel("WaveLength (nm)")
plt.ylabel("albedo")
plt.title("Albedo (SZA= %.1f $^{\circ}$)"%Zenith)
plt.grid()
plt.legend()
plt.show()
sys.exit()
