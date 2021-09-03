## Continuous Medium Photon-Tracking Model ##
import sys
sys.path.append("/Users/rdcrltwl/Desktop/CRRELRTM/CRREL-GOSRT")
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import glob as glob
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from main import SlabModel, RTcode
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib import cm


def CurveFit(z,mu):
    return np.exp(-mu*z)

def albedoPlot(wavl,data,Ko=9./7.,b=4.29,d0=0.15):
    wv=data.wave.values*1E-6
    kice=data.Im

    kice1=np.interp(np.log(wavl),np.log(wv),np.log(kice))

    kice1=np.exp(kice1)
    abso=4.*np.pi*kice1/wavl

    R=np.exp(-Ko*b*np.sqrt(abso*d0/1000.))
    return R,kice1

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

Slab=SlabModel.SlabModel()
Slab.Initialize()
Azi,Zenith=Slab.GetZenith()



WaveLength=np.arange(400,1300,50)

nPhotons=1500

Edown=RTcode.PlankFunction(WaveLength/1000.)
cols=Slab.WaveLengthToColor(WaveLength, gamma=0.8)

plt.figure(figsize=(10,10))
ax=plt.subplot(111)
Albedo, Absorption,Transmiss,transmissionDict=Slab.GetSpectralAlbedo(WaveLength,Zenith,Azi,nPhotons=nPhotons)

#Slab.WriteSpectralToFile('/Users/rdcrltwl/Desktop/NewRTM/Examples/multi_21.5cm.txt',
#    nPhotons,Zenith,Azi,WaveLength,Albedo,Absorption,Transmiss,filename='21.5cm MultiLayer')

plt.plot(WaveLength,Albedo,lw='2',color='r',label="Test Spectral Albedo")
addAlbedo(ax,[0.3],labels=False)

## SHOW VISIBLE SPECTRUM!
for idx,i in enumerate(WaveLength):
    plt.scatter(i,0.2,color=cols[idx],marker='s',edgecolor='k')

plt.xlabel("WaveLength (nm)")
plt.ylabel("albedo")
plt.title("Albedo (SZA= %.1f $^{\circ}$)"%Zenith)
plt.grid()
plt.legend()
plt.show()
sys.exit()
