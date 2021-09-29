## Continuous Medium Photon-Tracking Model ##
import sys
import numpy as np
from matplotlib import pyplot as plt
import glob as glob
from crrelGOSRT import SlabModel, Utilities


## Initialize Slab Model
Slab=SlabModel.SlabModel()

## Set Depth (in mm)
depth=300 ## depth in mm
## Define a range of WaveLengths to use in GetSpectralAlbedo()
WaveLength=np.arange(400,1600,25)

## Specify the number of photons to use.  Typically 10,000 is sufficient for accuracy, more should (in theory) be more accurate,
## but SIGNIFICANTLY increases Computational Expense
nPhotons=25000

## Use the Utilities module to get a nice visual of the visible spectrum to add to the figure (optional)
cols=Utilities.WaveLengthToColor(WaveLength, gamma=0.8)

## define the figure.
plt.figure(figsize=(10,10))
ax=plt.subplot(111)

## Output the data to this path.
OutputFilePath='/Folder/to/write/out/data/'

## Here is where you need to modify the script to make sure it matches your file paths.
Slab.namelistDict['MasterPath']='YOUR/PATH/HERE/'
SnowTypes=['FineGrain_OpticalProps.txt','Facets_OpticalProps.txt']
Slab.namelistDict['LayerTops']=[depth,0] ## Single layer model, top is depth, bottom is zero.

Azi,Zenith=65.,60.
## loop over each snow type, and tun the spectral model.
for sdx, s in enumerate(SnowTypes):
    Slab.namelistDict['PropFileNames']=[s]
    Slab.Initialize() ## Initialize the model (i.e., encode the namelist dictionary into hard variables.)

    Albedo, Absorption,Transmiss,transmissionDict=Slab.GetSpectralAlbedo(WaveLength,Zenith,Azi,nPhotons=nPhotons) # RUN THE MODEL!

    ## WRITE OUTPUT TO FILE ##
    Slab.WriteSpectralToFile('%s/%s_%scm.txt'%(OutputFilePath,s,depth),
        nPhotons,Zenith,Azi,WaveLength,Albedo,Absorption,Transmiss,filename='%s_cm Single Layer Large'%dep)

    ## PLOT THE DATA! ##
    plt.plot(WaveLength,Albedo,lw='2',label="%s"%s)

## SHOW VISIBLE SPECTRUM!
for idx,i in enumerate(WaveLength):
    plt.scatter(i,0.2,color=cols[idx],marker='s',edgecolor='k')

## Add some formatting to the graph.
plt.xlabel("WaveLength (nm)")
plt.ylabel("albedo")
plt.title("Albedo (SZA= %.1f $^{\circ}$)"%Zenith)
plt.grid()
plt.legend()
plt.show()
sys.exit()
