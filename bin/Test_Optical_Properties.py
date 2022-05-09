## test photon track! ##
import sys
#sys.path.append('/Users/rdcrltwl/Desktop/CRRELRTM/CRREL-GOSRT/crrelGOSRT/')
from crrelGOSRT import PhotonTrack
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os

pwd=os.getcwd()

name='Spheres_0.6_212.0.vtk'

#parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/Feb12/VTK/Pit1_8_1redo/Snow/FULL/CRREL_MESH.vtk'
parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/CRREL_Mar22/compacted_top_Rec/FULL/CRREL_MESH.vtk'
GrainPath='/Users/rdcrltwl/Desktop/UVD_microCT/Feb12/VTK/Pit1_10_1/Snow/GRAINS_Top/'
#OutputName='/Users/rdcrltwl/Desktop/ESC_2021/OpticalProps/Facets/Facets_Props/SLEET_Example1.txt'

#parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/Feb12/VTK/Pit1_10_3redo/Snow/FULL_Bottom/CRREL_MESH.vtk'
#GrainPath='/Users/rdcrltwl/Desktop/UVD_microCT/Feb12/VTK/Pit1_8_1redo/Snow/FULL/Test_Phase/'
OutputName='/Users/rdcrltwl/Desktop/Circles/Outputs/Compact.txt'
#parentPath='/Users/rdcrltwl/Desktop/Circles/%s'%name
GrainPath='/Users/rdcrltwl/Desktop/PhaseFunctions/Spheres/'

#parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/Feb12/VTK/Pit1_10_3redo/Snow/FULL_Bottom/CRREL_MESH.vtk'

#parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/CRREL Snow/CRREL_20Feb21/CRREL_newsnowf_20Feb21/VTK/newsnowf_20Feb21_17um_Rec/VOI/Snow/FULL/CRREL_MESH.vtk'
#GrainPath='/Users/rdcrltwl/Desktop/UVD_microCT/CRREL Snow/CRREL_20Feb21/CRREL_newsnowf_20Feb21/VTK/newsnowf_20Feb21_17um_Rec/VOI/Snow/GRAINS/'
#OutputName='/Users/rdcrltwl/Desktop/CRRELRTM/OpticalProperties/CRREL_newsnowf_20Feb21/TestProps.txt'
MaterialPath = '/Users/rdcrltwl/Desktop/NewRTM/crrel-snow-rtm/Materials/'
WaveLength=np.arange(400,1600,25)
#WaveLength=['%inm'%i for i in WaveLength]
WaveLength='1000nm'
VoxelRes='19.88250um'

fig=PhotonTrack.RayTracing_OpticalProperties(parentPath,GrainPath,OutputName,MaterialPath,WaveLength,VoxelRes,
                                         verbose=True,nPhotons=2000,Multi=False,GrainSamples=150,Advanced=True,
                                         FiceFromDensity=False,straight=False,maxBounce=150,phaseSmooth=0,PhaseBins=180,
                                         particlePhase=False,AirOnly=False)

fig.savefig('/Users/rdcrltwl/Desktop/Compacted_Grains.png',dpi=120)
plt.show()
