## test photon track! ##
import sys
#sys.path.append('/Users/rdcrltwl/Desktop/CRRELRTM/CRREL-GOSRT/crrelGOSRT/')
from crrelGOSRT import PhotonTrack
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import os

pwd=os.getcwd()

parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/Feb12/VTK/Pit1_8_4/Snow/FULL/CRREL_MESH.vtk'
GrainPath='/Users/rdcrltwl/Desktop/UVD_microCT/Feb12/VTK/Pit1_8_4/Snow/GRAINS/'
#parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/Feb12/VTK/Pit1_10_1/Snow/FULL_Top/CRREL_MESH.vtk'
#GrainPath='/Users/rdcrltwl/Desktop/UVD_microCT/Feb12/VTK/Pit1_10_1/Snow/GRAINS_Top/'
OutputName='/Users/rdcrltwl/Desktop/ESC_2021/OpticalProps/Facets/Facets_Props/SLEET_Example1.txt'
#OutputName='/Users/rdcrltwl/Desktop/ESC_2021/OpticalProps/Virgin_Snow/29_27.txt'

#parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/CRREL Snow/CRREL_20Feb21/CRREL_newsnowf_20Feb21/VTK/newsnowf_20Feb21_17um_Rec/VOI/Snow/FULL/CRREL_MESH.vtk'
#GrainPath='/Users/rdcrltwl/Desktop/UVD_microCT/CRREL Snow/CRREL_20Feb21/CRREL_newsnowf_20Feb21/VTK/newsnowf_20Feb21_17um_Rec/VOI/Snow/GRAINS/'
#OutputName='/Users/rdcrltwl/Desktop/CRRELRTM/OpticalProperties/CRREL_newsnowf_20Feb21/TestProps.txt'
MaterialPath = '/Users/rdcrltwl/Desktop/NewRTM/crrel-snow-rtm/Materials/'
wavelen='900nm'
VoxelRes='19.88250um'

fig=PhotonTrack.RayTracing_OpticalProperties(parentPath,GrainPath,OutputName,MaterialPath,wavelen,VoxelRes,
                                         verbose=True,nPhotons=3500,Multi=False,GrainSamples=45,Advanced=True,
                                         FiceFromDensity=False,straight=False,maxBounce=120)

fig.savefig('/Users/rdcrltwl/Desktop/ESC_2021/OpticalProps/Virgin_Snow/CRUST2.png',dpi=90)
plt.show()
