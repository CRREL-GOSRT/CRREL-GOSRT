## test photon track! ##
import sys
from crrelGOSRT import PhotonTrack
import numpy as np
from math import sin, cos, radians
import pyvista as pv
import matplotlib.pyplot as plt
import os

pwd=os.getcwd()

parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/Feb12/VTK/Pit1_12_3/Snow/FULL/CRREL_MESH.vtk'
GrainPath='/Users/rdcrltwl/Desktop/UVD_microCT/Feb12/VTK/Pit1_12_3/Snow/GRAINS/'
OutputName='/Users/rdcrltwl/Desktop/ESC_2021/OpticalProps/Facets/Facets_Props/OpticalProp_Coarse.txt'

#parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/CRREL Snow/CRREL_20Feb21/CRREL_newsnowf_20Feb21/VTK/newsnowf_20Feb21_17um_Rec/VOI/Snow/FULL/CRREL_MESH.vtk'
#GrainPath='/Users/rdcrltwl/Desktop/UVD_microCT/CRREL Snow/CRREL_20Feb21/CRREL_newsnowf_20Feb21/VTK/newsnowf_20Feb21_17um_Rec/VOI/Snow/GRAINS/'
#OutputName='/Users/rdcrltwl/Desktop/CRRELRTM/OpticalProperties/CRREL_newsnowf_20Feb21/TestProps.txt'
MaterialPath = '/Users/rdcrltwl/Desktop/NewRTM/crrel-snow-rtm/Materials/'
wavelen='800nm'
VoxelRes='19.88250um'

fig=PhotonTrack.RayTracing_OpticalProperties(parentPath,GrainPath,OutputName,MaterialPath,wavelen,VoxelRes,
                                         verbose=True,nPhotons=2500,Multi=False,GrainSamples=25,Advanced=True)

fig.savefig('/Users/rdcrltwl/Desktop/ESC_2021/OpticalProps/Facets/Facets_Props/OpticalProp_Coarse.png',dpi=90)
plt.show()
