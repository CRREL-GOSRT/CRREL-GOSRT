## test photon track! ##
import sys
sys.path.append("/Users/rdcrltwl/Desktop/CRRELRTM")
from main import PhotonTrack
import numpy as np
from math import sin, cos, radians
import pyvista as pv
import matplotlib.pyplot as plt
import os

pwd=os.getcwd()

parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/1Feb_UVD_Pit1/Sample_1_20-13cm/VTK/1Feb_UVD_Pit1_1_2/Snow/FULL/CRREL_MESH.vtk'
GrainPath='/Users/rdcrltwl/Desktop/UVD_microCT/1Feb_UVD_Pit1/Sample_1_20-13cm/VTK/1Feb_UVD_Pit1_1_2/Snow/GRAINS/'
OutputName='/Users/rdcrltwl/Desktop/UVD_microCT/1Feb_UVD_Pit1/Sample_1_20-13cm/VTK/1Feb_UVD_Pit1_1_2/Snow/FULL/TestProps.txt'

MaterialPath = '/Users/rdcrltwl/Desktop/NewRTM/crrel-snow-rtm/Materials/'
wavelen='800nm'
VoxelRes='19.88250um'

fig=PhotonTrack.RayTracing_OpticalProperties(parentPath,GrainPath,OutputName,MaterialPath,wavelen,VoxelRes,
                                         verbose=True,nPhotons=500,Multi=False,GrainSamples=20)

fig.savefig('/Users/rdcrltwl/Desktop/OptProps.png',dpi=90)
plt.show()
