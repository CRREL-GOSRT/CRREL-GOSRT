## test photon track! ##
import sys
sys.path.append('C:\\Users\\RDCRLJTP\\Documents\\Projects\\Snow_Optics\\Code\\CRREL-GOSRT')
from main import PhotonTrack
import numpy as np
from math import sin, cos, radians
import pyvista as pv
import matplotlib.pyplot as plt
import os

pwd=os.getcwd()

# parentPath='/Users/rdcrltwl/Desktop/UVD_microCT/1Feb_UVD_Pit1/Sample_1_20-13cm/VTK/1Feb_UVD_Pit1_1_2/Snow/FULL/CRREL_MESH.vtk'
parentPath ='C:\\Users\\RDCRLJTP\\Documents\\Projects\\Snow_Optics\\MicroCT_Data\\12Feb_UVD\\VTK\\contour\\Pit1_9_1\\'
VTKFilename = os.path.join(parentPath,'CRREL_MESH.vtk')
# GrainPath='/Users/rdcrltwl/Desktop/UVD_microCT/1Feb_UVD_Pit1/Sample_1_20-13cm/VTK/1Feb_UVD_Pit1_1_2/Snow/GRAINS/'
GrainPath = os.path.join(parentPath,'GRAINS','')
# OutputName='/Users/rdcrltwl/Desktop/UVD_microCT/1Feb_UVD_Pit1/Sample_1_20-13cm/VTK/1Feb_UVD_Pit1_1_2/Snow/FULL/TestProps.txt'
OutputName = os.path.join(parentPath,'Optical_Properties_updated.txt')

# MaterialPath = '/Users/rdcrltwl/Desktop/NewRTM/crrel-snow-rtm/Materials/'
MaterialPath = 'C:\\Users\\RDCRLJTP\\Documents\\Projects\\Snow_Optics\\Code\\NewRTM\\Materials\\'
wavelen='900nm'
VoxelRes='19.88250um'

fig,pf,thetas,dtheta=PhotonTrack.RayTracing_OpticalProperties(VTKFilename,GrainPath,OutputName,MaterialPath,wavelen,VoxelRes,
                                         verbose=True,nPhotons=500,Multi=False,GrainSamples=30)

# fig.savefig('/Users/rdcrltwl/Desktop/OptProps.png',dpi=90)
fig.savefig(os.path.join(parentPath,'OptProps.png'),dpi=90)
plt.show()
