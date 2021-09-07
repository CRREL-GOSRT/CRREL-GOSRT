# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 16:04:42 2021

@author: RDCRLJTP
"""

from file_pathnames import *

import sys
# sys.path.append('C:\\Users\\RDCRLJTP\\Documents\\Projects\\Snow_Optics\\Code\\CRREL-GOSRT\\main')
import ImageSeg
import Utilities as util
from main import PhotonTrack
import glob
import os
from matplotlib import pyplot as plt
import numpy as np
from math import sin, cos, radians
import pyvista as pv
import matplotlib.pyplot as plt

#%% MicroCT Data Processing
# Find directory with microCT binary images
MCT_PATH = util.directory_find(MCT_IMAGE_PATH,'Snow')

# Set data parameters for sub-sampling and mesh generation
XYstart = 8.0 # The starting point in XY for the mesh subset within a sample image (in plane view) in millimeters, assuming the the left most pixel is 0.
XYend = 12.0 # The ending point in XY for the mesh subset within a sample image (in plane view) in millimeters, assuming the the left most pixel is 0.
depthTop = 300 # Top snow depth of scanned sample (in mm)
Ztop = 295.0  # Top depth selected for mesh sample subset
allowedBorder = 10000000  # Number of points allowed to be on a mesh border
minpoints = 25  # Minimum number of points allowed for each grain
minGrainSize = 0.4 # This sets the minimum grainsize (in mm) for the peak-local-max function
voxelRes=19.88250/1000. ## in millimeters, given in microCT log file
decimate = 0.9 # The decimal percentage of mesh triangles to eliminate
fullMeshName='CRREL_MESH.vtk' ## Name of FULL Mesh .VTK file. (i.e., mesh created by aggregating all the grains)


# Read MicroCT data
SNOW, X, Y, Z, origin_mm = ImageSeg.ImagesToArray(MCT_PATH,VTK_DATA_OUTPATH,XYstart,XYend,depthTop,Ztop,voxelRes,thresh=0.9,savePropsToCsv=True,saveMeshParams=True)

# Perform grain segmentation
grains, grain_labels, properties = ImageSeg.GrainSeg(SNOW,voxelRes,minGrainSize,VTK_DATA_OUTPATH,thresh=0.9,saveMeshParams=True)

# Generate mesh
ImageSeg.MeshGen(grains,grain_labels,properties,voxelRes,X,Y,Z,origin_mm,allowedBorder,minpoints,decimate,VTK_DATA_OUTPATH,fullMeshName,saveMeshParams=True,savePropsToCsv=True,check=False,create_individual=True)

#%% Run photon-tracking model to get sample optical properties

# Define voxel resolution and wavelength of light used in computing optical properties
wavelen='900nm'
VoxelRes='19.88250um'

# Output filename
OutputFile = 'Optical_Properties_updated.txt'

# File definitions (shouldn't have to change)
VTKFilename = os.path.join(VTK_DATA_OUTPATH,fullMeshName)
GrainPath = os.path.join(VTK_DATA_OUTPATH,'GRAINS','')
OutputName = os.path.join(OPT_PROP_OUTPATH,OutputFile)

# Compute optical properties
fig,pf,thetas,dtheta=PhotonTrack.RayTracing_OpticalProperties(VTKFilename,GrainPath,OutputName,MATERIAL_PATH,wavelen,VoxelRes,
                                         verbose=True,nPhotons=500,Multi=False,GrainSamples=30)

# Save figure
fig.savefig(os.path.join(OPT_PROP_OUTPATH,'OptProps.png'),dpi=90)
plt.show()
