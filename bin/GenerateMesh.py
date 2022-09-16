# -*- coding: utf-8 -*-
"""
This is an example of how to generate a mesh from microCT data (binarized image stack) for input into
the CRREL-GOSRT model. This script uses functions in ImageSeg.py to convert the binarized MicroCT image stacks
to a numpy array, perform snow grain segmentation, and generate a mesh of both individual snow grains
and the full snow sample.

Note that this requires the user to create a file_pathnames.py script with their local paths using
file_pathnames_example.py as a template.
"""

import file_pathnames as fp
from crrelGOSRT import ImageSeg
from crrelGOSRT import Utilities as util


### User input required ###
# Set data parameters for sub-sampling and mesh generation
XYstart = 5.0 # The starting point in XY for the mesh subset within a sample image (in plane view) in millimeters, assuming the the left most pixel is 0.
XYend = 10.0 # The ending point in XY for the mesh subset within a sample image (in plane view) in millimeters, assuming the the left most pixel is 0.
depthTop = 320 # Top snow depth of scanned sample (in mm)
Ztop = 315  # Top depth selected for mesh sample subset
allowedBorder = 10000000  # Number of points allowed to be on a mesh border
minpoints = 25  # Minimum number of points allowed for each grain
minGrainSize = 0.3 # This sets the minimum grainsize (in mm) for the peak-local-max function
voxelRes=19.88250/1000. ## in millimeters, given in microCT log file
decimate = 0.9 # The decimal percentage of mesh triangles to eliminate
fullMeshName='MESH.vtk' ## Name of FULL Mesh .VTK file. (i.e., mesh created by aggregating all the grains)
###########################

# Find directory with microCT binary images (change search term according to your file naming system)
MCT_PATH = util.directory_find(fp.MCT_IMAGE_PATH,'Snow')

# Read MicroCT data
SNOW, grid = ImageSeg.ImagesToArray(fp.MCT_IMAGE_PATH,fp.VTK_DATA_OUTPATH,XYstart,XYend,depthTop,Ztop,voxelRes)

# Perform grain segmentation
grains, grain_labels, properties = ImageSeg.GrainSeg(SNOW,voxelRes,minGrainSize,fp.VTK_DATA_OUTPATH)

# Generate mesh
ImageSeg.MeshGen(grains,grain_labels,properties,voxelRes,grid,allowedBorder,minpoints,decimate,fp.VTK_DATA_OUTPATH,fullMeshName,check=False)
