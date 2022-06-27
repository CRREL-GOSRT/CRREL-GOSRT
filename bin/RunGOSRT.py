# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 16:04:42 2021

@author: RDCRLJTP
"""
import sys
from crrelGOSRT import ImageSeg
from crrelGOSRT import Utilities as util
from crrelGOSRT import PhotonTrack
import os
from matplotlib import pyplot as plt
import numpy as np
from math import sin, cos, radians
import pyvista as pv
import matplotlib.pyplot as plt
import file_pathnames as fp



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


Latitude = 43.8163
Longitude = -72.2740
Time = '02-12 15:35'
Elevation = 553
TimeFormat='%m-%d %H:%M'

Azimuth,Zenith=GetZenith(Time,Latitude,Longitude,Elevation,TimeFormat)

#%% MicroCT Data Processing
# Find directory with microCT binary images
MCT_PATH = util.directory_find(fp.MCT_IMAGE_PATH,'Snow')


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
fullMeshName='CRREL_MESH.vtk' ## Name of FULL Mesh .VTK file. (i.e., mesh created by aggregating all the grains)


# Read MicroCT data
SNOW, grid = ImageSeg.ImagesToArray(fp.MCT_IMAGE_PATH,fp.VTK_DATA_OUTPATH,XYstart,XYend,depthTop,Ztop,voxelRes)

# Perform grain segmentation
grains, grain_labels, properties = ImageSeg.GrainSeg(SNOW,voxelRes,minGrainSize,fp.VTK_DATA_OUTPATH)

# Generate mesh
ImageSeg.MeshGen(grains,grain_labels,properties,voxelRes,grid,allowedBorder,minpoints,decimate,fp.VTK_DATA_OUTPATH,fullMeshName,check=False)



sys.exit()
#%% Run photon-tracking model to get sample optical properties

# Define voxel resolution and wavelength of light used in computing optical properties
wavelen='900nm'
VoxelRes='19.88250um'


# Output filename
OutputFile = 'Optical_Properties_updated2.txt'
fullMeshName='SphereMesh_025mm_8mm3.vtk'

# File definitions (shouldn't have to change)
VTKFilename = os.path.join(fp.VTK_DATA_OUTPATH,fullMeshName)
GrainPath = os.path.join(fp.VTK_DATA_OUTPATH,'GRAINS','')
OutputName = os.path.join(fp.OPT_PROP_OUTPATH,OutputFile)

# Compute optical properties
fig=PhotonTrack.RayTracing_OpticalProperties(VTKFilename,GrainPath,OutputName,fp.MATERIAL_PATH,wavelen,VoxelRes,
                                         verbose=True,nPhotons=3500,Multi=False,GrainSamples=30,Advanced=True,
                                         FiceFromDensity=False,straight=False,maxBounce=120,particlePhase=False)

# Save figure
fig.savefig(os.path.join(fp.OPT_PROP_OUTPATH,'OptProps.png'),dpi=90)
plt.show()

#%% Run Slab Model

import SlabModel

Slab=SlabModel.SlabModel(namelist='Mynamelist.txt')
Slab.Initialize()
Azi,Zenith=Slab.GetZenith()

#
Albedo,Absorption,Transmiss,transmissionDict=Slab.GetSpectralAlbedo(WaveLength,Zenith,Azi,nPhotons=10000)
