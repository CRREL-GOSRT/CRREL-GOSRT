"""
This is an example of how to run the PhotonTrack module to get the optical properties
(extinction coefficient, phase function, and ice path fraction) from the snow sample mesh using ray tracing.
The full descriptions of inputs for the RayTracing_OpticalProperties function can be found in PhotonTrack.
"""

import file_pathnames as fp
from crrelGOSRT import PhotonTrack
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt

#%% Run photon-tracking model to get sample optical properties

# Define voxel resolution and wavelength of light used in computing optical properties
wavelen='1000nm'
VoxelRes='19.88250um'  # voxel resolution, usually provided in microCT log file


# Output filename
OutputFile = 'Optical_Properties.txt'
fullMeshName = 'CRREL_MESH.vtk'

# File definitions (shouldn't have to change)
VTKFilename = os.path.join(fp.VTK_DATA_OUTPATH,fullMeshName)
GrainPath = os.path.join(fp.VTK_DATA_OUTPATH,'GRAINS','')
OutputName = os.path.join(fp.OPT_PROP_OUTPATH,OutputFile)

start = dt.now()
# Compute optical properties
fig=PhotonTrack.RayTracing_OpticalProperties(VTKFilename,GrainPath,OutputName,fp.MATERIAL_PATH,wavelen,VoxelRes,
                                         verbose=True,nPhotons=1200,GrainSamples=30,Advanced=True,maxBounce=150,
                                         phaseSmooth=0,PhaseBins=180,particlePhase=True,raylen='auto',
                                         PF_fromSegmentedParticles=False,maxTIR=35,Tolerance=0.001)
end = dt.now()
print("Execution time :", str(end-start))

# Save figure
fig.savefig(os.path.join(fp.OPT_PROP_OUTPATH,'OptProps.png'),dpi=120)
plt.show()
