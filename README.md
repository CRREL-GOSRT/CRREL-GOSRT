# CRREL-GOSRT
CRREL Geometric Optics Snow Radiative Transfer (GOSRT) Model.

![Four Panel Figure showing optical properties of a uCT sample from CRREL-GOSRT](bin/SampleData/Facets/CoarseGrain_4Panel.png?raw=true "Facets")

## Introduction

CRREL-GOSRT is a radiative transfer model that simulates radiative transfer through snow using photon tracking and 3D renderings of uCT
meshes.  The model operates within the geometric optics approximation (400-1400nm or so) and is a two step model that 1) uses photon tracking to estimate the
optical properties (extinction, absorption, scattering) from a closed manifold surface rendering of a uCT sample and 2) simulates the spectral albedo and BRDF using a 1D photon tracking medium model forced with those optical properties. This model has able to handle multiple layers with unique optical properties and is capable of simulating RT through snow packs with arbitrary depths and can simulate spectral reflectance off of a specified lower boundary.


## Recent Tasks

- [x] Initial Code Packaging with __init__.py and setup.py files
- [x] Reorganization of subfolders to match standard syntax (e.g., main -> crrelGOSRT and Examples -> bin)
- [x] Usable README.md file
- [x] added imageSegJulie.py to "bin" folder
- [ ] Clean code for publication
- [x] Add sample data and update paper figures to match.


## PhotonTrack model usage

```python
from crrelGOSRT import PhotonTrack
from matplotlib import pyplot as plt

parentPath='/Path/to/VTK/Mesh/CRREL_MESH.vtk'
GrainPath='/Path/to/VTK/Mesh/GRAINS/'
OutputName='/Path/to/OutputFile/OpticalProp.txt'
MaterialPath = '/Path/to/Material/csv/'

wavelen='800nm'
VoxelRes='19.88250um'

fig=PhotonTrack.RayTracing_OpticalProperties(parentPath,GrainPath,OutputName,
                                             MaterialPath,wavelen,VoxelRes)
plt.show()

```

## 1D (Slab) model usage

```python
from crrelGOSRT import SlabModel
from matplotlib import pyplot as plt

Slab=SlabModel.SlabModel()
Slab.Initialize()
## Get Zenith and Azimuth angle based on lat/lon/elevation and time in namelist using "solarposition"
Azi,Zenith=Slab.GetZenith()

Zenith=60
Azi=0

Wavelength = range(500,1000,50)
Albedo, Absorption,Transmiss,transDict=Slab.GetSpectralAlbedo(WaveLength,Zenith,
                                                              Azi,nPhotons=1000)

fig=plt.figure()
ax=plt.subplot(111)
ax.plot(WaveLength,Albedo)
ax.set_xlabel('WaveLength (nm)')
ax.set_ylabel('Albedo')

```

## Citations

Theodore Letcher, Julie Parno, Zoe Courville, Lauren Farnsworth, Jason Olivier, A generalized photon-tracking approach to simulate spectral snow albedo and transmissivity using X-ray microtomography and geometric optics, The Cryosphere.  In Review.

Citation for Solar Position:
Ibrahim Reda, Afshin Andreas, Solar position algorithm for solar radiation applications, Solar Energy, Volume 76, Issue 5, 2004, Pages 577-589, ISSN 0038-092X, http://dx.doi.org/10.1016/j.solener.2003.12.003.
Keywords: Global solar irradiance; Solar zenith angle; Solar azimuth angle; VSOP87 theory; Universal time; ΔUT1
