# CRREL-GOSRT
CRREL Geometric Optics Snow Radiative Transfer (GOSRT) Model.

## Recent Tasks

- [x] Initial Code Packaging with __init__.py and setup.py files
- [x] Reorganization of subfolders to match standard syntax (e.g., main -> crrelGOSRT and Examples -> bin)
- [ ] test.py
- [ ] Usable README.md file
- [ ] added imageSegJulie.py to "bin" folder


## PhotonTrack model usage

```
from crrelGOSRT import PhotonTrack

parentPath='/Path/to/VTK/Mesh/CRREL_MESH.vtk'
GrainPath='/Path/to/VTK/Mesh/GRAINS/'
OutputName='/Path/to/OutputFile/OpticalProp.txt'
MaterialPath = '/Path/to/Material/csv/'

wavelen='800nm'
VoxelRes='19.88250um'

fig=PhotonTrack.RayTracing_OpticalProperties(parentPath,GrainPath,OutputName,MaterialPath,wavelen,VoxelRes)
```

## 1D (Slab) model usage

```
from crrelGOSRT import SlabModel

Slab=SlabModel.SlabModel()
Slab.Initialize()

Wavelength = range(500,1000,50)
Albedo, Absorption,Transmiss,transmissionDict=Slab.GetSpectralAlbedo(WaveLength,Zenith,Azi,nPhotons=nPhotons)
```
