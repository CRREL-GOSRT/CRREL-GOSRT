# CRREL-GOSRT
CRREL Geometric Optics Snow Radiative Transfer (GOSRT) Model.

![Four Panel Figure showing optical properties of a uCT sample from CRREL-GOSRT](bin/SampleData/Facets/CoarseGrain_4Panel.png?raw=true "Facets")

## Introduction

CRREL-GOSRT is a radiative transfer model that simulates radiative transfer through snow using photon tracking and 3D renderings of μCT
meshes.  The model operates within the geometric optics approximation (400-1400nm or so) and is a two step model that 1) uses photon tracking to estimate the
optical properties (extinction, absorption, scattering) from a closed manifold surface rendering of a μCT sample and 2) simulates the spectral albedo and BRDF using a 1D photon tracking medium model forced with those optical properties. This model has able to handle multiple layers with unique optical properties and is capable of simulating RT through snow packs with arbitrary depths and can simulate spectral reflectance off of a specified lower boundary.

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

Zenith=60
Azi=0

Wavelength = range(500,1000,50)
Albedo, Absorption,Transmiss,transDict=Slab.GetSpectralAlbedo(WaveLength,Zenith,
                                                              Azi,nPhotons=1000)

## Write data out to file ##
Slab.WriteSpectralToFile('./TestOutput.txt',
      1000,Zenith,Azi,WaveLength,Albedo,Absorption,Transmiss,filename='Test Output')

## Plot a figure ##
fig=plt.figure()
ax=plt.subplot(111)
ax.plot(WaveLength,Albedo)
ax.set_xlabel('WaveLength (nm)')
ax.set_ylabel('Albedo')
plt.show()
```

## Use with "solarposition" get get Zenith/Azimuth angle from latlon:

Here is a helpful function that uses the "solarposition" package ([https://github.com/s-bear/sun-position.git](https://github.com/s-bear/sun-position.git))
to get the azimuth and zenith angle for a specific location and time.

```python
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

Latitude = 44.11
Longitude = -73.92
Time = '02-12 15:35'
Elevation = 1628
TimeFormat='%m-%d %H:%M'

Azimuth,Zenith=GetZenith(Time,Latitude,Longitude,Elevation,TimeFormat)
```

## Citations

**Citation for CRREL-GOSRT:**

* Theodore Letcher, Julie Parno, Zoe Courville, Lauren Farnsworth, Jason Olivier, A generalized photon-tracking approach to simulate spectral snow albedo and transmissivity using X-ray microtomography and geometric optics, The Cryosphere.  *Submitted 09/2021: In Review*.

**Citation for Solar Position:**

* Ibrahim Reda, Afshin Andreas, Solar position algorithm for solar radiation applications, Solar Energy, Volume 76, Issue 5, 2004, Pages 577-589, ISSN 0038-092X, http://dx.doi.org/10.1016/j.solener.2003.12.003.
Keywords: Global solar irradiance; Solar zenith angle; Solar azimuth angle; VSOP87 theory; Universal time; ΔUT1
