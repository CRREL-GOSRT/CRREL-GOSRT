===========
CRREL-GOSRT
===========

The Cold Regions Research and Engineering Lab Geometric Optics Snow Radiative Transfer model (CRREL-GOSRT)
is a unique snow radiative transfer model that combines 3D microCT snow renderings with ray-tracing and a generalized photon-tracking model to simulate snow
optical properties and snow albedo, transmissivity, and absorption at wavelengths between 350 - 1350nm (visible and NIR). Typical usage
often looks like this::

    #!/usr/bin/env python

    from crrelGOSRT import SlabModel, PhotonTrack


    Slab=SlabModel.SlabModel(namelist='namelist.txt')
    Slab.Initialize()

    WaveLength=np.arange(400,1300,50)
    Zenith=60
    Azi=0
    nPhotons=10000
    Albedo, Absorption,Transmiss,transmissionDict=Slab.GetSpectralAlbedo(WaveLength,Zenith,Azi,nPhotons=nPhotons)


Paragraphs are separated by blank lines. *Italics*, **bold**,
and ``monospace`` look like this.

Publications
=========

Publications

* Letcher et al., 2022: A generalized photon-tracking approach to simulate spectral snow
albedo and transmissivity using X-ray microtomography and
geometric optics, The Cryosphere | accepted

* Parno et al., 2021: A Blended Approach Toward Simulating Spectral Snow Reflectivity and
                      Transmissivity Using Monte Carlo Photon-Tracking and
                      X-ray Microtomography Surface Rendering, AGU Fall Meeting, 2021

Model Components
-------------

Numbered lists look like you'd expect:

1. Generate 3D rendering of snow micro structure (not incorporated in the model)

2. Compute optical properties from microCT rendering for use in 1D Slab Spectral Albedo Model

3. Simulate Snow Albedo, Transmissivity, Absorption using 1D photon tracking model with optical properties computed from step 2.

`GitHub Link <hhttps://github.com/CRREL-GSORT/CRREL-GOSRT>`_.
