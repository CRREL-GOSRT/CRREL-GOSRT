===========
CRREL-GOSRT
===========

Materials :: The following .csv files contain refractive index (R=n+mi) information for a few different materials.
The file format is structured as three columns (wave,Re,Im) corresponding to:

Wavelength (um) ,Real Part,  Imaginary Part

Mostly, you will reference the ice_Ri material in the RTM.  These optical constants are published in Warren and Brant, 2008* and available here:

https://atmos.uw.edu/ice_optical_constants/

* Warren, S. G., and R. E. Brandt (2008), Optical constants of ice from the ultraviolet to the microwave: A revised compilation.
J. Geophys. Res., 113, D14220, doi:10.1029/2007JD009744

The remaining materials are adapted from data available here: https://refractiveindex.info, which includes reference sources.  These files are being used to test the very much developmental "LAP" contaminant module in the model.  Not recommended for use in research without additional development and validation.
