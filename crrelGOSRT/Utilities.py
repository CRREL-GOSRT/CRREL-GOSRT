# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:58:56 2021

@author: RDCRLJTP
"""

import os
import numpy as np

# search for Snow directory tool
def directory_find(root, word):
    for path, dirs, files in os.walk(root):
        if word in dirs:
            return os.path.join(path, word)


def PlankFunction(wavelen,T=5778.):
    """Plank Function
        Wavelength must be in meters"""

    c1=1.191042E8
    c2=1.4387752E4
    L=c1/(wavelen**5*(np.exp(c2/(wavelen*T))-1))
    return L

def WaveLengthToColor(WaveLength, gamma=0.8):
    '''This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''
    cols=[]

    for i in WaveLength:
        wavelength = float(i)
        if wavelength >= 380 and wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
            G = 0.0
            B = (1.0 * attenuation) ** gamma
        elif wavelength >= 440 and wavelength <= 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440)) ** gamma
            B = 1.0
        elif wavelength >= 490 and wavelength <= 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490)) ** gamma
        elif wavelength >= 510 and wavelength <= 580:
            R = ((wavelength - 510) / (580 - 510)) ** gamma
            G = 1.0
            B = 0.0
        elif wavelength >= 580 and wavelength <= 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580)) ** gamma
            B = 0.0
        elif wavelength >= 645 and wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation) ** gamma
            G = 0.0
            B = 0.0
        else:
            R = 0.6
            G = 0.6
            B = 0.6

        cols.append((R,G,B))
    return cols
