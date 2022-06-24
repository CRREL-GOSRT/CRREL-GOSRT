import sys
import numpy as np
import pandas as pd
import glob as glob
from crrelGOSRT import BRDFFunctions as BRDFFunc
import os

## For debug only!
from matplotlib import pyplot as plt

class SlabModel:
    """SlabModel is the main class for the CRREL Photon Tracking Snow Geometric Optics RTM.

        Version 0.2.1 --> Replaced version 0.2.1 September 2021

            Key 0.2.0 -> 0.2.1 Changes:
                - Updated the namelist dictionary to remove references to LayerPaths to cut down on superfluous folder paths, replaced layer paths with file names
                    e.g., instead of layerPath1/Properties.txt, layerPath2/Properties.txt
                          we have MasterPath/Properties1.txt, MasterPath/Properties2.txt
                - PropFileNames replaced PropFileName in the namelist dictionary and is counted as a list (see above bullet)
                - Updated string concatentation for paths with os.path.join() for more robust path setting
                - Moved WaveLengthToColor function into new "Utilities" code.
                - Finally added the PrintSlabProps() function to the model
                - Added clearSurface() function to clear any surface data
                - Added self.__initialized flag to the model class to indicate whether or not the model has been initialized and add some added checks in other code
                - Other general code cleaning / comments.

            Key  0.1.0 -> 0.2.0 Changes:
                - Replaced standard model integration with optimized (vectorized) structure
                - Replaced the standard medium absorption coefficient with "ice-fraction"
                    to explicity calculate energy absorption during integration
                - Replaced wavelength variable extinction coefficient and phase function is assumed
                    constant across the VNIR range.  Greatly simplifies model code, relies on the fact that
                    real part of the ice-refractive index is generally constant in this spectral range.
                - Added feature to include light-absorbing-particles (LAPS) in the model framework
                - Added feature to include a semi-reflective lower boundary
                - Added documentation to model functions and cleaned up code

        This model is designed to track photon energy as it propagates through a combined snow air medium
        It allows for multiple layers with the following properties:
            1. Extinction coefficient
            2. Phase Function
            3. Fraction of distance traveled within the ice medium

        This model also requires a file containing the refractive indicies of ice to set the
        ice absorption coefficients.  Recommend pull from Warren, S. G., and R. E. Brandt (2008):
            https://atmos.uw.edu/ice_optical_constants/

        Accompanying sub-routines with this model provide a means to determine these properties from
        three-dimensional renderings of snow derived from Micro Tomography (MicroCT) scans of snow using
        geometric optics following the methods of Kaempfer et al. 2007.

        This model can have multiple layers and incorperate a surface lower boundary condition.
        It is designed for the investigation of trasmissivity and reflectivity of snow within a snowpack
        and is limited to the geometric optics approximation (i.e., wavelength < ~1500 nm).

        As of Early 2021, there was no treatment of diffraction, contaminates, or mixed snow/water phases.
        Perhaps in the future, such features will be added.

        To Run SlabModel, it must be imported as a stand-alone object within an external python script, e.g.,

            import SlabModel
            Model = SlabModel.SlabModel()

        Once imported, the model can be initialized.  Note that, prior to running SlabModel,
        a text file (default named Properties.txt), must first be created for every layer prescribed wthin the
        namelist, or intrinsic namelist dictionary.  This file contains the required snow properties to run the
        slab model, and is automatically generated when running the accompanying SetSlabProps.py script on MicroCT data.

        Model initialization reads in the namelist dictionary / file and cross-checks to make sure property files exist, etc.
        Continuing with the above example:

            Model.Initialize()

        Once initalized, the following functions perform various simulations with the model:

        1. RunBRDF() --> Estimates the surface BRDF for a given wavelength and incident ray dirction
        2. GetSpectralAlbedo() --> Computes spectral albedo for a range of wavelenghts and a given incident ray direction
        3. PhotonTrace() --> A simple function that fires a handful of photons into the medium such
            that their position can be tracked
        4. GetZenith() --> A function that reads in lat/lon/time from the namelist and
            returns a solar Zenith and Azimuth Angle
        5. WriteSpectralToFile() --> A function that writes out model output from GetSpectralAlbedo()
            to a .txt file with model parametersfrom the namelist dictionary along with other function inputs.
        6. loadSurface() --> Function that adds a lower boundary to the model object that will determine
            how radiation reaching the model bottom is reflected/absorbed
        7. clearSurface() --> Clears surface data and removes lower boundary.
        8. PrintSlabProps() --> Prints out current slab information from the properties file!

        Most other functions are private, called as part of the model routines and should not be accessed by the user.

    """


    def __init__(self,namelist='namelist.txt'):
        """
           Inititialization Function that calls a namelist, reads it in, and configures model layering
        """

        ## read namelist function here, but it doesn't exist yet. ##

        self.namelistDict={'MasterPath':'/Users/rdcrltwl/Desktop/SnowOpticsProject/MicroCTData/micro-CT/VTK/',
                           'LayerTops':[150],
                           'DepthUnits':'mm',
                           'WaveUnits':'nm',
                           'PropFileNames':['Properties.txt'],
                           'Fsoot':[0.0],
                           'Contaminate':'diesel',
                           'PhaseFunc':2,
                           'MaterialPath':'/Users/rdcrltwl/Desktop/NewRTM/Materials/',
                           'Asymmetry':[0.87],
                           'PhaseSamples':10000,
                           'RussianRouletteThresh':0.01,
                           'RussianRouletteWeight':6,
                           'xSize':10,
                           'ySize':10,
                           'DiffuseFraction':0,
                           'Absorption':1,
                           'Extinction':1
                           }

        cwd=os.getcwd()
        self.__initialized =False
        self.namelistPath=os.path.join(cwd,namelist)
        if os.path.isfile(self.namelistPath) and os.access(self.namelistPath, os.R_OK):
            print("Initializing Namelist Dictionary (self.namelistDict) from %s"%self.namelistPath)
            print("Namelist File Exists...")
            self.__ParseNamelist()
        else:
            print("The namelist file %s does not exist, using default values for self.namelistDict"%self.namelistPath)
            print("To make changes to the model parameters, please access the namelist dictionary inline.")


        ## Now that your namelist is set, first check the layers for consistency and add a "zero" to
        ## the bottom of the layer if one doesn't exist yet.
        print("Finished initializing model with namelist parameters and options")
        print("To intialize the slab model with these parameters use the initalize function:")
        print("-->  Slab.Initialize()")
        print("!!!! -------------------- !!!!")
        print("            WARNING       !!!!")
        print("!!!! -------------------- !!!!")
        print("Any changes to the model parameters must be made in the namelist dictionary: (namelistDict)")
        print("e.g., Slab.namelistDict['LayerTops']=[50] will change the top of the model layer to 50mm")
        print("For these changes to take place, the model must be reinitialized before they take effect!")
        print("!!!! -------------------- !!!!")
        print("")
        print("!!!! -------------------- !!!!")
        return

    def Initialize(self):

        """
            Key initialization function that reads in the namelist data, performs some critical checks,
            must be performed after any changes to the namelist dictionary are set inline in order to for these
            changes to take effect.
        """

        print("---------")
        print("Initializing CRREL Slab RTM!")
        print("---------")

        ## Here we define the allowed units for one to use for snow depth, in the end, everything is
        ## converted to millimeters to match the extinction coefficients computed from the
        ## explicit ray-tracing part of the code that sets the properties.

        ## We also set the allowed wavelength units, which are then converted to nanometers.
        AllowedUnits = {'mm':1.,'millimeters':1.,'meters':1000.,'m':1000.,'cm':10.,'centimeters':10.,
                        'in':24.5,'inches':25.4}

        AllowedWLUnits=['m','cm','mm','um','nm']

        if self.namelistDict['WaveUnits'] not in AllowedWLUnits:
            self.Warning("Specified Depth Units not in Allowed Units! --> Defaulting to (nm)")
            print("Allowed Units are: ")
            for i in AllowedUnits.keys():
                print('-- %s'%i)
            self.namelistDict['WaveUnits'] = 'nm'

        if self.namelistDict['DepthUnits'] not in AllowedUnits.keys():
            self.Warning("Specified Depth Units not in Allowed Units! --> Defaulting to (mm)")
            print("Allowed Units are: ")
            for i in AllowedUnits.keys():
                print('-- %s'%i)
            self.namelistDict['DepthUnits'] = 'mm'


        ## We then check and see if the person added a 0 to the bottom layer,
        ## if not, add it for them.
        if self.namelistDict['LayerTops'][-1] != 0:
            self.namelistDict['LayerTops'].append(0.0)

        ## Hold over from some older code, but I think it works well here,
        ## define the medium material as Ice, this helps set the absorbtion coefficient for the Material
        ## From the complex part of the refractive index.  Note this function reads in a csv file that has the
        ## refractive index data in it and saves the data to the "self" object.
        self.AssignMaterial('ice',filePath=self.namelistDict['MaterialPath'])

        ## Now, compare length of various options for consistency. ##
        self.NumLayers=len(self.namelistDict['LayerTops'])-1 ## Number of layers.

        ## need to check and make sure the length of the layerpaths is correct.
        if len(self.namelistDict['PropFileNames']) != self.NumLayers:
            self.FatalError("The number of Layer Paths is not equal to the number of layers!")

        ## Check to make sure that PhaseFunc is either 1/2 to set how the phase function / photon scattering
        ## direction is determined
        if int(self.namelistDict['PhaseFunc']) not in [1,2]:
            self.Warning("This Phase Function option (%s) is not allowed! --> Defaulting to 1 (Henyey-Greenstein approximation)"%self.namelistDict['PhaseFunc'])
            self.namelistDict['PhaseFunc']=1

        if int(self.namelistDict['PhaseFunc']) == 1:
            ## if using a simplfied HeyNey Greenstein Phase Function, need to make sure that the
            ## Length of the asymmetry parameters matches the number of layers
            if len(self.namelistDict['Asymmetry']) != self.NumLayers:
                self.FatalError("The number of Asymmetry parameters is not equal to the number of layers!")

        ## Now need to find the property files in the master folder!
        self.__PropFiles=[glob.glob(os.path.join(self.namelistDict['MasterPath'],i))[0] for i in self.namelistDict['PropFileNames']]

        ## now that everything is set, fix units of depth to be millimeters!
        ## this matches the assumed units of the extintion and absorption coefficients
        self.__layerTops=np.array(self.namelistDict['LayerTops'])*AllowedUnits[self.namelistDict['DepthUnits']]
        self.__layerIds=[i for i in range(self.NumLayers)]

        ## now set the data for the exitinction/absorption coefficients ##
        ## Note, need to be data frames with wavelength in them. ##
        ## Also, set the phase function if using a single value for all wave lengths
        ##       (as will be most common, when not using idealized).
        ## Also, note that if using a wavelength dependent phase function or idealized phase function,
        ## these dictionarys are simply set to "None" and the phase functions are computed in the actual model
        ## integration functions.

        ## Now that the phase function is set, need to set the coefficients
        ## Note that SOOT is unique from Fice, in that it is determined in the namelist, not the properties file,
        ## this may change in future versions.

        ## If Absorption == 2, then assume using an absorbtion coefficient, require a supplementary file
        ## with wavelength dependent Kext and Kabs.

        if self.namelistDict['Absorption'] == 2:
            self.__AbsorbtionMethod = 2
            print("Using Medium Absorption Coefficient instead of F-ice to simulate absorption")
            print("Note, this requires supplementary files that have WaveLength dependent values for")
            print("Kext and Kabs.")

            ## First check for files! , will crash if there are no files here, that's a good thing.
            self.__SuppFiles=[glob.glob(os.path.join(self.namelistDict['MasterPath'],i.split('.txt')[0]+'_ExtAbso.txt'))[0] for i in self.namelistDict['PropFileNames']]
            ## NOW WE have a LIST of supplementary files.

            ## NOW WE WANT arrays OF Wavelengths, Kabs and Kext to go with EACH layer.
            ## INITIALIZE THESE Dictionarys here, all with the prefix "supp" to indicate supplementary
            self.__SuppExtCoeffDict={}
            self.__SuppKabsCoeffDict={}
            self.__SuppWavelengthDict={}

            for idx, i in enumerate(self.__SuppFiles):
                SuppData=pd.read_csv(i,header=1)
                self.__SuppExtCoeffDict[self.__layerIds[idx]]=SuppData['Extinction'][:]
                self.__SuppKabsCoeffDict[self.__layerIds[idx]]=SuppData['Absorption'][:]
                self.__SuppWavelengthDict[self.__layerIds[idx]]=SuppData['wavelength (nm)'][:]
                SuppData=None ## Clear the memory here.

            ## NOW WE HAVE THE NEEDED LISTS TO USE IF Absorption == 2.
        else:
            ## Otherwise, just set it to 1.

            if self.namelistDict['Extinction'] == 2:
                ## Just grab the extinction dictionary.
                self.__ExtinctionMethod = 2
                print("Using Wavelength-variable extinction coefficient.")
                print("Note, this requires supplementary files that have WaveLength dependent values for Kext")
                self.__SuppFiles=[glob.glob(os.path.join(self.namelistDict['MasterPath'],i.split('.txt')[0]+'_ExtAbso.txt'))[0] for i in self.namelistDict['PropFileNames']]
                ## NOW WE have a LIST of supplementary files.
                self.__SuppExtCoeffDict={}
                self.__SuppWavelengthDict={}
                for idx, i in enumerate(self.__SuppFiles):
                    SuppData=pd.read_csv(i,header=1)
                    self.__SuppExtCoeffDict[self.__layerIds[idx]]=SuppData['Extinction'][:]
                    self.__SuppWavelengthDict[self.__layerIds[idx]]=SuppData['wavelength (nm)'][:]
                    SuppData=None ## Clear the memory here.
            else:
                self.__ExtinctionMethod = 1

            self.__AbsorbtionMethod = 1

        self.__ExtCoeffDict={}
        self.__FiceDict={}

        self.__SootDict={}
        headerLines=[]
        for idx, i in enumerate(self.namelistDict['PropFileNames']):
            txtdata = open(self.__PropFiles[idx], 'r')
            Lines = txtdata.readlines()

            for cdx, c in enumerate(Lines):
                if c.strip() == '':
                    continue
                if 'Extinction Coefficient' in c:
                    self.__ExtCoeffDict[self.__layerIds[idx]] = float(c.split('=')[-1].split('(')[0])
                if '(Fice)' in c and 'Mean' in c:
                    self.__FiceDict[self.__layerIds[idx]] = float(c.split('=')[-1])
                if 'Theta (radians)' in c:
                    headerLines.append(cdx)

            ## Close the file and clear the memory within "Lines"
            txtdata.close()
            Lines=None
            self.__SootDict[self.__layerIds[idx]]=self.namelistDict['Fsoot'][idx]

        ## This is a fairly involved function that sets the probabilistic Scattering
        ## direction based on a binned phase-function.
        ## Note that a dictionay called __CosIs is saved with a number of random samples
        ## that are weighted by the phase-function probability distrubtion
        ## In the model integration, the scattering direction is pulled from this dictionary.
        if int(self.namelistDict['PhaseFunc']) == 2:
            print("Setting Phase Function for Each Layer ...")
            print("These functions will be used for all wave lenghths!")
            self.__PhaseDict={}
            self.__ThetaDict={}
            self.__CosIs={}
            self.__PhaseAsym={}

            for idx, i in enumerate(self.namelistDict['PropFileNames']):
                data=pd.read_csv(self.__PropFiles[idx],header=headerLines[idx])
                self.__ThetaDict[self.__layerIds[idx]] = data['Theta (radians)'].values
                self.__PhaseDict[self.__layerIds[idx]] = data[' Phase Function'].values

                nBins=len(self.__ThetaDict[self.__layerIds[idx]])

                dtheta=np.nanmean(np.abs(np.gradient(self.__ThetaDict[self.__layerIds[idx]])))

                probs=(self.__PhaseDict[self.__layerIds[idx]]*2*np.pi*np.sin(self.__ThetaDict[self.__layerIds[idx]])*dtheta)/(4.*np.pi)

                self.nSamples=int(self.namelistDict['PhaseSamples'])
                self.__CosIs[self.__layerIds[idx]]=self.__PhaseSamples(self.__ThetaDict[self.__layerIds[idx]],probs,self.nSamples,dtheta)

        else:
            ## Otherwise, No need to define these since you're using a Henyey-Greenstein function
            self.__PhaseDict=None
            self.__ThetaDict=None
            self.__CosIs=None
            self.__PhaseAsym=None

        ## These are okay to set allow the user to access, so no need to keep these private.
        self.rrConst=float(self.namelistDict['RussianRouletteWeight'])
        self.rrThresh=float(self.namelistDict['RussianRouletteThresh'])

        self.xSize=float(self.namelistDict['xSize'])
        self.ySize=float(self.namelistDict['ySize'])
        self.__slabDepth=np.max(self.__layerTops)
        self.__Surface = False

        ## in nm, model will not allow for, and will remove any wavelengths outside of this boundary
        ## This could be modified by the user, but I wouldn't recommend it.
        ## in future versions, this could be tied to the input refractive index files for soot/ice.
        self.WaveLengthLimits=[150,1600]

        ### We initialize all of the namelist variables as distinct attributes
        ### To ensure that only namelist values pass though the initialization function
        ### Are used in the actual model!

        self.__DiffuseFraction=float(self.namelistDict['DiffuseFraction'])

        if self.__DiffuseFraction < 0:
            self.__DiffuseFraction = 0.0
        if self.__DiffuseFraction > 1:
            self.__DiffuseFraction = 1.0

        ## Now initialize the lat/lon/time
        self.__initialized = True

        print("Finished initializing Model parameters")
        return

    def __RussianRoulette(self,Photons):
        """
            Russian Routlette function that helps remove photon packets with little
            remaining energy, while still preserving overall average energy conservation

            Note that the required input variables are contained within
            the model structure and are the energy threshold for which to trigger the
            russian Routlette function, and the constant value used to cull photons.
            Returns a -999 to stand in for all dead photons, these are removed within the main model.
        """
        dead=np.where(Photons < self.rrThresh)
        absorbed=0
        for ddx in dead[0]:
            rnum=np.random.uniform(0.,1.0)
            if rnum  <= 1./self.rrConst:
                Photons[ddx]=self.rrConst*Photons[ddx]
            else:
                absorbed+=Photons[ddx]
                Photons[ddx]=-999.

        return Photons,absorbed


    def __SetDirection(self,cosU,Scatter):

        """
            Critical sub-function that sets the NEW direction for the photon packet
            based on the input direction and an array representing the new directions for
            each photon packet.
        """

        ## PHASE FUNCTION TIME!
        azi=2.*np.pi*np.random.uniform(0.,1.,len(Scatter))
        sinI=np.sqrt(1.0-Scatter**2.)

        CosNew=np.zeros_like(cosU)
 #      ## New X direction
        CosNew[0,:]=((sinI*(cosU[0,:]*cosU[2,:]*np.cos(azi)-cosU[1,:]*np.sin(azi)))/
                (np.sqrt(1.-cosU[2,:]**2.))+cosU[0,:]*Scatter)

        ## New Y direction
        CosNew[1,:]=((sinI*(cosU[1,:]*cosU[2,:]*np.cos(azi)+cosU[0,:]*np.sin(azi)))/
                (np.sqrt(1.-cosU[2,:]**2.))+cosU[1,:]*Scatter)

        ## New Z direction
        CosNew[2,:]=-np.sqrt(1.-cosU[2,:]**2.)*sinI*np.cos(azi)+cosU[2,:]*Scatter

        ## Special contingency: mu_z is too close to either 1 or negative 1.
        mask1=np.ma.masked_inside(cosU[2,:],0.99,1.0).mask
        mask2=np.ma.masked_inside(cosU[2,:],-1.0,-0.99).mask

        CosNew[0,:]=np.ma.masked_array(CosNew[0,:],mask=mask1).filled(sinI*np.cos(azi))
        CosNew[1,:]=np.ma.masked_array(CosNew[1,:],mask=mask1).filled(sinI*np.sin(azi))
        CosNew[2,:]=np.ma.masked_array(CosNew[2,:],mask=mask1).filled(Scatter)

        CosNew[0,:]=np.ma.masked_array(CosNew[0,:],mask=mask2).filled(sinI*np.cos(azi))
        CosNew[1,:]=np.ma.masked_array(CosNew[1,:],mask=mask2).filled(-sinI*np.sin(azi))
        CosNew[2,:]=np.ma.masked_array(CosNew[2,:],mask=mask2).filled(-Scatter)
        ## Returns the new direction!
        return CosNew[:]


    def __ChangeLayers(self,u0,u1,currentBin,layerIds,Fice,extCoeff,Fsoot,
                    ExtDict,IceDict,SootDict,g,Scatter,ProbIdx):
        """
            Critical function that handles the transition from one layer to another

            Inputs:
                u0 = current position vector
                u1 = future position vector
                currentBin = array containing the physical height of the lower/upper boundary of
                    each layer for each photon
                layerIds = array with number indentification indicating which layer the photon is currently in
                Fice = ice-fraction applied to each photon based on it's current layer
                extCoeff = extintion coefficient applied to each photon based on it's current layer
                Fsoot = soot or dust fraction applied to medium
                ExtDict = dictionary of extiction coefficients corresponding to each unique layer
                IceDict = dictionary of ice-fraction corresponding to each unique layer
                SootDict = dictionary of soot fraction corresponding to each unique layer
                g       = Asymmetry parameter used in idealized HeyNey-Greenstein function
                Scatter = scattering direction for photon travel based on phase-function
                ProbIdx = Indicies used to pull random values from scattering direction array
        """

        ## Now we pluck out photons that changed layers! ##
        ## Photons moving from lower to upper layer first! ##

        ## Find indices where photons are moving from lower bin to upper bin.
        UpperMovIdx=np.where((u1[2,:] > currentBin[0,:]) & (u0[2,:] < currentBin[0,:]))
        ## Layer ids increase from top-to-bottom layer, so lower index corresponds to upper layers
        layerIds[UpperMovIdx] = layerIds[UpperMovIdx]-1 ## Subtract Index!
        fact=(u1[2,UpperMovIdx]-currentBin[0,UpperMovIdx])/(u1[2,UpperMovIdx]-u0[2,UpperMovIdx])

        ## Old check to make sure we aren't somehow getting a greater than 1 factor.
        if len(fact) > 1:
            self.Warning("Upper Move max factor = %.1f"%np.max(fact))

        ## interpolate Fice and Fsoot to weighted averaged between layers.
        Fice[UpperMovIdx]=(np.array([IceDict[i] for i in layerIds[UpperMovIdx]])*fact
                        + (1.-fact)*Fice[UpperMovIdx])

        Fsoot[UpperMovIdx]=(np.array([SootDict[i] for i in layerIds[UpperMovIdx]])*fact
                        + (1.-fact)*Fsoot[UpperMovIdx])

        ## No need to interpolate the extinction coefficient, since we assume that the "scattering event"
        ## already occurred within the new layer.
        extCoeff[UpperMovIdx]=np.array([ExtDict[i] for i in layerIds[UpperMovIdx]])

        ## Interpolate phase function, as a weighted averaged as well.
        if self.namelistDict['PhaseFunc'] == 1:
            rand=np.random.uniform(0.,1.,size=len(g[UpperMovIdx]))
            g[UpperMovIdx]=(np.array([self.namelistDict['Asymmetry'][i] for i in layerIds[UpperMovIdx]])* fact +
                        + (1.-fact)*g[UpperMovIdx])
            Scatter[UpperMovIdx]=(1./(2.*g[UpperMovIdx]))*(1.+g[UpperMovIdx]**2.-((1-g[UpperMovIdx]**2.)/(1.-g[UpperMovIdx]+2.*g[UpperMovIdx]*rand))**2.)
        else:
            ## This is a stubbornly complex process that requies looping through the layer ids and
            ## selecting new phase function properties from the __CosIs samples before averaging
            ScatterNew=np.zeros_like(Scatter)
            for i in self.__layerIds:
                ScatIdx=np.where(layerIds == i)
                ScatterNew[ScatIdx]=(self.__CosIs[i][ProbIdx])[ScatIdx]
            Scatter[UpperMovIdx] = (ScatterNew[UpperMovIdx]*fact
                            + (1.-fact)*Scatter[UpperMovIdx])

        ## Photons moving from the upper to the lower layers ##
        ## Basically, do the same process as above, but for lower indicies
        ## I won't repeat comments here, see above.
        LowerMovIdx=np.where((u1[2,:] < currentBin[1,:]) & (u0[2,:] > currentBin[1,:]))

        layerIds[LowerMovIdx] = layerIds[LowerMovIdx]+1 ## Subtract Index!
        fact=(u1[2,LowerMovIdx]-currentBin[1,LowerMovIdx])/(u1[2,LowerMovIdx]-u0[2,LowerMovIdx])

        mask=np.ma.masked_where(fact < 1,(u1[2,LowerMovIdx]-currentBin[1,LowerMovIdx])/(u1[2,LowerMovIdx]-u0[2,LowerMovIdx]))
        #print("Lower",np.max(fact),np.ma.masked_array(currentBin[1,LowerMovIdx],mask=mask.mask).compressed(),
        #            np.ma.masked_array(u1[2,LowerMovIdx],mask=mask.mask).compressed())

        if len(fact) > 1:
            self.Warning("Lower Move max factor = %.1f"%np.max(fact))

        Fice[LowerMovIdx]=(np.array([IceDict[i] for i in layerIds[LowerMovIdx]])*fact
                        + (1.-fact)*Fice[LowerMovIdx])

        Fsoot[LowerMovIdx]=(np.array([SootDict[i] for i in layerIds[LowerMovIdx]])*fact
                        + (1.-fact)*Fsoot[LowerMovIdx])

        extCoeff[LowerMovIdx]=np.array([ExtDict[i] for i in layerIds[LowerMovIdx]])

        if self.namelistDict['PhaseFunc'] == 1:
            rand=np.random.uniform(0.,1.,size=len(g[LowerMovIdx]))
            g[LowerMovIdx]=(np.array([self.namelistDict['Asymmetry'][i] for i in layerIds[LowerMovIdx]])* fact +
                        + (1.-fact)*g[LowerMovIdx])
            Scatter[LowerMovIdx]=(1./(2.*g[LowerMovIdx]))*(1.+g[LowerMovIdx]**2.-((1-g[LowerMovIdx]**2.)/(1.-g[LowerMovIdx]+2.*g[LowerMovIdx]*rand))**2.)
        else:
            ScatterNew=np.zeros_like(Scatter)
            for i in self.__layerIds:
                ScatIdx=np.where(layerIds == i)
                ScatterNew[ScatIdx]=(self.__CosIs[i][ProbIdx])[ScatIdx]
            Scatter[LowerMovIdx] = (ScatterNew[LowerMovIdx]*fact
                            + (1.-fact)*Scatter[LowerMovIdx])

        if np.max(np.abs(Scatter)) > 1:
            self.FatalError("Scattering Phase Function Greater than 1!!!! Max = %.1f"%np.max(np.abs(Scatter)))

        ## Retuns LayerIds to set next iteration, also the unique scattering phase function and ext/abs coefficients
        return layerIds, Scatter, Fice, extCoeff, Fsoot


    def PrintSlabProps(self):
        """This function prints out to the terminal screen, the slab physical and optical properties from each snow layer"""

        print("-------------------------------------------------------------------------")
        print("-----------          Printing Out Slab Properties            ------------")
        print("-------------------------------------------------------------------------")

        print("Initialized? %s"%str(self.__initialized))
        print("")
        print("Filepaths:")
        print(" Namelist file path: %s"%self.namelistPath)
        print(" Material Path: %s"%self.namelistDict['MaterialPath'])
        print(" Properties Filepath: %s"%self.namelistDict['MasterPath'])
        print("")

        print("Namelist Options:")
        for i in self.namelistDict.keys():
            if i not in ['MaterialPath','MasterPath']:
                print(" %s: %s"%(i,str(self.namelistDict[i])))

        print("")

        if self.__initialized == True:
            print("Curremnt Model Configuration")
            print("** Note, if this does not match the namelist, then reinialize the model prior to running **")

            print("Total Snow Depth: %s (%s)"%(self.__slabDepth,self.namelistDict['DepthUnits']))
            print("Snow Layer Tops (%s)"%self.namelistDict['DepthUnits'])
            for i in range(len(self.__layerIds)):
                print("  Id = %s | z = %s"%(self.__layerIds[i],self.__layerTops[i]))

            print("Optical Property Files")
            for i in range(len(self.__layerIds)):
                print("  Id = %s | File = %s"%(self.__layerIds[i],self.__PropFiles[i]))


            print("Extinction Coefficient/Fice/Fsoot")
            for i in self.__layerIds:
                print("  Id = %s | ext = %s | Fice = %s | Fsoot = %s"%(i,self.__ExtCoeffDict[i],self.__FiceDict[i],self.__SootDict[i]))

            print("Diffuse Fraction %s"%self.__DiffuseFraction)

            print("Surface? %s"%str(self.__Surface))
            print("")

            if self.__Surface == True:
                print("Surface Properties")
                print(" BRDF: %s"%self.__SfcBRDF)
                for i in self.BRDFParams.keys():
                    print("%s: %s"%(i,self.BRDFParams[i]))


        return


    def RunBRDF(self,WaveLength,Zenith,Azimuth,nPhotons=10000,binSize=10,angleUnits='Degrees'):

        """ User accessible function to estimate the Bidirectional Reflectance distribution Fuction (BRDF)
            following Kaempfer et al. 2007.

            Inputs:
                WaveLength = wavelength of light entering the medium (in nm)
                Zenith angle = incident zenith angle
                Azimuth Angle = incident azimuth angle (counter-clockwise from east)
                nPhotons (optional) = number of photons to launch into the medium
                binSize (optional) = size of angular bin overwhich to compute the BRDF
                angleUnits (optional) = Units of input angle, either degrees or radians.

            Retuns:
                BRDFArray = 2D array containing the BRDF information
                BRAziBins = Azimuthal bin centroid
                BRZenBins = Zenith bin centroid
                albedo = overall reflected fraction
                absorbed = overall absorbed fraction
                transmiss = overall transmitted fraction
        """

        if self.__initialized == False:
            print("The Model has not been initalized.  You must initialize this model before running 'RunBRDF'. Exiting.")
            print(" --> mod.Initialize()")
            sys.exit()

        AllowedAngleUnits=['deg','rad','degrees','radians']
        if angleUnits.lower() not in AllowedAngleUnits:
            self.Warning("The angle unit %s is not allowed, assuming your input angles are in degrees"%angleUnits)
            angleUnits='degrees'

        if not self.WaveLengthLimits[0] <= WaveLength < self.WaveLengthLimits[1]:
            self.FatalError("The Wavelength %s is outside of the allowed list of wavelengths!"%WaveLength)

        ## Get absorption coefficients for the specified wavelength for ice and any contaminates

        RI,n,kappaIce=self.GetRefractiveIndex(WaveLength,units='nm')
        self.AssignMaterial(self.namelistDict['Contaminate'],filePath=self.namelistDict['MaterialPath'])
        Rsoot,nsoot,kappaSoot=self.GetRefractiveIndex(WaveLength,units='nm')
        self.AssignMaterial('ice',filePath=self.namelistDict['MaterialPath'])

        if angleUnits.lower() in ['deg','degrees']:
            ## if the units are degrees, convert to radians!
            ## fix azimuth to 0 on x, and rotate so the incident direction is aimed correctly.
            Azimuth=np.pi+np.radians(Azimuth)
            Zenith=np.radians(Zenith)

        if np.cos(Zenith) <=0:
            self.FatalError("The Zenith angle %.1f rad is below the horizon!  Cannot use for incident radiation!"%Zenith)

        cosUStart=[np.sin(Zenith)*np.cos(Azimuth),np.sin(Zenith)*np.sin(Azimuth),-np.cos(Zenith)]
        ## FOR BRDF STUFF ##
        angularArea = np.radians(binSize)**2.
        ## output comes in degrees!
        BRAziBins=np.arange(0,360+binSize,binSize)
        BRZenBins=np.arange(0,90+binSize, binSize)
        ## Stagger Zenith to center the bins! ##
        BRZenBins=(BRZenBins[:-1]+BRZenBins[1:])/2.
        BRDFArray=np.zeros([len(BRAziBins),len(BRZenBins)])

        ## initial position of the photons, random within x and y BOUNDARIES
        ## z position set to top of layer!
        xx = np.random.uniform(self.xSize,size=nPhotons)
        yy = np.random.uniform(self.ySize,size=nPhotons)
        zz = np.ones_like(xx)*self.__slabDepth

        ## Now, I need to set up coordinates for all the layer idenfication that matches
        ## The number of photons!
        ## Starting at top layer, so all layer ids start at zero!
        layerIds = np.zeros([nPhotons],dtype=np.short) ## short integer

        extCoeff = np.array([self.__ExtCoeffDict[i] for i in layerIds])
        Fice = np.array([self.__FiceDict[i] for i in layerIds])
        Fsoot = np.array([self.__SootDict[i] for i in layerIds])
        if self.namelistDict['PhaseFunc'] == 1: ## If using idealized Henyey Green Function!
            g=np.array([self.namelistDict['Asymmetry'][i] for i in layerIds])
        else:
            g=self.NumLayers*[0.9]

        ## Current bin defines the lower and upper boundaries of the bin!
        currentBin=np.array([[self.__layerTops[i],self.__layerTops[i+1]] for i in layerIds]).T

        Photons=np.ones([nPhotons],dtype=np.double)
        u0=np.array([xx,yy,zz])

        DiffusePhotons = int(nPhotons*self.__DiffuseFraction)
        DirectPhotons = nPhotons-DiffusePhotons

        randAzi=np.random.uniform(0.0,2.*np.pi,size=DiffusePhotons)
        randZen=np.random.uniform(0.0,np.pi/2.,size=DiffusePhotons)
        diffuse=[[np.sin(randZen[j])*np.cos(randAzi[j]),np.sin(randZen[j])*np.sin(randAzi[j]),-np.cos(randZen[j])] for j in range(DiffusePhotons)]

        cosU=np.array(DirectPhotons*[cosUStart]+diffuse).T

        iterNum=0.0
        print("Running %i Photons at Wavelength %s nm to compute BRDF"%(nPhotons,WaveLength))

        absorbed=0.0
        albedo=0.0
        transmiss=0.0
        ## Run the photon tracking model.
        while(len(Photons) > 0):

            ## Find out if we're killing any photons, and how much energy is absorbed
            Photons,abso=self.__RussianRoulette(Photons)
            absorbed+=abso ## add to absorbed total

            #Note, that at this point, no array sizes have been altered, but "dead" photons have a value of -999.

            ## Get the distance traveled
            rand=np.random.uniform(0.0,1.,size=len(Photons))
            rand=np.array(3*[rand])
            s=-np.log(rand)/extCoeff ## Distance!
            u1=u0+cosU*s ## New position!

            ## Where photons have gone out of the top of the snowpack!
            OutTop=np.ma.masked_where(u1[2,:] < self.__slabDepth, Photons)
            albedo+=np.sum(np.ma.masked_less(OutTop.compressed(),0).compressed()) ## Add all to albedo.
            ## note the mask below zero rejects all dead photons from the sum.

            ## If a surface boundary is included in the analysis.
            if self.__Surface == True:
                ## Surfaced with defined BRDF is at this level!
                SfcIdx=np.where(u1[2,:] <=0)[0] ## indicies where photons hit surface!

                ## We only need to enter this function if any photons hit the surface.
                if np.sum(Photons[SfcIdx]) > 0:
                    Indir=cosU[:,SfcIdx]

                    Zen=np.arccos(-Indir[2,:])
                    Azi=np.pi-np.arctan(Indir[1,:]/Indir[0,:])
                    OutDir=np.zeros_like(Indir)
                    ## SO FAR, it is recommended that only Lambert and Specular
                    ## options are used, since they can compute the reflected
                    ## direction as vectorized quantities
                    ## using a more complicated BRDF is too computationally expensive in accounting to
                    ## individual incident angles.  The said, the code is here to do that.
                    if self.__SfcBRDF == 'lambert':
                        outWeights=np.ones_like(Zen)*self.BRDFParams['albedo']
                        RandZen=np.random.uniform(0,1,size=len(Zen))*np.pi/2.
                        RandAzi=np.random.uniform(0,1,size=len(Zen))*np.pi*2.
                        OutDir[0,:]=np.cos(RandAzi)
                        OutDir[1,:]=np.sin(RandAzi)
                        OutDir[2,:]=np.cos(RandZen)
                    elif self.__SfcBRDF == 'specular':
                        outWeights=np.ones_like(Zen)*self.BRDFParams['albedo']
                        for kdx in range(Indir.shape[1]):
                            cosThetai =  -np.dot(Indir[:,kdx],np.array([0,0,-1]))
                            OutDir[:,kdx]= Indir[:,kdx] + 2. * cosThetai * np.array([0,0,-1])
                    else:
                        ## if using an explity BRDF for the surface.
                        ## again, recommended that this isn't used at the time.
                        outWeights=np.zeros([len(Zen)])
                        for kdx in range(len(Zen)):
                            HRDF, NewZen,NewPhi=BRDFFunc.BRDFProb(1,Zen[kdx],Azi[kdx],
                                self.__BRDFPhi,self.__BRDFTheta,self.__BRDFdTheta,
                                ParamDict=self.BRDFParams,BRDF=self.__SfcBRDF)

                            outWeights[kdx]=HRDF
                            OutDir[:,kdx]=[np.sin(NewZen)*np.cos(NewPhi),np.sin(NewZen)*np.sin(NewPhi),np.cos(NewZen)]

                    transmiss+=np.sum(np.ma.masked_less((1.-outWeights)*Photons[SfcIdx],0).compressed()) ## Add all to transmiss
                    Photons[SfcIdx]=outWeights*Photons[SfcIdx] ## reflectance!
                    u1[2,SfcIdx] = 0.00001 ## set position to surface
                    cosU[:,SfcIdx] = OutDir[:]


            ## This will track transmissivity out of the surface.
            OutBottom=np.ma.masked_where(u1[2,:] >=0, Photons) ## out of the bottom
            ## it also determines which photons to keep, removes all dead photons + photons that exit the top or bottom.
            keepIdx=np.squeeze([(u1[2,:] >= 0) & (u1[2,:] < self.__slabDepth) & (Photons >= 0)]) ## Which photons to keep!
            transmiss+=np.sum(np.ma.masked_less(OutBottom.compressed(),0).compressed()) ## Add all to transmiss

            ## PUT BRDF FUNCTIONS HERE! ##
            ## Get indicies where reflected.
            whereAlbedo=np.where(u1[2,:] > self.__slabDepth)
            BRPhoton=Photons[whereAlbedo]
            if np.sum(BRPhoton) > 0: ##Only take the trouble if this is true!
                BRDrct=cosU[:,whereAlbedo]
                yMag=np.sqrt(BRDrct[0,:]**2.+BRDrct[1,:]**2.)

                BRAziAngle=np.degrees(np.arctan(BRDrct[1,:]/BRDrct[0,:]))
                negX=np.where(BRDrct[0,:] < 0)
                BRAziAngle[negX]=BRAziAngle[negX]+180.
                BRAziAngle[BRAziAngle < 0] = 360.+BRAziAngle[BRAziAngle < 0]
                BRZenAngle=90.-np.degrees(np.arctan(BRDrct[2,:]/yMag))
                for aBdx in range(len(BRAziBins)):
                    ## First mask out everything outside of the Azimuth Bin, then do the Zeith Bin
                    AziBinMin=BRAziBins[aBdx]-binSize/2.
                    AziBinMax=BRAziBins[aBdx]+binSize/2.
                    if AziBinMax > 360:
                        xmask=~np.ma.masked_outside(BRAziAngle,AziBinMax-360.,AziBinMin).mask
                    elif AziBinMin < 0:
                        xmask=~np.ma.masked_outside(BRAziAngle,360-binSize/2.,AziBinMax).mask
                    else:
                        xmask=np.ma.masked_outside(BRAziAngle,AziBinMin,AziBinMax).mask
                    Zens=np.ma.masked_array(BRZenAngle,mask=xmask).compressed()
                    ZPhotons=np.ma.masked_array(BRPhoton,mask=xmask).compressed()
                    for zBdx in range(len(BRZenBins)):
                        zmask=np.ma.masked_outside(Zens,BRZenBins[zBdx]-binSize/2.,BRZenBins[zBdx]+binSize/2.).mask
                        BRDFArray[aBdx,zBdx]+=np.sum(np.ma.masked_less(
                                    np.ma.masked_array(ZPhotons,mask=zmask).compressed(),0).compressed())/(angularArea*np.cos(np.radians(BRZenBins[zBdx]))*np.sin(np.radians(BRZenBins[zBdx])))


            ## update all necessary arrays to keep only indexes that you need! ##
            ## Now start altering the size of the arrays to match
            ## to only the photons you want to keep for the next iteration.
            u1=u1[:,keepIdx]
            u0=u0[:,keepIdx]
            Photons=Photons[keepIdx]

            ## special case, if there is only 1 photon left, add to absorbed and kill it
            ## required to ensure that array lengths and indexing can work.
            ## as long as you launch more than 1000 photons, this shouldn't matter.
            if len(Photons) <= 1:
                absorbed+=np.sum(Photons)
                break
            cosU=cosU[:,keepIdx]
            layerIds=layerIds[keepIdx]
            s=s[0,keepIdx]
            ## Reset the current layer boundaries to match the current layer ids!
            currentBin=np.array([[self.__layerTops[i],self.__layerTops[i+1]] for i in layerIds]).T
            ## Reset coefficients to current ids!
            extCoeff=np.array([self.__ExtCoeffDict[i] for i in layerIds])
            Fice = np.array([self.__FiceDict[i] for i in layerIds])
            Fsoot = np.array([self.__SootDict[i] for i in layerIds])

            if self.namelistDict['PhaseFunc'] == 1: ## If using idealized Reset Phase Function Scattering
                g=np.array([self.namelistDict['Asymmetry'][i] for i in layerIds])

            ## okay, to recap, we have performed the russian roulette routine to handel dead PHOTONS
            ## Moved the photon through space following the direction vector and current position.
            ## updated albedo transmissivity based on whether photons have left the snowpack entirely
            ## removed all photons no long in the snowpack, and update the current boundaries and coefficients.

            ## Now need to do the phase function (it not idelized!)
            if self.namelistDict['PhaseFunc'] == 2: ## if it's not idealized!
                ProbIdx=np.random.randint(0,self.nSamples,size=len(Photons))
                Scatter=np.zeros([len(Photons)])
                ScatterNew=np.zeros_like(Scatter)
                for i in self.__layerIds:
                    ScatIdx=np.where(layerIds == i)
                    Scatter[ScatIdx]=(self.__CosIs[i][ProbIdx])[ScatIdx]
            else:
                rand=np.random.uniform(0,1.,size=len(Photons))
                Scatter=(1./(2.*g))*(1.+g**2.-((1-g**2.)/(1.-g+2.*g*rand))**2.)
                ProbIdx=0
            ## Change the layers if needed.
            layerIds, Scatter, Fice, extCoeff,Fsoot=self.__ChangeLayers(u0,u1,currentBin,layerIds,Fice,extCoeff,Fsoot,
                                                                self.__ExtCoeffDict,self.__FiceDict,self.__SootDict,
                                                                g,Scatter,ProbIdx)


            ## Absorb some photons! ##
            deltaW=(1.-np.exp(-s*(kappaIce*Fice+kappaSoot*Fsoot)))*Photons
            absorbed+=np.sum(deltaW)
            Photons=Photons-deltaW

            ## set new direction vector.
            cosU=self.__SetDirection(cosU,Scatter)
            u0[:]=u1[:]  ## set starting position to new position

            ## end loop --> go back up!
            ## break from loop once the size of the photon array is zero.

        return BRDFArray,BRAziBins,BRZenBins,albedo,absorbed,transmiss

    def GetSpectralAlbedo(self,WaveLength,Zenith,Azimuth,nPhotons=10000,
                          verbose = True,angleUnits='degrees',
                          transmission=None,TransCompute=10):
        """
            User accessible function to compute the spectral albedo of the medium from photon-tracking

            Inputs:
                WaveLength = wavelength of light entering the medium (in nm) --> Can be multiple values.
                Zenith angle = incident zenith angle
                Azimuth Angle = incident azimuth angle (counter-clockwise from east)
                nPhotons = number of photons to launch into the medium

                verbose (optional) = Turn on some additional printout text during model integration
                angleUnits (optional) = Units of input angle, either degrees or radians

                transmission (optional) = allows user to input a list, or 1D array of transmission depths
                    will track the fraction of energy that propagates deeper than each depth.
                    Note, this adds significant computational expense to the model due to the overall
                    number of integrations before all of the photon energy is removed from the snowpack
                    Recommend that this array is small in size (e.g., < 10) and that TransCompute is used to
                    reduce computational time
                TransCompute (optional) = Controls how over to perform transmission computation
                    when transmission is not equal to "None"  Will limit how often the model
                    computes transmitted Fraction, note that as long as the transmission depths are
                    sufficiently far away, even a relatively large number (>~20) won't impact the results
                    since the distances traveled are usually small.  A good rule of thumb would be:
                    N = Ddepth/(2*extCoeff), where N is "TransCompute" and Ddepth is the difference
                    between in transmission depths.

            Returns:
                albedo = overall reflected fraction
                absorbed = overall absorbed fraction
                transmiss = overall transmitted fraction
                transmissionPower = Transmission power for each depth in "transmission"
        """


        if self.__initialized == False:
            print("The Model has not been initalized.  You must initialize this model before running 'GetSpectralAlbedo'. Exiting.")
            print(" --> mod.Initialize()")
            sys.exit()

        WaveLength=np.array(WaveLength)

        ## Make sure WaveLength is in the limits
        WaveLength = np.ma.masked_outside(WaveLength,self.WaveLengthLimits[0],
                                          self.WaveLengthLimits[1]).compressed()

        if len(WaveLength) == 0:
            self.FatalError("Your WaveLength Array Length is Zero, you probably used a value outside of the allowed limits!")


        AllowedAngleUnits=['deg','rad','degrees','radians']
        if angleUnits.lower() not in AllowedAngleUnits:
            self.Warning("The angle unit %s is not allowed, assuming your input angles are in degrees"%angleUnits)
            angleUnits='degrees'

        if isinstance(transmission, type(None)):
            transmissionDict=None
            TrackTransmission=False
        else:
            TrackTransmission=True
            transmissionDict={}
            transmissionPower={}
            transmission=np.array(transmission)

        if angleUnits.lower() in ['deg','degrees']:
            ## if the units are degrees, convert to radians!
            ## fix azimuth to point in the direction of the inident radiation (i.e., where it's coming from!)
            Azimuth=np.pi-np.radians(Azimuth)
            Zenith=np.radians(Zenith)

        if np.cos(Zenith) <=0:
            self.FatalError("The Zenith angle %.1f rad is below the horizon!  Cannot use for incident radiation!"%Zenith)

        cosUStart=[np.sin(Zenith)*np.cos(Azimuth),np.sin(Zenith)*np.sin(Azimuth),-np.cos(Zenith)]

        print("Computing Spectral Albedo of this Snowpack From %.1f to %.1f"%(np.min(WaveLength),np.max(WaveLength)))

        SpectralAlbedo=[]
        SpectralAbsorption=[]
        SpectralTransmissivity=[]
        ## Loop through all wave lengths!
        ## Special function here that will allow you to track the transmitted energy
        ## at specified depths within the snowpack., more of a diagnostic.
        for wdx,wavelen in enumerate(WaveLength):
            ## set up transmission dictionary here for wavelength.
            if TrackTransmission == True:
                PhotonId=np.arange(nPhotons)
                transmissionDict[wavelen]={}
                transmissionPower[wavelen]={}
                for i in transmission:
                    transmissionDict[wavelen][i]=[]
                    transmissionPower[wavelen][i]=[]

            if verbose == True:
                print("Working on Wavelength %.1f nm"%wavelen)
            ## initial position of the photons, random within x and y BOUNDARIES
            ## z position set to top of layer!
            xx = np.random.uniform(self.xSize,size=nPhotons)
            yy = np.random.uniform(self.ySize,size=nPhotons)
            zz = np.ones_like(xx)*self.__slabDepth

            RI,n,kappaIce=self.GetRefractiveIndex(wavelen,units='nm')
            self.AssignMaterial(self.namelistDict['Contaminate'],filePath=self.namelistDict['MaterialPath'])
            Rsoot,nsoot,kappaSoot=self.GetRefractiveIndex(wavelen,units='nm')
            self.AssignMaterial('ice',filePath=self.namelistDict['MaterialPath'])
            ## Now, I need to set up coordinates for all the layer idenfication that matches
            ## The number of photons!
            ## Starting at top layer, so all layer ids start at zero!
            layerIds=np.zeros([nPhotons],dtype=np.short) ## short integer

            if self.__AbsorbtionMethod == 2:
                ##NOTE HERE THAT WE ARE RESETTING THE DICTIONARYS FOR F_ICE AND EXT TO MATCH THE
                ##CURRENT WaveLength!!!
                for i in self.__layerIds:
                    SubbWave=self.__SuppWavelengthDict[i]
                    AbsArray=self.__SuppKabsCoeffDict[i]
                    self.__FiceDict[i] = np.interp(wavelen,SubbWave,AbsArray)

            if self.__ExtinctionMethod == 2:
                for i in self.__layerIds:
                    SubbWave=self.__SuppWavelengthDict[i]
                    ExtArray=self.__SuppExtCoeffDict[i]
                    self.__ExtCoeffDict[i] = np.interp(wavelen,SubbWave,ExtArray)


            extCoeff=np.array([self.__ExtCoeffDict[i] for i in layerIds])
            Fice = np.array([self.__FiceDict[i] for i in layerIds])
            Fsoot = np.array([self.__SootDict[i] for i in layerIds])


            if self.namelistDict['PhaseFunc'] == 1: ## If using idealized HeyNey Green Function!
                g=np.array([self.namelistDict['Asymmetry'][i] for i in layerIds])
            else:
                g=self.NumLayers*[0.87]

            ## Current bin defines the lower and upper boundaries of the bin!
            currentBin=np.array([[self.__layerTops[i],self.__layerTops[i+1]] for i in layerIds]).T

            Photons=np.ones([nPhotons],dtype=np.double)
            u0=np.array([xx,yy,zz])
            DiffusePhotons = int(nPhotons*self.__DiffuseFraction)
            DirectPhotons = nPhotons-DiffusePhotons

            TotalPathLength=np.zeros_like(Photons)

            randAzi=np.random.uniform(0.0,2.*np.pi,size=DiffusePhotons)
            randZen=np.random.uniform(0.0,np.pi/2.,size=DiffusePhotons)
            diffuse=[[np.sin(randZen[j])*np.cos(randAzi[j]),np.sin(randZen[j])*np.sin(randAzi[j]),-np.cos(randZen[j])] for j in range(DiffusePhotons)]

            cosU=np.array(DirectPhotons*[cosUStart]+diffuse).T

            iterNum=0.0

            absorbed=0.0
            albedo=0.0
            transmiss=0.0
            iterNum=1
            TotalPathVals=[]
            while(len(Photons) > 0):
                Photons,abso=self.__RussianRoulette(Photons)
                absorbed+=abso

                ## Get Direction!
                rand=np.random.uniform(0.0,1.,size=len(Photons))
                rand=np.array(3*[rand])
                s=-np.log(rand)/extCoeff ## Distance!

                TotalPathLength[:]+=s[0,:]
                u1=u0+cosU*s ## New position!

                ## Where photons have gone out of the top of the snowpack!
                OutTop=np.ma.masked_where(u1[2,:] < self.__slabDepth, Photons)
                TotalPathVals+=list(TotalPathLength[np.where(u1[2,:] >= self.__slabDepth)])
                albedo+=np.sum(np.ma.masked_less(OutTop.compressed(),0).compressed()) ## Add all to albedo.

                ## See comments on surface section in the RunBRDF function.
                if self.__Surface == True:
                    ## Surfaced with defined BRDF is at this level!
                    SfcIdx=np.where(u1[2,:] <=0)[0] ## indicies where photons hit surface!

                    if np.sum(Photons[SfcIdx]) > 0:
                        Indir=cosU[:,SfcIdx]

                        OutDir=np.zeros_like(Indir)
                        Zen=np.arccos(-Indir[2,:])
                        Azi=np.arctan(Indir[1,:]/Indir[0,:])
                        ## SO FAR, it is recommended that only Lambert and Specular
                        ## options are used, since they can compute the reflected
                        ## direction as vectorized quantities
                        ## using a more complicated BRDF is too computationally expensive in accounting to
                        ## individual incident angles.  The said, the code is here to do that.
                        if self.__SfcBRDF == 'lambert':
                            if self.BRDFParams['fromFile'] == True: ## albedo is special, called from file ditionary!
                                self.BRDFParams['albedo']=np.interp(wavelen,
                                                    self.SFCFileData['wavelength (nm)'].values,
                                                    self.SFCFileData['reflect'].values)

                            outWeights=np.ones_like(Zen)*self.BRDFParams['albedo']
                            RandZen=np.random.uniform(0,1,size=len(Zen))*np.pi/2.
                            RandAzi=np.random.uniform(0,1,size=len(Zen))*np.pi*2.
                            OutDir[0,:]=np.cos(RandAzi)
                            OutDir[1,:]=np.sin(RandAzi)
                            OutDir[2,:]=np.cos(RandZen)
                        elif self.__SfcBRDF == 'specular':
                            if self.BRDFParams['fromFile'] == True: ## albedo is special, called from file ditionary!
                                self.BRDFParams['albedo']=np.interp(wavelen,
                                                    self.SFCFileData['wavelength (nm)'].values,
                                                    self.SFCFileData['reflect'].values)

                            outWeights=np.ones_like(Zen)*self.BRDFParams['albedo']
                            for kdx in range(Indir.shape[1]):
                                cosThetai =  -np.dot(Indir[:,kdx],np.array([0,0,-1]))
                                OutDir[:,kdx]= Indir[:,kdx] + 2. * cosThetai * np.array([0,0,-1])
                        else:
                            outWeights=np.zeros([len(Zen)])
                            for kdx in range(len(Zen)):
                                HRDF, NewZen,NewPhi=BRDFFunc.BRDFProb(1,Zen[kdx],Azi[kdx],
                                    self.__BRDFPhi,self.__BRDFTheta,self.__BRDFdTheta,
                                    ParamDict=self.BRDFParams,BRDF=self.__SfcBRDF)

                                outWeights[kdx]=HRDF
                                OutDir[:,kdx]=[np.sin(NewZen)*np.cos(NewPhi),np.sin(NewZen)*np.sin(NewPhi),np.cos(NewZen)]

                        transmiss+=np.sum(np.ma.masked_less((1.-outWeights)*Photons[SfcIdx],0).compressed()) ## Add all to transmiss
                        Photons[SfcIdx]=outWeights*Photons[SfcIdx] ## reflectance!
                        u1[2,SfcIdx] = 0.00001 ## set position to surface
                        cosU[:,SfcIdx] = OutDir[:]

                OutBottom=np.ma.masked_where(u1[2,:] >=0, Photons) ## out of the bottom
                TotalPathVals+=list(TotalPathLength[np.where(u1[2,:] < 0)])
                keepIdx=np.squeeze([(u1[2,:] >= 0) & (u1[2,:] < self.__slabDepth) & (Photons >= 0)]) ## Which photons to keep!
                transmiss+=np.sum(np.ma.masked_less(OutBottom.compressed(),0).compressed()) ## Add all to transmiss
                ## update all necessary arrays to keep only indexes that you need! ##

                TotalPathLength=TotalPathLength[keepIdx]
                u1=u1[:,keepIdx]
                u0=u0[:,keepIdx]
                Photons=Photons[keepIdx]
                if self.namelistDict['PhaseFunc'] == 1:
                    g=g[keepIdx]

                s = s[0,keepIdx]
                ## special case, if there is only 1 photon left, add to absorbed and kill it
                ## required to ensure that array lengths and indexing can work.
                if len(Photons) <= 1:
                    absorbed+=np.sum(Photons)
                    break
                cosU=cosU[:,keepIdx]
                layerIds=layerIds[keepIdx]

                if TrackTransmission == True:
                    PhotonId=PhotonId[keepIdx]
                    if iterNum%TransCompute == 1:
                        TransIdx=int(np.max([np.argmin(np.abs(sorted(transmissionDict[wavelen].keys())-np.min(u1[2,:])))-1,0]))
                        #print(TransIdx,len(PhotonId),np.min(u1[2,:]))
                        for i in sorted(transmissionDict[wavelen].keys())[TransIdx:]:
                            transmitted=np.where(u1[2,:] < i)
                            transmissionPower[wavelen][i]=transmissionPower[wavelen][i]+[Photons[list(PhotonId[transmitted]).index(j)] for j in PhotonId[transmitted] if j not in transmissionDict[wavelen][i]]
                            transmissionDict[wavelen][i]=transmissionDict[wavelen][i]+[j for j in PhotonId[transmitted] if j not in transmissionDict[wavelen][i]]

                ## Reset the current layer boundaries to match the current layer ids!
                currentBin=np.array([[self.__layerTops[i],self.__layerTops[i+1]] for i in layerIds]).T
                ## Reset coefficients to current ids!
                extCoeff=np.array([self.__ExtCoeffDict[i] for i in layerIds])
                Fice = np.array([self.__FiceDict[i] for i in layerIds])
                Fsoot = np.array([self.__SootDict[i] for i in layerIds])
                ## okay, to recap, we have performed the russian roulette routine to handel dead PHOTONS
                ## Moved the photon through space following the direction vector and current position.
                ## updated albedo transmissivity based on whether photons have left the snowpack entirely
                ## removed all photons no long in the snowpack, and update the current boundaries and coefficients.
                ## Now need to do the phase function (it not idelized!)
                if self.namelistDict['PhaseFunc'] > 1: ## if it's not idealized!
                    ProbIdx=np.random.randint(0,self.nSamples,size=len(Photons))
                    Scatter=np.zeros([len(Photons)])
                    ScatterNew=np.zeros_like(Scatter)
                    for i in self.__layerIds:
                        ScatIdx=np.where(layerIds == i)
                        Scatter[ScatIdx]=(self.__CosIs[i][ProbIdx])[ScatIdx]
                else:
                    rand=np.random.uniform(0,1.,size=len(Photons))
                    Scatter=(1./(2.*g))*(1.+g**2.-((1-g**2.)/(1.-g+2.*g*rand))**2.)
                    ProbIdx=0

                layerIds, Scatter, Fice, extCoeff,Fsoot=self.__ChangeLayers(u0,u1,currentBin,layerIds,Fice,extCoeff,Fsoot,
                                                                    self.__ExtCoeffDict,self.__FiceDict,self.__SootDict,
                                                                    g,Scatter,ProbIdx)

                ## Absorb some photons! ##
                if self.__AbsorbtionMethod == 1:
                    deltaW=(1.-np.exp(-s*(kappaIce*Fice+kappaSoot*Fsoot)))*Photons

                else:
                    ## Remember, if absorption method == 2, then Fice IS the absorption coefficient
                    deltaW=(1. - np.exp(-s*Fice))*Photons

                absorbed+=np.sum(deltaW)
                Photons=Photons-deltaW

                ## set new direction vector.
                cosU=self.__SetDirection(cosU,Scatter)
                u0[:]=u1[:]  ## set starting position to new position

                iterNum+=1
            ## Finished with this wave length!
            if verbose == True:
                print("Albedo:",albedo/nPhotons)
                print("Absorption:",absorbed/nPhotons)
                print("Transmission:",transmiss/nPhotons)
                print("Sum:",(albedo+absorbed+transmiss)/nPhotons)

            SpectralAlbedo.append(albedo/nPhotons)
            SpectralAbsorption.append(absorbed/nPhotons)
            SpectralTransmissivity.append(transmiss/nPhotons)

            if TrackTransmission == True:
                for i in transmissionDict[wavelen].keys():
                    transmissionPower[wavelen][i]=sum(transmissionPower[wavelen][i])/nPhotons
            else:
                transmissionPower=None

        return SpectralAlbedo, SpectralAbsorption, SpectralTransmissivity,transmissionPower


    def PhotonTrace(self,WaveLength,Zenith,Azimuth,nPhotons=100,
                      verbose = True,angleUnits='degrees'):

        """
            User accessible function to compute the spectral albedo of the medium from photon-tracking

            Inputs:
                WaveLength = wavelength of light entering the medium (in nm)
                Zenith angle = incident zenith angle
                Azimuth Angle = incident azimuth angle (counter-clockwise from east)
                nPhotons = number of photons to launch into the medium (keep less than 100)

                verbose (optional) = Turn on some additional printout text during model integration
                angleUnits (optional) = Units of input angle, either degrees or radians

            Returns:
                albedo = overall reflected fraction
                absorbed = overall absorbed fraction
                transmiss = overall transmitted fraction
                XOut = X position within the medium (list containing smaller lists)
                YOut = Y position within the medium (list containing smaller lists)
                ZOut = Z position within the medium (list containing smaller lists)
                PowerOut = Photon energy within the medium (list containting smaller lists)
        """

        if self.__initialized == False:
            print("The Model has not been initalized.  You must initialize this model before running 'PhotonTrace'. Exiting.")
            print(" --> mod.Initialize()")
            sys.exit()

        WaveLength=np.array(WaveLength)

        ## Make sure WaveLength is in the limits
        WaveLength = np.ma.masked_outside(WaveLength,self.WaveLengthLimits[0],
                                          self.WaveLengthLimits[1]).compressed()

        if len(WaveLength) == 0:
            self.FatalError("Your WaveLength Array Length is Zero, you probably used a value outside of the allowed limits!")

        ## since single tracking, only showing one wave length!

        WaveLength=WaveLength[0]
        RI,n,kappaIce=self.GetRefractiveIndex(WaveLength,units='nm')

        self.AssignMaterial(self.namelistDict['Contaminate'],filePath=self.namelistDict['MaterialPath'])
        Rsoot,nsoot,kappaSoot=self.GetRefractiveIndex(WaveLength,units='nm')
        self.AssignMaterial('ice',filePath=self.namelistDict['MaterialPath'])

        AllowedAngleUnits=['deg','rad','degrees','radians']
        if angleUnits.lower() not in AllowedAngleUnits:
            self.Warning("The angle unit %s is not allowed, assuming your input angles are in degrees"%angleUnits)
            angleUnits='degrees'

        if angleUnits.lower() in ['deg','degrees']:
            ## if the units are degrees, convert to radians!
            ## fix azimuth to point in the direction of the inident radiation (i.e., where it's coming from!)
            Azimuth=np.pi-np.radians(Azimuth)
            Zenith=np.radians(Zenith)

        if np.cos(Zenith) <=0:
            self.FatalError("The Zenith angle %.1f rad is below the horizon!  Cannot use for incident radiation!"%Zenith)

        cosUStart=[np.sin(Zenith)*np.cos(Azimuth),np.sin(Zenith)*np.sin(Azimuth),-np.cos(Zenith)]

        print("Computing Spectral Albedo of this Snowpack From %.1f to %.1f"%(np.min(WaveLength),np.max(WaveLength)))

        XOut=[]
        YOut=[]
        ZOut=[]
        PowerOut=[]

        if verbose == True:
            print("Running Short Simulation for Photon Tracing on Wavelength %.1f nm"%WaveLength)
        ## initial position of the photons, random within x and y BOUNDARIES
        ## z position set to top of layer!
        xx = np.random.uniform(self.xSize,size=nPhotons)
        yy = np.random.uniform(self.ySize,size=nPhotons)
        zz = np.ones_like(xx)*self.__slabDepth
        ## Now, I need to set up coordinates for all the layer idenfication that matches
        ## The number of photons!
        ## Starting at top layer, so all layer ids start at zero!
        layerIds=np.zeros([nPhotons],dtype=np.short) ## short integer

        extCoeff=np.array([self.__ExtCoeffDict[i] for i in layerIds])
        Fice = np.array([self.__FiceDict[i] for i in layerIds])
        Fsoot = np.array([self.__SootDict[i] for i in layerIds])
        if self.namelistDict['PhaseFunc'] == 1: ## If using idealized HeyNey Green Function!
            g=np.array([self.namelistDict['Asymmetry'][i] for i in layerIds])
        else:
            g=self.NumLayers*[0.9]

        ## Current bin defines the lower and upper boundaries of the bin!
        currentBin=np.array([[self.__layerTops[i],self.__layerTops[i+1]] for i in layerIds]).T

        Photons=np.ones([nPhotons],dtype=np.double)
        u0=np.array([xx,yy,zz])
        DiffusePhotons = int(nPhotons*self.__DiffuseFraction)
        DirectPhotons = nPhotons-DiffusePhotons

        randAzi=np.random.uniform(0.0,2.*np.pi,size=DiffusePhotons)
        randZen=np.random.uniform(0.0,np.pi/2.,size=DiffusePhotons)
        diffuse=[[np.sin(randZen[j])*np.cos(randAzi[j]),np.sin(randZen[j])*np.sin(randAzi[j]),-np.cos(randZen[j])] for j in range(DiffusePhotons)]

        cosU=np.array(DirectPhotons*[cosUStart]+diffuse).T
        iterNum=0.0

        absorbed=0.0
        albedo=0.0
        transmiss=0.0
        iterNum=1

        XOut.append(xx)
        YOut.append(yy)
        ZOut.append(zz)
        PowerOut.append(Photons)
        while(len(Photons) > 0):
            Photons,abso=self.__RussianRoulette(Photons)
            absorbed+=abso

            ## Get Direction!
            rand=np.random.uniform(0.0,1.,size=len(Photons))
            rand=np.array(3*[rand])
            s=-np.log(rand)/extCoeff ## Distance!
            u1=u0+cosU*s ## New position!

            ## Where photons have gone out of the top of the snowpack!
            OutTop=np.ma.masked_where(u1[2,:] < self.__slabDepth, Photons)
            albedo+=np.sum(np.ma.masked_less(OutTop.compressed(),0).compressed()) ## Add all to albedo.
            if self.__Surface == True:
                ## Surfaced with defined BRDF is at this level!
                SfcIdx=np.where(u1[2,:] <=0)[0] ## indicies where photons hit surface!

                if np.sum(Photons[SfcIdx]) > 0:
                    Indir=cosU[:,SfcIdx]


                    Zen=np.arccos(-Indir[2,:])
                    Azi=np.arctan(Indir[1,:]/Indir[0,:])
                    OutDir=np.zeros_like(Indir)
                    if self.__SfcBRDF == 'lambert':
                        outWeights=np.ones_like(Zen)*self.BRDFParams['albedo']
                        RandZen=np.random.uniform(0,1,size=len(Zen))*np.pi/2.
                        RandAzi=np.random.uniform(0,1,size=len(Zen))*np.pi*2.
                        OutDir[0,:]=np.cos(RandAzi)
                        OutDir[1,:]=np.sin(RandAzi)
                        OutDir[2,:]=np.cos(RandZen)

                    elif self.__SfcBRDF == 'specular':
                        outWeights=np.ones_like(Zen)*self.BRDFParams['albedo']
                        for kdx in range(Indir.shape[1]):
                            cosThetai =  -np.dot(Indir[:,kdx],np.array([0,0,-1]))
                            OutDir[:,kdx]= Indir[:,kdx] + 2. * cosThetai * np.array([0,0,-1])
                    else:
                        outWeights=np.zeros([len(Zen)])
                        for kdx in range(len(Zen)):
                            HRDF, NewZen,NewPhi=BRDFFunc.BRDFProb(1,Zen[kdx],Azi[kdx],
                                self.__BRDFPhi,self.__BRDFTheta,self.__BRDFdTheta,
                                ParamDict=self.BRDFParams,BRDF=self.__SfcBRDF)

                            outWeights[kdx]=HRDF
                            OutDir[:,kdx]=[np.sin(NewZen)*np.cos(NewPhi),np.sin(NewZen)*np.sin(NewPhi),np.cos(NewZen)]

                    transmiss+=np.sum(np.ma.masked_less((1.-outWeights)*Photons[SfcIdx],0).compressed()) ## Add all to transmiss
                    Photons[SfcIdx]=outWeights*Photons[SfcIdx] ## reflectance!
                    u1[2,SfcIdx] = 0.00001 ## set position to surface
                    cosU[:,SfcIdx] = OutDir[:]

            OutBottom=np.ma.masked_where(u1[2,:] >=0, Photons) ## out of the bottom
            keepIdx=np.squeeze([(u1[2,:] >= 0) & (u1[2,:] < self.__slabDepth) & (Photons >= 0)]) ## Which photons to keep!
            transmiss+=np.sum(np.ma.masked_less(OutBottom.compressed(),0).compressed()) ## Add all to transmiss
            ## update all necessary arrays to keep only indexes that you need! ##

            u1=u1[:,keepIdx]
            u0=u0[:,keepIdx]
            Photons=Photons[keepIdx]

            s = s[0,keepIdx]
            ## special case, if there is only 1 photon left, add to absorbed and kill it
            ## required to ensure that array lengths and indexing can work.
            if len(Photons) <= 1:
                absorbed+=np.sum(Photons)
                break
            cosU=cosU[:,keepIdx]
            layerIds=layerIds[keepIdx]
            Fice=Fice[keepIdx]
            extCoeff=extCoeff[keepIdx]

            ## Reset the current layer boundaries to match the current layer ids!
            currentBin=np.array([[self.__layerTops[i],self.__layerTops[i+1]] for i in layerIds]).T
            ## Reset coefficients to current ids!
            extCoeff=np.array([self.__ExtCoeffDict[i] for i in layerIds])
            Fice = np.array([self.__FiceDict[i] for i in layerIds])
            Fsoot = np.array([self.__SootDict[i] for i in layerIds])
            ## okay, to recap, we have performed the russian roulette routine to handel dead PHOTONS
            ## Moved the photon through space following the direction vector and current position.
            ## updated albedo transmissivity based on whether photons have left the snowpack entirely
            ## removed all photons no long in the snowpack, and update the current boundaries and coefficients.
            ## Now need to do the phase function (it not idelized!)
            if self.namelistDict['PhaseFunc'] > 1: ## if it's not idealized!
                ProbIdx=np.random.randint(0,self.nSamples,size=len(Photons))
                Scatter=np.zeros([len(Photons)])
                ScatterNew=np.zeros_like(Scatter)
                for i in self.__layerIds:
                    ScatIdx=np.where(layerIds == i)
                    Scatter[ScatIdx]=(self.__CosIs[i][ProbIdx])[ScatIdx]

            else:
                rand=np.random.uniform(0,1.,size=len(Photons))
                Scatter=(1./(2.*g))*(1.+g**2.-((1-g**2.)/(1.-g+2.*g*rand))**2.)
                ProbIdx=0

            layerIds, Scatter, Fice, extCoeff,Fsoot=self.__ChangeLayers(u0,u1,currentBin,layerIds,Fice,extCoeff,Fsoot,
                                                                self.__ExtCoeffDict,self.__FiceDict,self.__SootDict,
                                                                g,Scatter,ProbIdx)

            ## Absorb some photons! ##
            deltaW=(1.-np.exp(-s*(kappaIce*Fice+kappaSoot*Fsoot)))*Photons

            absorbed+=np.sum(deltaW)
            Photons=Photons-deltaW

            ## set new direction vector.
            cosU=self.__SetDirection(cosU,Scatter)
            u0[:]=u1[:]  ## set starting position to new position

            XOut.append(u0[0,:])
            YOut.append(u0[1,:])
            ZOut.append(u0[2,:])
            PowerOut.append(Photons)
            iterNum+=1

        ## Finished with this wave length!
        if verbose == True:
            print("Albedo:",albedo/nPhotons)
            print("Absorption:",absorbed/nPhotons)
            print("Transmission:",transmiss/nPhotons)
            print("Sum:",(albedo+absorbed+transmiss)/nPhotons)


        return albedo, absorbed, transmiss,XOut,YOut,ZOut,PowerOut


    def PlotBRDF(self,BRAziBins,BRZenBins,BRDFArray,norm=1,figsize=(10,10),**kwargs):
        figure=plt.figure(figsize=figsize)
        ax=plt.subplot(111,projection='polar')

        Z=ax.contourf(np.radians(BRAziBins),BRZenBins,np.transpose(BRDFArray)/(norm),
                **kwargs)
        return figure,ax,Z


    def FatalError(self,message):
        raise Exception(message)

    def Warning(self,message):
        import warnings
        warnings.warn("Warning...........%s"%message)

    def __PhaseSamples(self,Theta,probs,nSamples,dtheta):
        """
            This function reads in an array of scattering angles (Theta) and returns an
            array of scattering directions (cos[theta]) with size nSamples, based on a
            probability distrubtion for scattering angle derived from a scattering phase function.

            Theta: Angle in radians array of size (number of bins, ranges from -pi to pi)
            probs: Probability distribution weighting function for Theta, must be equal in size to Theta,
                    and approximately equal to 1.
            nSamples: Number of CosI directions to return and pull from in the model.
                    The more samples, the better, 10000, is probably good.
            dtheta: Size of Angular bin, used to add randomness into the larger samples

            Returns:
            --------
                Cos[Theta] = Array with size nSamples containing probabilistic scattering direction samples.
        """
        ## Build array of samples that matches the probability in probability function
        choice=np.random.choice(Theta, nSamples, p=list(probs/np.sum(probs)))
        ## Assume that the sample will fall within +- binsize/2 of the sample bin.
        rand=np.random.uniform(-1,1,nSamples)*dtheta/2.
        CosIs=np.cos(choice+rand)
        ## Check to make sure the sample is within -1 and 1, and fix.
        CosIs=np.ma.masked_greater(CosIs,1).filled(1.0)
        CosIs=np.ma.masked_less(CosIs,-1).filled(-1.0)
        return CosIs


    def clearSurface(self):
        """Simple function that is used to clear the lower boundary:

           Example usage:
                mod.clearSurface()

        """

        self.__Surface=False
        self.__SfcBRDF=None
        self.BRDFParams=None
        self.__BRDFTheta=None
        self.__BRDFPhi=None
        self.__BRDFdTheta=None
        self.__sfcFile=None

        print("Lower boundary and all associated data have been cleared.  RTM will no longer run with a lower boundary.")
        return

    def loadSurface(self,BRDF,binsize=4.,fromFile=False,sfcFile='/Users/rdcrltwl/Desktop/NewRTM/Surfaces/BlackPlate.csv'):

        """
            User accessible function to load a lower boundary into the model.
            The lower boundary sets a surface reflectance value that can reflect a wavelength dependent (or independent)
            portion of rays that reach the bottom of the snowpack upward back into the snowpack.
            Surface boundarys can be lambertian (random reflected direction) or Specular (Mirror reflected) or be given a specific BRDF.
            Note that while the BRDF specification appears to work okay, it is extraordinarily slow due to having to compute a probabilistic
            reflected direction for all incident angles.
            Further, only the lambertian boundary has been rigorously tested, so use other options at your own risk.
            **Note that only lambert and specular boundaries include a surface file.

            Inputs:
                BRDF = Type of reflectance function --> Must be one of: 'lambert','specular','hapke','rpv'
                binSize (optional) = BRDF bin size (degrees), only used for 'hapke' and 'rpv' options.
                fromFile (optional) = Boolean flag to determine if the reflected portion is wavelength dependent or not.
                    if this value is False, the albedo will be wavelength indepenent and set within the BRDFParams dictionary as "albedo"
                sfcFile (optional) = Fulle File path to a .csv file that has the wavelength dependent reflectance values

            Returns:
                Sets the self.__Surface value = True, and sets self.BRDFParams dictionary.
        """

        if self.__initialized == False:
            print("The Model has not been initalized.  You must initialize this model before running 'loadSurface'. Exiting.")
            print(" --> mod.Initialize()")
            sys.exit()

        if BRDF.lower() not in ['hapke','rpv','lambert','specular']:
            Warning("The BRDF %s is not allowed!"%BRDF)
            print("Defaulting to 'lambert'")
            BRDF='lambert'

        print("-----------------------------------")
        print("|  Setting Surface with %s BRDF"%BRDF)
        if BRDF.lower() == 'hapke':
            print("Surface parameters can be accessed inline with the BRDFParams dictionary")
            print("e.g., Slab.BRDFParams['omega'] = 0.4")
            self.BRDFParams={'omega':0.6,'b_0':1.0,'h_h':0.06}

        if BRDF.lower() in ['lambert','specular']:
            print("Surface parameters can be accessed inline with the BRDFParams dictionary")
            print("e.g., Slab.BRDFParams['albedo'] = 0.4")
            self.BRDFParams={'albedo':0.32,'fromFile':fromFile}

            if fromFile == True:
                self.__sfcFile=sfcFile
                self.SFCFileData=pd.read_csv(self.__sfcFile)

        if BRDF.lower() == 'rpv':
            print("Surface parameters can be accessed inline with the BRDFParams dictionary")
            print("e.g., Slab.BRDFParams['g'] = -0.4")
            self.BRDFParams={'rho_0':0.027,'k':0.647,'g':-0.169,'h_0':0.1}

        self.__Surface=True
        self.__SfcBRDF=BRDF.lower()

        ## Need to Assign n array of thetas and phis to use in BRDF probability function!
        ## This helps determine a probability of refleted direction for every incident direction
        Theta=np.radians(np.arange(0,90+binsize,binsize))
        theta=(Theta[:-1]+Theta[1:])/2. ## Stagger
        phi=np.radians(np.arange(0,360+binsize,binsize))
        phi=(phi[:-1]+phi[1:])/2. ## Stagger

        theta,phi=np.meshgrid(theta,phi)
        self.__BRDFTheta=theta
        self.__BRDFPhi=phi
        self.__BRDFdTheta=np.radians(binsize)
        ## Clear memory!
        theta,phi,Theta=None,None,None

        print("Surface has been set.  CRREL RTM will now run with a lower boundary")
        print("To access the BRDF parameters, use the BRDFParams dictionary.")
        print("-----------------------------------")
        return


    def AssignMaterial(self,material,filePath):
        import glob as glob
        from pandas import read_csv

        files=glob.glob('%s/*.csv'%filePath)
        allowed=[os.path.basename(f).split('_')[0] for f in files]
        if material.lower() not in allowed:
            print('%s is not allowed, defaulting to %s '%(material.lower(),allowed[0]))
            print("Allowed Materials:")
            for i in allowed:
                print(" -- %s"%i)
            material=allowed[0]

        RIfile=glob.glob('%s/*%s*.csv'%(filePath,material.lower()))[0]
        self.__RIFile=RIfile

        ##open Refractive index file
        RIdata=read_csv(RIfile)
        self.__RefracData=read_csv(RIfile)
        ## convert from um to meters
        self.__RefracData.wave=self.__RefracData.wave*1E-6
        self.__Material=material

        return

    def GetRefractiveIndex(self,wavelength,units='nm'):
        import numpy as np
        import re
        if isinstance(self.__RefracData,type(None)):
            print("This Object class does not have an assinged material")
            print("Please assign a material using the 'AssignMaterial' function before trying to get refractive indicies")
        allowedUnits={'m':1,'cm':100,'mm':1000,'um':1e6,'nm':1e9}

        if type(wavelength) == str:
            res = re.split('([-+]?\d+\.\d+)|([-+]?\d+)', wavelength.strip())
            res_f = [r.strip() for r in res if r is not None and r.strip() != '']
            if len(res_f) == 1:
                ## no units, you just put it in as a float.
                wavelength=res_f
            elif len(res_f) == 2:
                wavelength,units=res_f
                wavelength=float(wavelength)
            else:
                print("This is an unacceptable input.")

        if units not in allowedUnits:
            print("%s is not an allowed unit"%units)
        else:
            wavelength=wavelength/allowedUnits[units]

        ## Use log interpolation for refractive index and refractive indices
        k=np.exp(np.interp(np.log(wavelength),np.log(self.__RefracData.wave.values),np.log(self.__RefracData.Im.values)))
        RI=np.interp(np.log(wavelength),np.log(self.__RefracData.wave.values),self.__RefracData.Re.values)
        abso=(4.*np.pi*k/wavelength)*1e-3 # absorption coefficient
        n=RI ## real part of refractive index

        return RI, n, abso

    def __ParseNamelist(self):
        """
            This helper function parses a namelist file and replaces values
            Automatically called during the __init__ function after
            confirming that the namelist file actually exists.

        """
        NamelistDict=self.namelistDict
        NamelistFile=self.namelistPath
        namelistKeys=[i for i in NamelistDict.keys()]

        NamelistData = open(self.namelistPath, 'r')
        Lines = NamelistData.readlines()

        listKeys=['PropFileNames','LayerTops','Fsoot','Asymmetry']
        # Strips the newline character
        for line in Lines:
            line=line.split('#')[0]
            key = line.split('=')[0].lstrip().rstrip()
            if key in namelistKeys:
                if key in listKeys:
                    values=line.split('=')[1].split('\n')[0].lstrip().rstrip().split(',')
                    try:
                        NamelistDict[key] = [float(i) for i in values if i != '']
                    except:
                        NamelistDict[key] = [i for i in values if i != '']
                else:
                    value=line.split('=')[1].lstrip().rstrip().split('\n')[0].split(',')[0]
                    try:
                        NamelistDict[key] = float(value)
                    except:
                        NamelistDict[key] = value

        self.namelistDict=NamelistDict
        return


    def WriteSpectralToFile(self,outfileName,nPhotons,Zenith,Azimuth
                ,WaveLength,Albedo,Absorption,Transmission,filename='Untitled'):
        """
            This function writes spetral data out to a .txt file including
            all of the namelist dictionary options in a header.

            !! WARNING !!
            Note, if you run the model and change the namelist options,
            you will save incorrect metaData to the header
            information in the output file, even if you haven't initialized the new model.
            This admittedly is a result of lazy/poor coding, and may be fixed in the future,
            but for now, a good "best practice" is the run this as soon as you've run the Spectral model
            prior to doing anything else.
        """
        from datetime import datetime

        outfile= open(outfileName,"w+")
        ## write header information first! ##
        nowtime=datetime.now().strftime('%c')

        listKeys=['PropFileNames','LayerTops','Fsoot','Asymmetry']

        outfile.write("Simulated Spectral Output From CRREL Snow RTM\n")
        outfile.write("File Created: %s\n"%nowtime)
        outfile.write("Short Description: %s\n"%filename)
        outfile.write("%i photons used in simulation\n"%nPhotons)
        outfile.write("Incident Zenith Angle = %.2f Degrees\n"%Zenith)
        outfile.write("Incident Azimuth Angle = %.2f Degrees\n"%Azimuth)
        outfile.write("Slab Depth = %.2f mm\n"%self.__slabDepth)
        outfile.write("Surface Boundary = %s\n"%str(self.__Surface))
        if self.__Surface == True:
            outfile.write("Surface Reflectance Function: %s\n"%self.__SfcBRDF)
            for i in self.BRDFParams.keys():
                outfile.write('- %s:%s\n'%(i,str(self.BRDFParams[i])))

        outfile.write("Namelist Options:\n")

        for i in self.namelistDict.keys():
            if i in listKeys:
                outlist=[str(i) for i in self.namelistDict[i]]
                value=",".join(outlist)
            else:
                value=self.namelistDict[i]

            outfile.write('- %s:%s\n'%(i,value))
        outfile.write("------------------------------------------------------  \n")
        outfile.write("CSV Header Below this line\n")
        outfile.write('WaveLength,Albedo,Transmissivty,Absorption\n')
        for i in range(len(WaveLength)):
            outfile.write('%.3f,%.3f,%.3f,%.3f\n'%(WaveLength[i],Albedo[i],Transmission[i],Absorption[i]))

        outfile.close()
        return
