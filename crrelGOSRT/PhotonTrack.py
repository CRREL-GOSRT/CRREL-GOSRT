import sys
from crrelGOSRT import CRRELPolyData,RenderFunctions,RTcode
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import vtk
import pyvista as pv
import pandas as pd
from datetime import datetime
import glob as glob
import os


def RayTracing_OpticalProperties(VTKFilename,GrainFolder,OutputFilename,MaterialPath,wavelen,VoxelRes,
                                 nPhotons=1500,Absorb=True,GrainSamples=40,maxBounce=100,PhaseBins=180,
                                 verbose=False,distScale=1.2,VoxelUnits='um',wavelenUnits='nm',plot=True,
                                 phaseSmooth=300,raylen=2000,PF_fromSegmentedParticles=False,
                                 Advanced = True,TrackThresh=0.1,TrackDepthMax=4,
                                 Polar = False,particlePhase=True,
                                 Tolerance = 0.001,MeshDescription='Snow Mesh',MaxTIR=30):

    """
        This function calls individual sub-functions to get optical properties and saves them to file for
        1D spectral albedo model:

        Version 0.3.0 --> Major updates to the code to reflect the replacement of the Xiong et al. (2015) method for computing the extiction
                          coefficient with the method described in Randrianaliosa and Baillis (2010) method that uses the mean free-path
                          through explicit ray-tracing.

                        Key Updates:
                          - Compute extinction function is no longer called as part of the ray-trace
                          - Computing the B parameter is no longer an option, instead it is required and used in the slab model.

                        Minor Updates:
                          - Removed options to compute Fice from density
                          - Added option for user to set raylen, can speed up the code if the ray doesn't span the entire sample, however could potentially miss intersections this way.
                          - Added optino for the user to specify the tolerance for idenifying intersections in the mesh.  Should be no greater than 0.005.

        Version 0.2.1 --> Some bug fixes and code cleaning has been performed (September 2021)
            - Key Updates:
                Added "Advanced" photon tracking to allow for more complete phase function calculations for complicated particles (e.g., aggregates or hollow colunms)
                Added FiceFromDensity as a simple approximation for Fice using linear regression, instead of the computationally expensive ray-tracing

        Version 0.2 --> Replaced version 0.1 January 2021
            - Key Updates: moved "research" code to packaged functions --> Added comments / Function strings.

        In order the following optical properties are determined through photon-tracking:
            1. Extinction coefficient
            2. F_{ice}
            3. Scattering Phase Function

        Inputs:
            VTKFilename - (string) Path to full 3D snow sample mesh.  Mesh MUST be manifold.
            GrainFolder - (string) Path to folder where individual 3D grain meshes are stored (used in scattering phase function)
            OutputFilename - (string) Full Path to output .txt file that will include the optical properties.
            MaterialPath - (string) Path to .csv files with refractive indicies for specified materials.  Note, there must be a .csv file with the "Ice" Material
            wavelen - (string/float) Wavelength of light used in computing optical properites, optical properties are generally insensitive to wavelength over the allowed range.
                       if this is a string, it will compute the units from the characters within the string, if a float, it will get units from the "wavelenUnits" variable.
            VoxelRes - (string/float) Physical resolution of a voxel. If this is a string, it will compute the units from the characters within the string, if a float, it will get units from the "VoxelUnits" variable.

            nPhotons (optional) - Number of photons used to compute optical properties via ray-tracing,  Higher number will be more statistically robust, but more computationally expensive.
            Absorb (optional) - Boolean flag to determine if absorbtion should be included in the computation of the scattering phase function.  If false, absorption if neglected in this calculation.
            GrainSamples (optional) - Number of individual grain samples to pull when computing the scattering phase function.
                                      Note, that each grain will be sampled nPhoton number of times
                                      If the number of rendered grains in GrainFolder is less than GrainSamples, then some grains will be sampled multiple times.

            maxBounce (optional) - This helps kill photons that are stuck in infinite total internal reflection loops within ice particles,
                                   by limiting the max number of bounces within a particle.  This also cuts back on the accumulation of errors caused by imperfect rendering
            PhaseBins (optional) - Number of bins to use in representing the scattering phase function
            verbose (optional) - Boolean flag to control how much information is written to the screen during the function
            distScale (optional) - Controls how to control the distance increment in the computation of the extinction coefficient (should be less than 2)
                                   i.e: d=d_o * distScale, where d_o is the previous distance.

            VoxelUnits (optional): If VoxelRes is given as a float, then use this for the units, otherwise, this argument is ignored.
            wavelenUnits (optional): If wavelen is a given as a float, then use this for the units, otherwise, this argument is ignored.
            Plot (optional): If True, return a figure with 4 panels showing
                             1. The extinction coefficient curve fit
                             2. 3D Sample Mesh
                             3. Histogram of F_{ice}
                             4. Scattering Phase Function

            phaseSmooth (optional): Number of iterations to use in smoothing function to refine the mesh prior to running the model.
            raylen (optional): distance (mm) to fire a ray.  Should be long enough to ensure it will either hit a particle, or traverse the sample.

            PF_fromSegmentedParticles (optional / boolean) : If True will look for segmented individual particles in the folder specified by "GrainFolder"
                                                             Note that this will change how some of the other aspects of the optical properties are computed.

            Advanced (optional): If True, allows for multiple interactions with an irregularly shaped particle.
            Specifically, photons that exit the particle with remaining energy greater than some threshold are checked for re-intersections with the particle.
            Any reintersections are subsequently tracked.  Can be particularly useful for hollow facets or other hollow particles where reintersections are likely.
            Expect significantly longer computational time when this is on.

            TrackThresh (optional): Minimum energy threshold for which to track photons for re-intersections.

            TrackDepthMax (optional): Maximum number of times ("depths") for which to track for reinterections.
                Generally, most particles will go less than 3 reintersections before running out of steam, but this allows for a hard limit.

            !------------------------------------------------------------------------------------------------!
            Developmental Options (Note, these are not rigorously tested, and should not be used in operation)
            !------------------------------------------------------------------------------------------------!



            Polar (optional/developmental): Adds polarization to the scattering phase function calculation.
                                            Currently, no polarization data is actually saved to the output data file,
                                            but can be used to look at linear polarization of a specific particle shape by plotting.


            Tolerance (float) - Tolerance used in vtk.obbTree to determine how close a ray needs to be to a mesh "cell" to
                            count as an intersection.  Recommended that a value of no larger than 0.001 is used, especially for
                            versions of vtk < 9.1.x. 0.01 seems to work for vtk versions > 9.1.
                            To be safe, I have set an assertion that this value is < 0.02.

            MeshDescription (optional/string) - You can add a descriptive name to the mesh that will be passed to the optical file.

            MaxTIR (optional/int) - Number of total internal reflections to allow within the ice-phase before killing the photon.
                                    Helps reduce the effect of TIR loops on computational expense.

        Returns:
            Saves optical properties to the OutputFilename, and returns a figure showing the optical properites
            if plot == false, then the returned figure is "None"
        """

    GrainFiles=glob.glob(GrainFolder+'*.vtk')
    VTKfile=glob.glob(VTKFilename)

    ## Comment out for speedier run time with single grain, instead of full mesh.  Good for debugging.
    #VTKFilename=GrainFiles[1]

    ## Perform initial checks ##
    if PF_fromSegmentedParticles == True:
        assert len(GrainFiles) > 0, "There are no Grain Files in %s, I cannot compute the phase function without grain files when PF_fromSegmentedParticles == True."%GrainFolder
    assert len(VTKfile) > 0, "There is no VTK Mesh file with the name %s.  I cannot compute the optical properties without a mesh file."%VTKFilename

    ## put a print statement here.

    VoxelRes_mm=ConvertVoxel2mm(VoxelRes,VoxelUnits)
    if verbose == True:
        print("Voxel Resolution = %.5f mm"%VoxelRes_mm)

    ## Get SnowMesh
    SnowMesh,sampleVolumeOut,Density,SSA,GrainDiam=ReadMesh(VTKFilename,VoxelRes_mm,verbose=verbose,
                                                            Tolerance=Tolerance,description=MeshDescription,smooth=phaseSmooth)

    if str(raylen).lower() == 'auto':
        raylen = 8.*1./(SSA*Density/4000.) ## assume a distance 20 x the theoretical scattering coefficient.
        print("Auto computing raylen from mesh properties = %.1f"%(raylen))


    print("Finished loading mesh ... ")
    SnowMesh.AssignMaterial('ice',filePath=MaterialPath)  ##Assign Material

    distances=[]
    dd=VoxelRes_mm
    while(dd < 1.*(SnowMesh.xBounds[1]-SnowMesh.xBounds[0])):
        dd+=2.*VoxelRes_mm
        distances.append(dd)
    distances=np.array(distances)
    if isinstance(wavelen,list) == True:
        nIce,kIce=[],[]
        for wdx,w in enumerate(wavelen):
            nIce1,kIce1,Nc=SnowMesh.GetRefractiveIndex(w,units=wavelenUnits)
            if wdx == int(len(wavelen)/2):
                nIce=nIce1
            kIce.append(kIce1)
    else:
        nIce,kIce,Nc=SnowMesh.GetRefractiveIndex(wavelen,units=wavelenUnits)


    text="SSA = %.1f m^2/kg \nrho_s = %.1f kg/m^3 \nSample Vol = %.1f mm^3"%(SSA,Density,sampleVolumeOut)

    if plot == True:
        plotter = pv.Plotter(off_screen=True, notebook=False,window_size =[1024, 1024])
        #plotter.add_text(text, font_size=20)
        #plotter.add_text(VTKFilename,font_size=20,position='lower_left',color='k')
        plotter.add_mesh(pv.wrap(SnowMesh.GetPolyData()),
               show_edges=False, opacity=1., color="w",
               diffuse=0.8, smooth_shading=True,specular=0.2)

        #_,perspective = plotter.show(screenshot=True)  ### this is returning a NoneType object for me, causing issue with subplot 2,2,2 below





    ## Next Up, F_Ice and PhaseFunction
    ###
    time1=datetime.now()
    if PF_fromSegmentedParticles == True:
        Fice,Fice_Straight,TotalLengths,kExt,missed=ComputeFice(SnowMesh,nIce,kIce,nPhotons,verbose=verbose,maxBounce=maxBounce,
                                              PhaseBins=PhaseBins,particlePhase=particlePhase,raylen = raylen,TIR=MaxTIR)
    else:
        Fice,TotalLengths,POWER,thetas,Fice_Straight,kExt,missed=ComputeFice(SnowMesh,nIce,kIce,nPhotons,verbose=verbose,maxBounce=maxBounce,
                                              PhaseBins=PhaseBins,particlePhase=particlePhase,raylen = raylen,TIR=MaxTIR)

        bins=np.linspace(0,np.pi,PhaseBins)
        dtheta=np.abs(bins[1]-bins[0])
        asymmparam = 0.5*dtheta*np.sum(np.cos(thetas)*np.sin(thetas)*POWER)

    time2=datetime.now()
    print("Found F_{ice} = %.2f after %.1f seconds"%(np.nanmean(Fice),(time2-time1).total_seconds()))
    print("Found k_{sca} = %.2f"%(kExt))
    print("Total Missed Photons =%i --> %.3f percent of all photons"%(missed,100.*missed/nPhotons))
    print("------------------------------------")

    #####

    if plot == True:

        fig=plt.figure(figsize=(9,9))
        ax=fig.add_subplot(2,2,1)
        ax.plot(distances,1.0-np.exp(-(kExt*distances)),color='indigo',ls='-',label='$\gamma_{sca}$ = %.2f'%kExt)
        ax.set_ylabel("POE")
        ax.set_xlabel("$d$ (mm)")
        plt.title("Curve fit for $\gamma_{ext}$")
        ax.grid()

        # ax=fig.add_subplot(2,2,2)
        # ax.imshow(perspective)  ## commented for now until we debug perspective variable
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        ax=fig.add_subplot(2,2,3)
        ax.hist(np.array(Fice),edgecolor='k')
        ax.axvline(np.nanmean(Fice),color='r',lw=2.0,ls='--')
        ax.set_xlabel("$F_{ice}$",fontsize=12)
        ax.set_ylabel("Count")
        ax.grid()
        plt.title("Histogram of $F_{ice}$",loc='left')


    if PF_fromSegmentedParticles == True:
        print("Computing Scattering Phase Function using %s Grain Samples..."%GrainSamples)

        time1=datetime.now()
        POWER,thetas,PhaseText,dtheta,ScatAlb=ComputeScatteringPhaseFunc(PhaseBins,GrainSamples,GrainFiles,nPhotons,SnowMesh,nIce,kIce,
                                                          VoxelRes_mm,verbose=verbose,Absorb=Absorb)

        ## need multi-stuff here! ##
        Ka=np.array(kExt)*(1.-np.array(ScatAlb))

        asymmparam = 0.5*dtheta*np.sum(np.cos(thetas)*np.sin(thetas)*POWER)
        time2=datetime.now()
        print("Finished computing the scattering phase function after %.1f seconds"%((time2-time1).total_seconds()))
        print("------------------------------------")

    else:
        Ka = Density/917.0*(np.nanmean(Fice)/np.nanmean(Fice_Straight))*kIce
        dtheta=np.nanmean(np.gradient(thetas))
        PhaseText='Phase function computed within sample mesh '
        if particlePhase == True:
            PhaseText+='assuming scattering angles are computed between air->ice phase incident angles and exiting photons, \nreplicating whole particle scattering within the medium'
        else:
            PhaseText+='along particle boundaries, replicating the localized phase function (e.g., Xiong et al. 2015)'
        ScatAlb=1.-4.*(np.nanmean(Fice)/np.nanmean(Fice_Straight))*kIce/(917.0*SSA)

    ### Now write the data out to a file!
    today=datetime.now().strftime('%c')
    ## Compute B parameter from Fice
    B=np.nanmean(Fice)/np.nanmean(Fice_Straight)
    with open(OutputFilename, 'w') as file:
       file.write("Optical Properties File Created From %s \n"%VTKFilename)
       file.write("Mesh Description: %s"%SnowMesh.description)


       print('B Parameter: %.3f'%(B))

       file.write("File Created on %s \n"%today)
       file.write("Central Wavelength and refractive index: %s, %.2f \n"%(wavelen,nIce))
       file.write("Number of Photons used in Monte Carlo Sampling: %i \n"%nPhotons)
       if Advanced == True and PF_fromSegmentedParticles == True:
           file.write("Advanced photon-tracking that allows for multiple re-intersections with the particle was used!\n")
       if PF_fromSegmentedParticles == False:
           file.write("!!! Phase Function Computed for boundary (instead of whole particle) scattering !!!\n")
           file.write("!!! This is similar to Xiong et al., 2015. !!!\n")
           file.write("!!! Accordingly, Single Scattering Albedo, Asymmetry Parameter and Absorption Coefficient are not calculated. !!!\n")
       file.write(PhaseText)
       file.write("Sample Volume = %.2f (mm^{3}) \n"%sampleVolumeOut)
       file.write("Estimated Sample Density = %.2f (kg m^{-3}) \n"%Density)
       file.write("Estimated Sample SSA = %.2f (m^{2} kg^{-1}) \n"%SSA)
       file.write("Estimated Sample Grain Diameter = %.2f (mm) \n"%GrainDiam)
       file.write("Max Number Bounces allowed in phase function and Ice Path Sampling: %i \n"%maxBounce)
       file.write("Extinction Coefficient = %.4f (mm^{-1}) \n"%(kExt))
       file.write("Estimated Geometric Asymmetry Parameter (g) = %.2f (-) \n"%asymmparam)
       file.write("Estimated Asymmetry Parameter (g) = %.2f (-) \n"%(0.5*(asymmparam+1.)))
       file.write("Estimated Single Scattering Albedo = %.2f (-)\n"%np.nanmean(ScatAlb))
       file.write("Estimated Absorption Coefficient = %.6f (mm^{-1})\n"%Ka)
       file.write("Mean fractional distance traveled in ice medium (Fice) = %.3f \n"%np.nanmean(Fice))
       file.write("Median fractional distance traveled in ice medium (Fice) = %.3f \n"%np.nanmedian(Fice))
       file.write("Standard Deviation fractional distance traveled in ice medium (Fice) = %.3f \n"%np.nanstd(Fice))

       file.write("Mean Absorption Enhancment Parameter (B) = %.3f \n"%(B))
       file.write("B estimated from Fice and Density B = %.3f \n"%((np.nanmean(Fice)-917./Density*np.nanmean(Fice))/(np.nanmean(Fice)-1)))

       file.write("Absorption scale (eta) =%.2f\n"%((1.+(B-1.)*Density/917.0)))
       file.write("Number of bins in phase function = %i \n"%PhaseBins)
       file.write("Angular Bin Size (radians) = %.5f"%dtheta)
       file.write("------------------------------------------------------  \n")
       file.write("Theta (radians), Phase Function\n")
       for tdx in range(len(thetas)):
           file.write("%.4f, %.4f \n"%(thetas[tdx],POWER[tdx]))

    print("Optical Properties saved to: %s"%OutputFilename)


    if plot == True:
        ax=fig.add_subplot(2,2,4)
        ax.plot(np.degrees(thetas),POWER,label='%s'%wavelen)
        #ax.text(5,500,"g = %.2f"%asymmparam,ha='left')
        ax.set_yscale('log')
        ax.set_xlabel("Theta (degrees)")
        ax.set_ylabel("Power")
        ax.set_xlim(0,180)
        ax.set_ylim(0.01,10000)
        ax.grid()
        ax.legend()
        ax.set_title("Scattering Phase Function",loc='left')

    else:
        fig=None

    return fig

def ConvertVoxel2mm(resolution,units='um'):
    """ This simple function is used to "smartly" convert the voxel resolution from the allowed units
        specified in "allowedUnits" dictionary below to the native millimeters.
        First it checks to see if it is a string with an allowed units value in the character part of the string,
        converts to meters, and then back to mm when returning the value.  If the resolution argument is a float,
        it assumes the units are specified by the optional units argument"""

    import numpy as np
    import re

    allowedUnits={'m':1,'cm':100,'mm':1000,'um':1e6,'nm':1e9}

    if type(resolution) == str:
        res = re.split('([-+]?\d+\.\d+)|([-+]?\d+)', resolution.strip())
        res_f = [r.strip() for r in res if r is not None and r.strip() != '']
        if len(res_f) == 1:
            ## no units, you just put it in as a float.
            Voxel=res_f
        elif len(res_f) == 2:
            Voxel,units=res_f
            Voxel=float(Voxel)
        else:
            print("This is an unacceptable input.")

    if units not in allowedUnits:
        print("%s is not an allowed unit"%units)
    else:
        Voxel=(Voxel/allowedUnits[units])*1000. ## convert to mm

    return Voxel


def ComputeFice(SnowMesh,nIce,kIce,nPhotons,Polar = False, maxBounce = 100,verbose=False,
                PhaseBins=180,particlePhase=False,PF_fromSegmentedParticles=False,raylen = 1000.,TIR=30):
    """Computes the F_{ice} optical property by running a customized version of the
       explicit photon tracking model described in Kaempfer et al., 2007.

       Photons are initialized at random locations on the lateral boundaries, and aimed with random trajectories towards
       the opposite boundary.  They reflect/refract at each air/ice interface according to nIce, until they exit the sample.
       Note that there is no absorption within this model.

       The fraction of the path traveled in ice, is returned for each photon fired through the sample.

       Inputs:
            SnowMesh (CRRELPolyData) - CRREL polydata object with 3D mesh
            nIce (float) - Real part of the refractive index
            nPhotons (float) - Number of photons used to compute F_{ice}
            Polar (optional:bool) - Allows for fresnel reflection / refraction to be have polarity (False, in development)
            maxBounce (optional:integer) - Number of total internal reflections allowed before the particle is killed for computational efficiency.
            verbose (optional:bool) - if True, will print out notifications every 5% of model completion.
            straight (optional:bool) - Computes Fice following a straight ray through the sample (old, leave as false)
            PhaseBins (optional:integer) - Number of Bins used in the phase function
            particlePhase (optional:bool) - True: Compute the phase function from a whole particle perspective |
                                            False: Compute phase function at each boundary collision.
                                            The "whole particle" perspective compares scattering angles during the Fice ray-tracing
                                            assuming that the incident radiation occurs at an air->ice collision, and the scattered angles are
                                            for subsequent ice->air collisions until the trace is transmitted out of the particle.

            PF_fromSegmentedParticles (optional:bool) - If true, assumes that the phase function will be computed on individual segmented
                                                        particles in a different function.  Returns fewer variables.  Note that this is
                                                        distinct from the "particlePhase" option which computes the phase function as part of
                                                        this (Fice) function.

            raylen (optional:float) - length of ray (in mm) to use when checking for intersections.  Smaller values may result in faster code, but could miss intersections.

            TIR (optional:int) - Number of total internal reflections to allow within the ice-phase before killing the photon. Helps reduce the effect of TIR loops on computational expense.

        Returns:
            Fice - (1D array of length: nPhotons) - Fractional Ice Path
            TotalLengths - (1D array of length: nPhotons) the total lengths traveled through the medium for each particle (mm)
            missed - number of photons that were fired through the medium with no collisions (note that F_ice = 0 for these cases)
            POWER  - (1D array of length: nBins) - Phase function Power
            thetas - (1D array of length: nBins) - Phase function bin centroid (radians)
            Fice_Straight (1D array of length: nBins) - Fractional Ice Path assuming a straight ray through the medium.

    """

    print("Running Photon Tracking to Determine F_{ice}...")
    time1=datetime.now()
    normalsMesh=SnowMesh.GetNormalsMesh()
    obbTree=SnowMesh.GetObbTree()
    percent5=int(0.05*nPhotons)
    Fice=[]
    Fice_Straight=[]
    TotalLengths=[]
    missed=0

    mfp = 0  # mean free path
    for ii in range(nPhotons):
        axis=np.random.randint(0,6) ## Choose random axis (x,y,z)
        if axis == 0: ## If x Axis
            x1,x2=SnowMesh.xBounds[0],SnowMesh.xBounds[1]
            y1,y2=np.random.uniform(SnowMesh.yBounds[0],SnowMesh.yBounds[1],2)
            z1,z2=np.random.uniform(SnowMesh.zBounds[0],SnowMesh.zBounds[1],2)
        elif axis == 1: ## If y Axis
            x1,x2=np.random.uniform(SnowMesh.xBounds[0],SnowMesh.xBounds[1],2)
            y1,y2=SnowMesh.yBounds[0],SnowMesh.yBounds[1]
            z1,z2=np.random.uniform(SnowMesh.zBounds[0],SnowMesh.zBounds[1],2)
        elif axis == 2: ## If z Axis
            x1,x2=np.random.uniform(SnowMesh.xBounds[0],SnowMesh.xBounds[1],2)
            y1,y2=np.random.uniform(SnowMesh.yBounds[0],SnowMesh.yBounds[1],2)
            z1,z2=SnowMesh.zBounds[0],SnowMesh.zBounds[1]

        elif axis == 3: ## If x Axis
            x1,x2=SnowMesh.xBounds[1],SnowMesh.xBounds[0]
            y1,y2=np.random.uniform(SnowMesh.yBounds[0],SnowMesh.yBounds[1],2)
            z1,z2=np.random.uniform(SnowMesh.zBounds[0],SnowMesh.zBounds[1],2)
        elif axis == 4: ## If y Axis
            x1,x2=np.random.uniform(SnowMesh.xBounds[0],SnowMesh.xBounds[1],2)
            y1,y2=SnowMesh.yBounds[1],SnowMesh.yBounds[0]
            z1,z2=np.random.uniform(SnowMesh.zBounds[0],SnowMesh.zBounds[1],2)
        else: ## If z Axis
            x1,x2=np.random.uniform(SnowMesh.xBounds[0],SnowMesh.xBounds[1],2)
            y1,y2=np.random.uniform(SnowMesh.yBounds[0],SnowMesh.yBounds[1],2)
            z1,z2=SnowMesh.zBounds[1],SnowMesh.zBounds[0]

        p11=[x1,y1,z1]
        p22=[x2,y2,z2]

        pDir=RTcode.pts2unitVec(p11,p22)

        ##Run the photon tracking model through the code!
        if PF_fromSegmentedParticles == True:
            TotalIceLength,TotalLength,intersections,Fstraight,first_length,num_scatter_events=RTcode.TracktoAbs(p11,pDir,nIce,normalsMesh,obbTree,
                    nAir=1.00003,polar=Polar,maxBounce=maxBounce,raylen=raylen,MaxTIRbounce=TIR)

        else:
            if ii == 0:
                bins=np.linspace(0,np.pi,PhaseBins)
                dtheta=np.abs(bins[1]-bins[0])
                bins=np.cos(bins)
                binCenters=(bins[:-1]+bins[1:])/2.
                POWER=np.zeros_like(binCenters)

            TotalIceLength,TotalLength,intersections,weights,COSPHIS,Fstraight,num_scatter_events=RTcode.TracktoAbsWPhaseF(p11,pDir,nIce,kIce,normalsMesh,obbTree,
                    nAir=1.00003,polar=Polar,maxBounce=maxBounce,particle=particlePhase,raylen=raylen,MaxTIRbounce=TIR)


            for cdx,c in enumerate(COSPHIS):
                index=np.argmin(np.abs(binCenters-c))
                POWER[index]+=weights[cdx]

        if TotalLength == 0:
            missed+=1
            #Fice.append(0.0)
            continue

        if ii%percent5 == 0:
            if verbose == True:
                timeNow=datetime.now()
                print("Total percent complete =%.1f | Time Elapsed: %.1f seconds"%(100.*float(ii)/nPhotons,(timeNow-time1).total_seconds()))

        if num_scatter_events > 0:
            mfp += (TotalLength/num_scatter_events)
        if TotalLength > 0:
            Fice.append(TotalIceLength/TotalLength)
            Fice_Straight.append(Fstraight)
            TotalLengths.append(TotalLength)

    if PF_fromSegmentedParticles == True:
        kExt = 1./(mfp/nPhotons)
        return Fice,Fice_Straight,TotalLengths,kExt,missed

    else:
        thetas=np.arccos(binCenters)
        N = np.sum(POWER[:])
        dOmega = np.sin(thetas[:])*dtheta*2*np.pi
        POWER=4.*np.pi*POWER[:]/(N*dOmega)
        kExt = 1./(mfp/nPhotons)

        return Fice,TotalLengths,POWER,thetas,Fice_Straight,kExt,missed



def ReadMesh(VTKfile,VoxelResolution,Tolerance=0.001,verbose=False,description='Real Snow Mesh',smooth=0):
    """Helper function to read in data from a VTK file and format the 3D mesh
       into the CRRELPolyData Format and compute some physical properties for the mesh using pyvista.

       Inputs:
            VTKfile (string) - Filepath of .vtk or .stl file.
            VoxelResolution - (float) Voxel resolution (in mm)

            Tolerance (optional/float) - Tolerance used in vtk.obbTree to determine how close a ray needs to be to a mesh "cell" to
                            count as an intersection.  Recommended that a value of no larger than 0.001 is used, especially for
                            versions of vtk < 9.1.x. 0.01 seems to work for vtk versions > 9.1.
                            To be safe, I have set an assertion that this value is < 0.02.

            verbose (optional) - Boolean Flag - if true , will print out computed physical properites of the mesh.

            description (optional/string) - You can add a descriptive name to the mesh that will be passed to the optical file.

        Returns:
            Mesh (CRRELPolyData) - Mesh Data
            sampleVolumeOut (float) - Total volume bounded by mesh (in mm^3)
            Density (float) - Snow density of the mesh (kg/m^3)
            SSA (float) - Snow Specific Surface Area (m^2/kg)
            GrainDiam (float) - GrainDiameter computed from SSA (in mm)
       """


    ## Read in Mesh Data! ##
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(VTKfile)
    reader.Update()
    shell = reader.GetOutput()
    #help(shell.BOUNDING_BOX())
    xBounds=tuple(shell.GetBounds()[:2])
    yBounds=tuple(shell.GetBounds()[2:4])
    zBounds=tuple(shell.GetBounds()[4:])

    sampleVolumeOut=(xBounds[1]-xBounds[0])*(yBounds[1]-yBounds[0])*(zBounds[1]-zBounds[0]) ## in Cubic mm

    pvob=pv.wrap(shell)
    if smooth > 0:
        pvob = pvob.smooth(n_iter=smooth)

    snowMass=pvob.volume/(1000**3)*917.0

    cellAreas=pvob.compute_cell_sizes(length=False, area=True, volume=False, progress_bar=False)
    SfcArea=np.nansum(cellAreas.cell_arrays["Area"])/(1000**2)

    SampleVol=sampleVolumeOut/(1000**3.)  ## Convert to Meters to compute Snow Density.
    Density=snowMass/SampleVol ## Compute Snow Density
    SSA=SfcArea/snowMass ## Compute SSA (m^2/kg)
    GrainDiam=6./(917.0*SSA)*1000. ## Estimate Optical Diameter from SSA and convert to mm

    Mesh=CRRELPolyData._CRRELPolyData(shell,xBounds,yBounds,zBounds,VoxelResolution,
                                      Density,description=description,Tolerance=Tolerance)

    if verbose == True:
        print("Density, %.2f kg/m^3"%Density)
        print("SSA, %.2f m^2/kg"%SSA)
        print("Grain Size, %.2f mm"%GrainDiam)
        print("Sample Volume (mm^3) %.2f"%(SampleVol*(1000**3)))


    return Mesh,sampleVolumeOut,Density,SSA,GrainDiam

def ComputeScatteringPhaseFunc(PhaseBins,GrainSamples,GrainFiles,nPhotons,SnowMesh,nIce,kIce,
                               VoxelRes_mm,verbose=False,Absorb=False,
                               Advanced = True,TrackThresh=0.1,TrackDepthMax=4):

    """ Function that computes scattering phase function from individual snow grains within the medium
        Follows method described in Letcher et al. 2021 (in prep).  Photons are fired at a particle with random directions
        and the angles and photon weights are tracked relative to the incident angle.

        Inputs:
            PhaseBins (integer) - Number of bins used to resolve the scattering phase function
            GrainSamples (integer) - Number of grain samples to use in computing the phase function
            GrainFiles (string) - Path to folder containing .vtk or .stl files for individual grains
            nPhotons (integer) - Number of photons used for EACH grain sample (total photons fired = nPhotons*GrainSamples)
            SnowMesh (CRRELPolyData) - CRRELPolyData object containing 3D mesh information
            nIce (float) - (Real part of the Ice refractive index)
            kIce (float) - (Absorption Coefficient of Ice)
            VoxelRes_mm (float) - Voxel Resolution (in mm)

            verbose (optional) - Boolean flag, if True, will print out % complete notices when a sample grain is finished
            Absorb (optional) - Boolean flag, if True, intra-particle absorption will be included in the phase function (Usually negligle, but matters for longer wavelenghts)

        Returns:
            POWER (array) - Scattering Phase function: p(cos(THETA)) array with length PhaseBins.
            thetas (array) - center angles for each phase bin (radians)
            PhaseText (string) - Text containing metadata about how the phase function was calculated - Saved to output file
            dtheta (float) - bin size (radians)
    """

    import pyvista as pv

    bins=np.linspace(0,np.pi,PhaseBins)
    dtheta=np.abs(bins[1]-bins[0])
    bins=np.cos(bins)
    binCenters=(bins[:-1]+bins[1:])/2.
    POWER=np.zeros_like(binCenters)

    ScatterAlb=[]
    time1=datetime.now()

    totalHits=0
    for sdx in range(GrainSamples):
        fileIdx=np.random.randint(0,len(GrainFiles))
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(GrainFiles[fileIdx])
        reader.Update()
        shell = reader.GetOutput()
        #help(shell.BOUNDING_BOX())
        xBounds=tuple(shell.GetBounds()[:2])
        yBounds=tuple(shell.GetBounds()[2:4])
        zBounds=tuple(shell.GetBounds()[4:])



        CRRELPD=CRRELPolyData._CRRELPolyData(shell,xBounds,yBounds,zBounds,VoxelRes_mm,917,description='Grain Sample')

        normalsMesh=CRRELPD.GetNormalsMesh()
        obbTree=CRRELPD.GetObbTree()

        ## Choosing a random position is a little tricky:
        ## We want to choose an intial position on one of the boundary planes and fire it in a random direction
        ## towards the other plane.  So, we choose a random axis to have fixed start/end points, and randomize the other two axes.
        scaleScat=0
        for ii in range(nPhotons):
            axis=np.random.randint(0,6) ## Choose random axis (x,y,z)
            if axis == 0: ## If x Axis
                x1,x2=SnowMesh.xBounds[0],SnowMesh.xBounds[1]
                y1,y2=np.random.uniform(SnowMesh.yBounds[0],SnowMesh.yBounds[1],2)
                z1,z2=np.random.uniform(SnowMesh.zBounds[0],SnowMesh.zBounds[1],2)
            elif axis == 1: ## If y Axis
                x1,x2=np.random.uniform(SnowMesh.xBounds[0],SnowMesh.xBounds[1],2)
                y1,y2=SnowMesh.yBounds[0],SnowMesh.yBounds[1]
                z1,z2=np.random.uniform(SnowMesh.zBounds[0],SnowMesh.zBounds[1],2)
            elif axis == 2: ## If z Axis
                x1,x2=np.random.uniform(SnowMesh.xBounds[0],SnowMesh.xBounds[1],2)
                y1,y2=np.random.uniform(SnowMesh.yBounds[0],SnowMesh.yBounds[1],2)
                z1,z2=SnowMesh.zBounds[0],SnowMesh.zBounds[1]

            elif axis == 3: ## If x Axis
                x1,x2=SnowMesh.xBounds[1],SnowMesh.xBounds[0]
                y1,y2=np.random.uniform(SnowMesh.yBounds[0],SnowMesh.yBounds[1],2)
                z1,z2=np.random.uniform(SnowMesh.zBounds[0],SnowMesh.zBounds[1],2)
            elif axis == 4: ## If y Axis
                x1,x2=np.random.uniform(SnowMesh.xBounds[0],SnowMesh.xBounds[1],2)
                y1,y2=SnowMesh.yBounds[1],SnowMesh.yBounds[0]
                z1,z2=np.random.uniform(SnowMesh.zBounds[0],SnowMesh.zBounds[1],2)
            else: ## If z Axis
                x1,x2=np.random.uniform(SnowMesh.xBounds[0],SnowMesh.xBounds[1],2)
                y1,y2=np.random.uniform(SnowMesh.yBounds[0],SnowMesh.yBounds[1],2)
                z1,z2=SnowMesh.zBounds[1],SnowMesh.zBounds[0]

            p11=[x1,y1,z1]
            p22=[x2,y2,z2]

            if isinstance(kIce,list) == True:
                if sdx == 0 and ii == 0:
                    ## IF THIS IS A LIST, COMPUTE PHASE FUNCTION FOR ALL WAVELENGTHS TO GET WAVELENGTH Dependent SingleScat Albedo!
                    ## NOTE, the PHASE FUNCTION SAVED to the OUTPUT will be the "Middle" value in the list!
                    ScatterAlb=np.zeros([len(kIce),GrainSamples])
                    print("Computing Phase function for multiple wavelenghts...")
                    print("This will take some extra time.")
                    print("Note that the phase function saved to the output file will be at the median value.")

                for wdx, w in enumerate(kIce):
                    if Advanced == False:
                        weights,COSPHIS,intersections,ScatAlb,dummy=RTcode.ParticlePhaseFunction(CRRELPD,p11,p22,normalsMesh,obbTree,nIce,w,absorb=Absorb)
                    else:
                        weights,COSPHIS,intersections,ScatAlb,dummy=RTcode.AdvParticlePhaseFunction(CRRELPD,p11,p22,normalsMesh,obbTree,nIce,w,absorb=Absorb,
                                                                                         TrackThresh=TrackThresh,Tdepth=TrackDepthMax)
                    if dummy == True or len(COSPHIS) ==0:
                        continue

                    if wdx == 0: ## only add this scale on the first wavelenght.!
                        scaleScat+=1
                    if wdx == int(len(kIce)/2):
                        ## ONLY SAVE THE PHASE FUNCTION IF its the approximate middle wavelength in the list!
                        for cdx,c in enumerate(COSPHIS):
                            index=np.argmin(np.abs(binCenters-c))
                            POWER[index]+=weights[cdx]

                        totalHits+=len(COSPHIS)

                    ScatterAlb[wdx,sdx]+=ScatAlb

            else:

                if ii == 0 and sdx == 0:
                    print(kIce,nIce)

                if Advanced == False:
                    weights,COSPHIS,intersections,ScatAlb,dummy=RTcode.ParticlePhaseFunction(CRRELPD,p11,p22,normalsMesh,obbTree,nIce,kIce,absorb=Absorb)
                else:
                    weights,COSPHIS,intersections,ScatAlb,dummy=RTcode.AdvParticlePhaseFunction(CRRELPD,p11,p22,normalsMesh,obbTree,nIce,kIce,absorb=Absorb,
                                                                                  TrackThresh=TrackThresh,Tdepth=TrackDepthMax)
                if dummy == True or len(COSPHIS) ==0:
                    continue

                totalHits+=len(COSPHIS)
                for cdx,c in enumerate(COSPHIS):
                    index=np.argmin(np.abs(binCenters-c))
                    POWER[index]+=weights[cdx]

                ScatterAlb.append(ScatAlb)

        if isinstance(kIce,list) == True:
            ScatterAlb[:,sdx]=ScatterAlb[:,sdx]/float(scaleScat)  ## thus it becomes an array averaged over all photons!

        if verbose == True:
            timeNow=datetime.now()
            print("Total percent complete =%.1f | Time Elapsed: %.1f seconds"%(100.*(1.+sdx)/(GrainSamples),(timeNow-time1).total_seconds()))

    thetas=np.arccos(binCenters)

    print(totalHits)
    N = np.sum(POWER[:])
    dOmega = np.sin(thetas[:])*dtheta*2*np.pi
    POWER=4.*np.pi*POWER[:]/(N*dOmega)

    if isinstance(kIce,list) == True:
        ScatterAlb=np.nanmean(ScatterAlb,axis=1) ## average over all particles, save only wave length dependence
    else:
        ScatterAlb=np.nanmean(ScatterAlb) ## Otherwise, just return a single value.

    PhaseText='Phase function computed using "particle-oriented" approach with %s grain samples and %s photons per grain \nAbsorption included in phase function = %s \n'%(GrainSamples,nPhotons,Absorb)

    return POWER,thetas,PhaseText,dtheta,ScatterAlb


def convertStrWave2float(wavelength,outunit='nm'):
    import numpy as np
    import re
    allowedUnits={'m':1,'cm':100,'mm':1000,'um':1e6,'nm':1e9}

    if type(wavelength) == str:
        res = re.split('([-+]?\d+\.\d+)|([-+]?\d+)', wavelength.strip())
        res_f = [r.strip() for r in res if r is not None and r.strip() != '']
        if len(res_f) == 1:
            ## no units, you just put it in as a float.
            units=outunit
            wavelength=res_f
        elif len(res_f) == 2:
            wavelength,units=res_f
            wavelength=float(wavelength)
        else:
            print("This is an unacceptable input.")
            units=outunit

    if units not in allowedUnits:
        print("%s is not an allowed unit"%units)

    return wavelength*allowedUnits[outunit]/allowedUnits[units]
