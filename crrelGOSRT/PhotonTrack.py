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
                                 Advanced = True,TrackThresh=0.1,TrackDepthMax=4,straight=False,
                                 Multi = False, Polar = False,FiceFromDensity=False):

    """
        This function calls individual sub-functions to get optical properties and saves them to file for
        1D spectral albedo model:

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

            Advanced (optional): If True, allows for multiple interactions with an irregularly shaped particle.
            Specifically, photons that exit the particle with remaining energy greater than some threshold are checked for re-intersections with the particle.
            Any reintersections are subsequently tracked.  Can be particularly useful for hollow facets or other hollow particles where reintersections are likely.
            Expect significantly longer computational time when this is on.

            TrackThresh (optional): Minimum energy threshold for which to track photons for re-intersections.

            TrackDepthMax (optional): Maximum number of times ("depths") for which to track for reinterections.
                Generally, most particles will go less than 3 reintersections before running out of steam, but this allows for a hard limit.

            straight (optional): if True computes F_{ice} by assuming a straight ray (not reccomended, but can speed up computational time.)

            !------------------------------------------------------------------------------------------------!
            Developmental Options (Note, these are not rigorously tested, and should not be used in operation)
            !------------------------------------------------------------------------------------------------!

            Multi (optional/developmental): Boolean flag to turn on the use of pvistas "vectorized" ray casting in the extinction coefficient and F_{ice}
                                            calculations.  Should lead to speed up, but doesn't quite yield the same answers as the tested sequential versions.
                                            This also requires additional libraries to be installed prior to use, and a modification to the Pyvista python code,
                                            so it is not recommended for use at this time -> see: https://docs.pyvista.org/user-guide/optional_features.html
                                            for more information

            Polar (optional/developmental): Adds polarization to the scattering phase function calculation.
                                            Currently, no polarization data is actually saved to the output data file,
                                            but can be used to look at linear polarization of a specific particle shape by plotting.

            FiceFromDensity (optional/developmental): Boolean Flag (True/False).  Because the computation from Fice runs a version of the Kaempfer et al. 2007 model,
                                                      it requires the most time to run.  Fice is also very strongly correlated to sample density (Letcher et al. 2021)
                                                      So, this option allows users to skip running the Kaemfer model, and simply set Fice according to sample density.
                                                      Setting this to True this can save 1-2 hours of run time for a given simulation, but then the ray-tracing is run
                                                      for this option.

        Returns:
            Saves optical properties to the OutputFilename, and returns a figure showing the optical properites
            if plot == false, then the returned figure is "None"
        """

    GrainFiles=glob.glob(GrainFolder+'Grain*')
    VTKfile=glob.glob(VTKFilename)

    ## Comment out for speedier run time with single grain, instead of full mesh.  Good for debugging.
    #VTKFilename=GrainFiles[1]

    ## Perform initial checks ##
    assert len(GrainFiles) > 0, "There are no Grain Files in %s, I cannot compute the phase function without grain files."%GrainFolder
    assert len(VTKfile) > 0, "There is no VTK Mesh file with the name %s.  I cannot compute the optical properties without a mesh file."%VTKFilename

    ## put a print statement here.

    VoxelRes_mm=ConvertVoxel2mm(VoxelRes,VoxelUnits)
    if verbose == True:
        print("Voxel Resolution = %.5f mm"%VoxelRes_mm)

    ## Get SnowMesh
    SnowMesh,sampleVolumeOut,Density,SSA,GrainDiam=ReadMesh(VTKFilename,VoxelRes_mm,verbose=verbose)

    print("Finished loading mesh ... ")
    SnowMesh.AssignMaterial('ice',filePath=MaterialPath)  ##Assign Material

    distances=[]
    dd=VoxelRes_mm
    while(dd < 1.05*(SnowMesh.xBounds[1]-SnowMesh.xBounds[0])):
        dd=dd*distScale
        distances.append(dd)
    distances=np.array(distances)
    nIce,kIce,Nc=SnowMesh.GetRefractiveIndex(wavelen,units=wavelenUnits)

    ## Get Extinction Coefficient... ##
    time1=datetime.now()
    popt,distances,extinction,bad=ComputeExtinction(SnowMesh,distances,kIce,nPhotons,Multi=Multi,verbose=verbose)
    time2=datetime.now()

    text="SSA = %.1f m^2/kg \nrho_s = %.1f kg/m^3 \nSample Vol = %.1f mm^3"%(SSA,Density,sampleVolumeOut)

    if plot == True:
        plotter = pv.Plotter(off_screen=True, notebook=False,window_size =[1024, 1024])
        plotter.add_text(text, font_size=20)
        plotter.add_text(VTKFilename,font_size=20,position='lower_left',color='k')
        plotter.add_mesh(pv.wrap(SnowMesh.GetPolyData()),
               show_edges=False, opacity=1., color="w",
               diffuse=0.8, smooth_shading=True,specular=0.2)

        _, perspective = plotter.show(screenshot=True)


        fig=plt.figure(figsize=(9,9))
        ax=fig.add_subplot(2,2,1)
        ax.scatter(distances,extinction,label='POE')
        ax.plot(distances, CurveFit(distances, *popt), 'r-',label='fit: Ke=%.2f' % (popt[0]))
        ax.legend()
        ax.set_ylabel("POE")
        ax.set_xlabel("$d$ (mm)")
        plt.title("Curve fit for $\gamma_{ext}$")
        ax.grid()

        ax=fig.add_subplot(2,2,2)
        ax.imshow(perspective)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    print("Found Extinction Coefficient = %.2f after %.1f seconds"%(popt[0],(time2-time1).total_seconds()))
    print("Total Bad Photons =%i --> %.3f percent of all photons"%(bad,len(distances)*100.*bad/nPhotons))
    print("------------------------------------")

    ## Next Up, F_Ice and PhaseFunction
    ###
    time1=datetime.now()
    if FiceFromDensity == False:
        Fice, TotalLengths,missed=ComputeFice(SnowMesh,nIce,nPhotons,verbose=verbose,maxBounce=maxBounce,straight=straight)
    else:
        print("WARNING, Fice Computed from density, not ray-tracing!")
        Fice, missed = 0.00074*Density + 0.25, 0.0
    time2=datetime.now()

    print("Found F_{ice} = %.2f after %.1f seconds"%(np.nanmean(Fice),(time2-time1).total_seconds()))
    print("Total Missed Photons =%i --> %.3f percent of all photons"%(missed,100.*missed/nPhotons))
    print("------------------------------------")

    #####

    if plot == True and FiceFromDensity == False:
        ax=fig.add_subplot(2,2,3)
        ax.hist(Fice,edgecolor='k')
        ax.axvline(np.nanmean(Fice),color='r',lw=2.0,ls='--')
        ax.set_xlabel("$F_{ice}$",fontsize=12)
        ax.set_ylabel("Count")
        ax.grid()
        plt.title("Histogram of $F_{ice}$",loc='left')

    print("Computing Scattering Phase Function using %s Grain Samples..."%GrainSamples)

    time1=datetime.now()
    POWER,thetas,PhaseText,dtheta=ComputeScatteringPhaseFunc(PhaseBins,GrainSamples,GrainFiles,nPhotons,SnowMesh,nIce,kIce,
                                                      VoxelRes_mm,verbose=verbose,Absorb=Absorb)

    asymmparam = 0.5*dtheta*np.sum(np.cos(thetas)*np.sin(thetas)*POWER)
    time2=datetime.now()
    print("Finished computing the scattering phase function after %.1f seconds"%((time2-time1).total_seconds()))
    print("------------------------------------")

    ### Now write the data out to a file!
    today=datetime.now().strftime('%c')
    with open(OutputFilename, 'w') as file:
       file.write("Optical Properties File Created From %s \n"%VTKFilename)
       file.write("File Created on %s \n"%today)
       if Multi == True:
           file.write("WARNING: Vectorized ray-casting through pyvista was used to generated k_ext and F_{ice}!\n")
       file.write("Central Wavelength and refractive index: %s, %.2f \n"%(wavelen,nIce))
       file.write("Number of Photons used in Monte Carlo Sampling: %i \n"%nPhotons)
       if FiceFromDensity == True:
           file.write("WARNING: F_{ice} computed using linear regression formula: Fice = 0.00074*rho_{s} + 0.25 instead of ray-tracing!\n")
       if Advanced == True:
           file.write("Advanced photon-tracking that allows for multiple re-intersections with the particle was used!\n")
       file.write(PhaseText)
       file.write("Sample Volume = %.2f (mm^{3}) \n"%sampleVolumeOut)
       file.write("Estimated Sample Density = %.2f (kg m^{-3}) \n"%Density)
       file.write("Estimated Sample SSA = %.2f (m^{2} kg^{-1}) \n"%SSA)
       file.write("Estimated Sample Grain Diameter = %.2f (mm) \n"%GrainDiam)
       file.write("Max Number Bounces allowed in phase function and Ice Path Sampling: %i \n"%maxBounce)
       file.write("Extinction Coefficient = %.4f (mm^{-1}) \n"%(popt[0]))
       file.write("Estimated Asymmetry Parameter (g) = %.2f (-) \n"%asymmparam)
       if FiceFromDensity == False:
           file.write("Mean fractional distance traveled in ice medium (Fice) = %.3f \n"%np.nanmean(Fice))
           file.write("Median fractional distance traveled in ice medium (Fice) = %.3f \n"%np.nanmedian(Fice))
           file.write("Standard Deviation fractional distance traveled in ice medium (Fice) = %.3f \n"%np.nanstd(Fice))
       else:
           file.write("Mean fractional distance traveled in ice medium (Fice) = %.3f \n"%Fice)
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


def ComputeFice(SnowMesh,nIce,nPhotons,Polar = False, maxBounce = 100,verbose=False,straight=False):
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

            verbose (optional) - if True, will print out notifications every 5% of model completion.

        Returns:
            Fice - (1D array of length: nPhotons)
            missed - number of photons that were fired through the medium with no collisions (note that F_ice = 0 for these cases)

    """

    print("Running Photon Tracking to Determine F_{ice}...")
    time1=datetime.now()
    normalsMesh=SnowMesh.GetNormalsMesh()
    obbTree=SnowMesh.GetObbTree()
    percent5=int(0.05*nPhotons)
    Fice=[]
    TotalLengths=[]
    missed=0
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
        if straight == False:
            TotalIceLength,TotalLength,intersections=RTcode.TracktoAbs(p11,pDir,nIce,normalsMesh,obbTree,
                    nAir=1.00003,raylen=1000,polar=Polar,maxBounce=maxBounce)
        else:
            TotalIceLength,TotalLength=RTcode.TracktoAbsStraight(p11,pDir,nIce,normalsMesh,obbTree,
                    nAir=1.00003,raylen=1000)

        if TotalLength == 0:
            missed+=1
            #Fice.append(0.0)
            continue

        if ii%percent5 == 0:
            if verbose == True:
                timeNow=datetime.now()
                print("Total percent complete =%.1f | Time Elapsed: %.1f seconds"%(100.*float(ii)/nPhotons,(timeNow-time1).total_seconds()))


        Fice.append(TotalIceLength/TotalLength)
        TotalLengths.append(TotalLength)


    return Fice,TotalLengths,missed

def ComputeExtinction(SnowMesh,distances,kIce,nPhotons,Multi=False,verbose=False,trim=False):
    """Computes the k_{ext} optical property by running a customized version of the
       curve fitting/ray tracing model described in Xiong et al. (2015):

       Photons are initialized within the sample at random locations and given random trajectories.
       Photons are launched a specified distance, and if they hit an air/ice boundary the probability of extinction over that distance
       is 1, otherwise it is zero.  In the special case where there is no interaction, but the distance is traveled within the ice,
       the probability of extinction is related to the absorption coefficient, yielding a minor wavelength dependence.
       the extinction coefficient is determined by fitting Beer's law to the extinction probability.


       Inputs:
            SnowMesh (CRRELPolyData) - CRREL polydata object with 3D mesh
            distances (array) - Array of specified distances used to sample extinction probabilities
            kIce (float) - absorption coefficient of ice
            nPhotons (float) - Number of photons used to compute F_{ice}

            Multi (optiona/developmental) - Uses pyvista's vectorized ray-casting.  In development/not recommended  for use.
            verbose (optional) - if True, will print out notifications every 5% of model completion.


        Returns:
            popt - Extinction Coefficient determined from curve fit
            extinction (array) - Probability of extinction at each distance
            bad - total number of "bad" photons.  Occasionally, something goes wrong with the mesh that causes the ray-casting to crash.

    """
    time1=datetime.now()
    print("Finding extinction coefficient for medium...")
    print("------------------------------------")
    extinction=np.zeros_like(distances)
    bad=0
    percent5=int(0.05*nPhotons)

    if Multi == True:
        x1,x2=np.random.uniform(SnowMesh.xBounds[0],SnowMesh.xBounds[1],2)
        y1,y2=np.random.uniform(SnowMesh.yBounds[0],SnowMesh.yBounds[1],2)
        z1,z2=np.random.uniform(SnowMesh.zBounds[0],SnowMesh.zBounds[1],2)

        pSource=np.array([[x1[i],y1[i],z1[i]] for i in range(nPhotons)])
        vectors=np.array([RTcode.pts2unitVec([x1[i],y1[i],z1[i]], [x2[i],y2[i],z2[i]]).squeeze() for i in range(nPhotons)])

        dists,intersections,ind_tri,ind_ray,pvMesh=RTcode.TracktoExt_multi(SnowMesh,pSource,vectors)

        dists=np.ma.masked_invalid(dists).compressed()

        vectors = np.array(vectors)[ind_ray,:]
        pSource = np.array(pSource)[ind_ray,:]
        for ddx, d in enumerate(distances):
            mask=np.ma.masked_greater(dists,d).filled(0.0)/dists

            ZeroInds=np.where(mask == 0)
            ZeroInds=np.squeeze(ZeroInds)

            Asource=pSource[ZeroInds]

            PP=pv.PolyData(Asource)
            select = PP.select_enclosed_points(pvMesh,check_surface=False)
            inds=np.where(select['SelectedPoints'] == 1)
            mask[ZeroInds][inds]=1.-np.exp(-kIce*d)

            extinction[ddx]= np.nanmean(mask)
            if ddx%percent5 == 0:
                if verbose == True:
                    timeNow=datetime.now()
                    print("Total percent complete =%.1f | Time Elapsed: %.1f seconds"%(100.*(ddx+1.)/nPhotons,(timeNow-time1).total_seconds()))
    else:
        dists=[]
        dots=[]
        for ii in range(nPhotons):
            x1,x2=np.random.uniform(SnowMesh.xBounds[0],SnowMesh.xBounds[1],2)
            y1,y2=np.random.uniform(SnowMesh.yBounds[0],SnowMesh.yBounds[1],2)
            z1,z2=np.random.uniform(SnowMesh.zBounds[0],SnowMesh.zBounds[1],2)

            p11=[x1,y1,z1]
            pDir=np.random.uniform(-1,1,3)

            dist,dot=RTcode.TracktoExt(SnowMesh,p11,pDir,raylen=(SnowMesh.xBounds[1]-SnowMesh.xBounds[0]))
            # except:
            #     if verbose == True:
            #         print("Bad Photon! Something weird happened, so I'm skipping this!")
            #     bad+=1.
            #     continue
            dists.append(dist)
            dots.append(dot)

            if ii%percent5 == 0:
                if verbose == True:
                    timeNow=datetime.now()
                    print("Total percent complete =%.1f | Time Elapsed: %.1f seconds"%(100.*(ii+1.)/nPhotons,(timeNow-time1).total_seconds()))

        dots=np.array(dots)
        dists=np.array(dists)
        for ddx, d in enumerate(distances):
            mask=np.ma.masked_greater(dists,d).filled(0.0)/dists
            mask[mask==np.nan]=0.0

            ZeroInds=np.where(mask == 0)
            ZeroInds=np.squeeze(ZeroInds)

            NowDots=dots[ZeroInds]
            Absorb=np.ma.masked_less_equal(NowDots,0).filled(0.0)
            Absorb[Absorb !=0.0]=1.-np.exp(-kIce * d)

            mask[ZeroInds]=Absorb

            extinction[ddx]= np.nanmean(mask)


    distances=np.array(distances)

    if trim == True:
        UseDists=np.where(extinction < 1.00)
        distances=distances[UseDists]
        extinction=extinction[UseDists]

    popt, pcov = curve_fit(CurveFit, distances,extinction,p0=(0.5))
    return popt,distances,extinction,bad

def CurveFit(x,ke):
    """This function defines the function used to curve fit the extinction coefficient"""
    return 1.-np.exp(-ke*x)


def ReadMesh(VTKfile,VoxelResolution,verbose=False):
    """Helper function to read in data from a VTK file and format the 3D mesh
       into the CRRELPolyData Format and compute some physical properties for the mesh using pyvista.

       Inputs:
            VTKfile (string) - Filepath of .vtk or .stl file.
            VoxelResolution - (float) Voxel resolution (in mm)

            verbose (optional) - Boolean Flag - if true , will print out computed physical properites of the mesh.


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
    snowMass=pvob.volume/(1000**3)*917.0

    cellAreas=pvob.compute_cell_sizes(length=False, area=True, volume=False, progress_bar=False)
    SfcArea=np.nansum(cellAreas.cell_arrays["Area"])/(1000**2)

    SampleVol=sampleVolumeOut/(1000**3.)  ## Convert to Meters to compute Snow Density.
    Density=snowMass/SampleVol ## Compute Snow Density
    SSA=SfcArea/snowMass ## Compute SSA (m^2/kg)
    GrainDiam=6./(917.0*SSA)*1000. ## Estimate Optical Diameter from SSA and convert to mm

    Mesh=CRRELPolyData._CRRELPolyData(shell,xBounds,yBounds,zBounds,VoxelResolution,Density,description='REAL Snow Mesh')

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

    bins=np.linspace(0,np.pi,PhaseBins)
    dtheta=np.abs(bins[1]-bins[0])
    bins=np.cos(bins)
    binCenters=(bins[:-1]+bins[1:])/2.
    POWER=np.zeros_like(binCenters)

    time1=datetime.now()
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

            if Advanced == False:
                weights,COSPHIS,intersections,dummy=RTcode.ParticlePhaseFunction(CRRELPD,p11,p22,normalsMesh,obbTree,nIce,kIce,absorb=Absorb)
            else:
                weights,COSPHIS,intersections,dummy=RTcode.AdvParticlePhaseFunction(CRRELPD,p11,p22,normalsMesh,obbTree,nIce,kIce,absorb=Absorb,
                                                                                 TrackThresh=TrackThresh,Tdepth=TrackDepthMax)
            if dummy == True or len(COSPHIS) ==0:
                continue

            for cdx,c in enumerate(COSPHIS):
                index=np.argmin(np.abs(binCenters-c))
                POWER[index]+=weights[cdx]

        if verbose == True:
            timeNow=datetime.now()
            print("Total percent complete =%.1f | Time Elapsed: %.1f seconds"%(100.*(1.+sdx)/GrainSamples,(timeNow-time1).total_seconds()))

    thetas=np.arccos(binCenters)
    N = np.sum(POWER[:])
    dOmega = np.sin(thetas[:])*dtheta*2*np.pi
    POWER=4.*np.pi*POWER[:]/(N*dOmega)

    PhaseText='Phase function computed using "particle-oriented" approach with %s grain samples and %s photons per grain \nAbsorption included in phase function = %s \n'%(GrainSamples,nPhotons,Absorb)

    return POWER,thetas,PhaseText,dtheta
