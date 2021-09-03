import sys
sys.path.append('C:\\Users\\RDCRLJTP\\Documents\\Projects\\Snow_Optics\\Code\\CRREL-GOSRT\\main')
import DrawShapes,CRRELPolyData
import RenderFunctions
import RTcode
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

def HeyNey(PHI,g=0.847):
    COSPHI=np.cos(PHI)
    P=(1.-g**2.)/((1.+g**2.-2.*g*COSPHI)**(3./2.))
    return P

## paths -->
## Parent path
parentPath='C:\\Users\\RDCRLJTP\\Documents\\Projects\\Snow_Optics\\MicroCT_Data\\12Feb_UVD\\VTK'
## Path to Material csv files containing refractive index information.
MaterialPath = 'C:\\Users\\RDCRLJTP\\Documents\\Projects\\Snow_Optics\\Code\\NewRTM\\Materials\\'
## Path where VTK mesh file is stored --> Located in parentPath
subpath='contour\\test\\Pit1_1_3\\ssalb\\'

#subpath='13Mar_HBEF_1_1/16Mar_HBEF_1_1_Rec/VOI/Snow/'
#subpath='Spheres/'
## filename of vtk file --> located in parentPath/subpath/
#vtkFname='spheresD20.vtk'
vtkFname='CRREL_MESH.vtk'
## name of output file
outputFile='Properties.txt'

## Options for getting phase function with particle based approach.
## If phase type == 1, get phase function from particle based approach, otherwise do not.
## Grain path specifies the path to specific grains involved in sample.
##  NOTE, you should populate the grain folder with grains that do not intersect the sample boundaries (i.e., set border grains to <100 in ImageSeg)
##  This will ensure the grain shapes are realistic.  It's also not a bad idea to look over the grains and make sure they look good.
##  Grain samples is the number of grains to sample for computing the phase function:
##  for i in range(GrainSamples):
##      grain=Random Grain
##      for n in range(Nphotons):
##          Get scattering angle
## PhaseAbsorb = True, turns on absorption within the phase function, i.e., the photon weight is depleted via absorbtion within the particle
##      Doesn't matter much for visible wavelengths, definitely matters for IR.
PhaseType=1
GrainPath='GRAINS/'
# FullPath='FULL/'
GrainSamples=30
PhaseAbsorb=True
#############################################
TimeTest=False ## Determines timing of individual functions.

VoxelRes=19.88250 ## in micrometers

## This flag should always be true, and indicates you are reading a vtk file in, Otherwise
## you can try with other shapes (collection of spheres for example.)
read_file=True

## If True, then render the mesh for visual inspection.
## Note, the actual model won't be run if this is true.
checkMesh = False

## This parameter accounts for the fact that sometimes photons just "get stuck" inside particles
## through total interal reflection (TIR).  maxBounce essentially sets the total number of times a particle
## is allowed to bounce around within the SAME particle before it's killed.  Code can get stuck in what is
## effectivly an infinite loop without this variable, and the larger it is, the longer the code takes.
maxBounce=10

# Choose a mean WaveLength value to set set the refractive index.
# Since the extinction coefficient is only very weakly influenced by absorption,
# and since the refractive index of ice is pretty constant with wavelength this should be okay.
wavelen='800nm'

## number of photons to use in the sample.  Order 1-10K is probably sufficient.
## More photons + larger sample = longer compute time.
nPhotons=1500

## Number of Bins for Phase Function (binSize = pi/nBins).
## You want to strike a nice balance between accurately characterizing the shape of the phase function
## while ensuring that each bin angle has enough samples with in it to make it statistically robust. (180 is good)
nBins=180
## Plot the extinction coefficient Curve fit and the phase function.
do_plot=True


### You shouldn't have to edit (much) below this line!

###
###
###

###

## The VTKfilename is the name of the mesh file that is read into this
VTKfilename= os.path.join(parentPath,subpath,vtkFname)  #'%s/%s/%s'%(parentPath,subpath,vtkFname)
outputFilename= os.path.join(parentPath,subpath,outputFile)  #"/".join(VTKfilename.split('/')[:-1])+'/'+outputFile

GrainFiles=glob.glob(os.path.join(parentPath,subpath,GrainPath)+'Grain*')

bins=np.linspace(0,np.pi,nBins)
dtheta=np.abs(bins[1]-bins[0])
bins=np.cos(bins)
binCenters=(bins[:-1]+bins[1:])/2.
POWER=np.zeros_like(binCenters)

today=datetime.now().strftime('%c')

## Resolution of each volxel in (mm) --> e.g., 20um/1000um/mm
resolution=VoxelRes/1000.

if read_file == True:
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(VTKfilename)
    reader.Update()
    shell = reader.GetOutput()
    #help(shell.BOUNDING_BOX())
    xBounds=tuple(shell.GetBounds()[:2])
    yBounds=tuple(shell.GetBounds()[2:4])
    zBounds=tuple(shell.GetBounds()[4:])

    sampleVolumeOut=(xBounds[1]-xBounds[0])*(yBounds[1]-yBounds[0])*(zBounds[1]-zBounds[0])

    size= xBounds[1]-xBounds[0]

    pvob=pv.wrap(shell)
    snowMass=pvob.volume/(1000**3)*917.0

    cellAreas=pvob.compute_cell_sizes(length=False, area=True, volume=False, progress_bar=False)
    SfcArea=np.nansum(cellAreas.cell_arrays["Area"])/(1000**2)

    SampleVol=(xBounds[1]-xBounds[0])*(yBounds[1]-yBounds[0])*(zBounds[1]-zBounds[0])/(1000**3.)
    Density=snowMass/SampleVol
    SSA=SfcArea/snowMass
    GrainDiam=6./(917.0*SSA)*1000.

    prism=CRRELPolyData._CRRELPolyData(shell,xBounds,yBounds,zBounds,resolution,Density,description='REAL Snow Mesh')

    camera = vtk.vtkCamera()
    camera.SetPosition(xBounds[0],yBounds[0],zBounds[1]+3*resolution)
    camera.SetFocalPoint(xBounds[0],yBounds[0],zBounds[1])
    camera.SetViewUp(0,0,0)
    renderer=RenderFunctions.Render3DMesh(prism,opacity=0.75)

    print("Density, %.2f"%Density)
    print("SSA, %.2f"%SSA)
    print("Grain Size, %.2f"%GrainDiam)
    print("Sample Volume (mm^3) %.2f"%(SampleVol*(1000**3)))

    if checkMesh == True:

        RenderFunctions.ShowRender(renderer,camera=camera)
        sys.exit()

prism.AssignMaterial('ice',filePath=MaterialPath)

def func(x,ke):
    return 1.-np.exp(-ke*x)

def Phome(x,ka):
    dist=x[0]
    ke=x[1][0]
    return (1.-np.exp(-ke*dist))*ka/ke

ke=4.5 ## default starting point....
add=resolution*5

distances=[]
dd=resolution
while(dd < size):
    distances.append(dd)
    dd=1.3*dd

distances=np.array(distances)

if do_plot == True:
    renderer=RenderFunctions.Render3DMesh(prism)

print("Finding Extinction and Absorption coefficients for medium.")
#
n,abso,RI=prism.GetRefractiveIndex(wavelen)
extinction=np.zeros_like(distances)

time1=datetime.now()
for ddx, d in enumerate(distances):
    print(d)
    Pexts=[]
    for ii in range(nPhotons):
        x1=np.random.uniform(prism.xBounds[0]+add,prism.xBounds[1]-add)
        y1=np.random.uniform(prism.yBounds[0]+add,prism.yBounds[1]-add)
        z1=np.random.uniform(prism.zBounds[0]+add,prism.zBounds[1]-add)

        x2=np.random.uniform(prism.xBounds[0],prism.xBounds[1])
        y2=np.random.uniform(prism.yBounds[0],prism.yBounds[1])
        z2=np.random.uniform(prism.zBounds[0],prism.zBounds[1])

        p11=[x1,y1,z1]
        p22=[x2,y2,z2]
        try:
            # Pext=RTcode.TracktoExt(prism,p11,p22,wavelen,float(d))
            Pext=RTcode.TracktoExt(prism,p11,p22)    # Julie Parno changed 8/2/2021
        except:
            print("Bad Photon! Something weird happened, so I'm skipping this!")
            continue
        Pexts.append(Pext)

    extinction[ddx]=np.nanmean(Pexts)

popt, pcov = curve_fit(func, distances,extinction)

time2=datetime.now()

if TimeTest == True:
    print("Time to compute Extinction Coefficient For %i Photons %.2f seconds"%(nPhotons,(time2-time1).total_seconds()))

if do_plot == True:
    plt.figure()
    plt.scatter(distances,extinction)
    plt.plot(distances, func(distances, *popt), 'r-',label='fit: Ke=%.1f' % popt[0])
    plt.legend()
    plt.grid()

print("Ke = %.2f | WaveLength = %s"%(popt[0],wavelen))
ke=popt[0]

# Single scattering albedo
ssalb = (ke-abso)/ke
print("Single Scattering Albedo = %.2f | WaveLength = %s"%(ssalb,wavelen))

## NOW DO Absorption AND PHASE FUNCTION
TotalLens=[]
Fice=[]
TotalPorous=[]
for ii in range(nPhotons):
    x1=np.random.uniform(prism.xBounds[0]+add,prism.xBounds[1]-add)
    y1=np.random.uniform(prism.yBounds[0]+add,prism.yBounds[1]-add)
    z1=np.random.uniform(prism.zBounds[0]+add,prism.zBounds[1]-add)

    x2=np.random.uniform(prism.xBounds[0],prism.xBounds[1])
    y2=np.random.uniform(prism.yBounds[0],prism.yBounds[1])
    z2=np.random.uniform(prism.zBounds[0],prism.zBounds[1])

    p11=[x1,y1,z1]
    p22=[x2,y2,z2]

    if TimeTest == True:
        time1=datetime.now()

    try:
        Tice,Total,intersections,COSPHIS,weights=RTcode.TracktoAbs(prism,p11,p22,wavelen,straight = False,units='nm')
    except:
        print("Bad Photon! Something weird happened, so I'm skipping this!")
        continue

    if TimeTest == True:
        time2=datetime.now()
        print("Time to run a single photon through the Kaempfer Model %.4f"%((time2-time1).total_seconds()))
        print("Estimated time for %i photons = %.2f"%(nPhotons,(time2-time1).total_seconds()*nPhotons))
        sys.exit()

    if len(COSPHIS) == 0: ## THIS TOTALLY MISSED ALL ICE! ##
        continue

    for cdx,c in enumerate(COSPHIS):
        index=np.argmin(np.abs(binCenters-c))
        POWER[index]+=weights[cdx]

    if Total == 0:
        continue
    TotalLens.append(Total)
    Fice.append(Tice/Total)

thetas=np.arccos(binCenters)
POWER=4.*np.pi*POWER[:]/(np.sum(POWER[:])*np.sin(thetas[:])*dtheta)

if PhaseType == 1: ## GET phase function by running paricle based approach through a bunch of grains!
    ## Note, this overwrites POWER variable above.
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

        sampleVolume=(xBounds[1]-xBounds[0])*(yBounds[1]-yBounds[0])*(zBounds[1]-zBounds[0])

        size= xBounds[1]-xBounds[0]
        CRRELPD=CRRELPolyData._CRRELPolyData(shell,xBounds,yBounds,zBounds,resolution,917,description='REAL Snow Mesh')

        CRRELPD.AssignMaterial('ice',filePath=MaterialPath)

        for ndx in range(nPhotons):
            x1=np.random.uniform(CRRELPD.xBounds[0],CRRELPD.xBounds[1])
            y1=np.random.uniform(CRRELPD.yBounds[0],CRRELPD.yBounds[1])
            z1=np.random.uniform(CRRELPD.zBounds[0],CRRELPD.zBounds[1])

            x2=np.random.uniform(CRRELPD.xBounds[0],CRRELPD.xBounds[1])
            y2=np.random.uniform(CRRELPD.yBounds[0],CRRELPD.yBounds[1])
            z2=np.random.uniform(CRRELPD.zBounds[0],CRRELPD.zBounds[1])

            p11=[x1,y1,z1]
            p22=[x2,y2,z2]

            weights,COSPHIS,intersections,dummy=RTcode.ParticlePhaseFunction(CRRELPD,p11,p22,wavelen,units='nm',absorb=PhaseAbsorb)

            if dummy == True:
                continue

            if len(COSPHIS) == 0: ## THIS TOTALLY MISSED ALL ICE! ##
                continue

            for cdx,c in enumerate(COSPHIS):
                index=np.argmin(np.abs(binCenters-c))
                POWER[index]+=weights[cdx]

    thetas=np.arccos(binCenters)
    POWER=4.*np.pi*POWER[:]/(GrainSamples*nPhotons*np.sin(thetas[:])*dtheta)



    PhaseText='Phase function computed using "particle-oriented" approach with %s grain samples and %s photons per grain \nAbsorption included in phase function = %s \n'%(GrainSamples,nPhotons,PhaseAbsorb)

else:
    PhaseText='Phase Function computed using "boundary-oriented" approach \n'


plt.figure(figsize=(10,10))
ax=plt.subplot(111)
ax.plot(np.degrees(thetas),POWER,label='%s'%wavelen)
ax.set_yscale('log')
plt.grid()
plt.legend()

FiceMean=np.nanmean(Fice)

print("Fice %.2f"%FiceMean)
print("------------------")

with open(outputFilename, 'w') as file:
   file.write("Optical Properties File Created From %s \n"%VTKfilename)
   file.write("File Created on %s \n"%today)
   file.write("Central Wavelength and refractive index: %s, %.2f \n"%(wavelen,n))
   file.write("Number of Photons used in Monte Carlo Sampling: %i \n"%nPhotons)
   file.write(PhaseText)
   file.write("Sample Volume = %.2f (mm^{3}) \n"%sampleVolumeOut)
   file.write("Estimated Sample Density = %.2f (kg m^{-3}) \n"%Density)
   file.write("Estimated Sample SSA = %.2f (m^{2} kg^{-1}) \n"%SSA)
   file.write("Estimated Sample Grain Diameter = %.2f (mm) \n"%GrainDiam)
   file.write("Max Number of Bounces allowed in phase function and Ice Path Sampling: %i \n"%maxBounce)
   file.write("Extinction Coefficient = %.4f (mm^{-1}) \n"%ke)
   file.write("Single Scattering Albedo = %.4f (mm^{-1}) \n"%ssalb)
   file.write("Mean fractional distance traveled in ice medium (Fice) = %.3f \n"%FiceMean)
   file.write("Number of bins in phase function = %i \n"%nBins)
   file.write("Angular Bin Size (radians) = %.5f"%dtheta)
   file.write("------------------------------------------------------  \n")
   file.write("Theta (radians), Phase Function\n")
   for tdx in range(len(thetas)):
       file.write("%.4f, %.4f \n"%(thetas[tdx],POWER[tdx]))

plt.show()
