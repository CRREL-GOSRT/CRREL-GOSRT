##Polar Phase.py
##Checks Phase function properties from Spheres
from file_pathnames import *
import sys
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
import os
import glob as glob


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def sph2cart(azi,el,r):
    x = r * np.sin(el) * np.cos(azi)
    y = r * np.sin(el) * np.sin(azi)
    z = r * np.cos(el)
    return x, y, z


print(1000./(2.*np.pi)*0.550/1000.)
meanRadius = 1000./(2.*np.pi)*0.550/1000.
numSpheres=1
numPhotons=5000 ## per sphere

voxelize=False

center=(0,0,0)
nTheta=100
nPhi=100

ThetaBounds=[-np.pi/2., np.pi/2.]
AziBounds=[-np.pi,np.pi]

nBins=180

VoxelRes=19.88250/1000.
exportPath=''
radii=[]
wavelens=[900]
wavelenUnits='nm'

allowed=['sphere','column','column','grain']


fig=plt.figure(figsize=(11,9))

# counters=[1,3,5,7,2,4,6,8]

counters=[1,2,3,4,1,2,3,4]
countIdx=0

for wdx,wavelen in enumerate(wavelens):
    if wdx == 0:
        kL=1000.  # size parameter
        k=2.*np.pi/(wavelen * 1E-6)

        L=kL/k  # radius of sphere
    LoA=[6,8,0.1,6] # aspect ratio

    print(L)
    GrainNum='136'

#    wavelenStr = '%snm'%wavelen

    labels=['Sphere (kL=%.1f mm)'%(kL),
            'Column (AR = %.1f, kL=%.1f)'%(LoA[1],kL),
            'Column (AR = %.1f, kL=%.1f)'%(LoA[2],kL),
            'Rendered Snow Grain from microCT']


    GrainsPath=os.path.join(VTK_DATA_OUTPATH,'GRAINS')

    outpath=OPT_PROP_OUTPATH

    thetas=np.linspace(0,np.pi,nBins)
    dtheta=np.abs(thetas[1]-thetas[0])
    thetaCenters=(thetas[:-1]+thetas[1:])/2.
    bins=np.cos(thetas)
    binCenters=(bins[:-1]+bins[1:])/2.

    GrainFiles=glob.glob(GrainsPath+'*%s*.vtk'%GrainNum)

    for sdx, shape in enumerate(allowed):
        print(shape)
        
        PhaseHPol=np.zeros_like(binCenters)
        PhaseVPol=np.zeros_like(binCenters)
        UnPol=np.zeros_like(binCenters)

        if shape.lower() == 'grain':

            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(GrainFiles[0])
            reader.Update()
            shell = reader.GetOutput()
            voxelRes=VoxelRes/1000.
            #help(shell.BOUNDING_BOX())
            xBounds=tuple(shell.GetBounds()[:2])
            yBounds=tuple(shell.GetBounds()[2:4])
            zBounds=tuple(shell.GetBounds()[4:])

            sampleVolume=(xBounds[1]-xBounds[0])*(yBounds[1]-yBounds[0])*(zBounds[1]-zBounds[0])

            size= xBounds[1]-xBounds[0]
            CRRELPD=CRRELPolyData._CRRELPolyData(shell,xBounds,yBounds,zBounds,voxelRes,917,description='REAL Snow Mesh')
            # print('grain bounds = ', CRRELPD.xBounds,CRRELPD.yBounds,CRRELPD.zBounds)

            pvob=pv.wrap(shell)
            snowMass=pvob.volume/(1000**3)*917.0

            cellAreas=pvob.compute_cell_sizes(length=False, area=True, volume=False, progress_bar=False)
            SfcArea=np.nansum(cellAreas.cell_arrays["Area"])/(1000**2)

            SampleVol=(xBounds[1]-xBounds[0])*(yBounds[1]-yBounds[0])*(zBounds[1]-zBounds[0])/(1000**3.)
            Density=snowMass/SampleVol
            SSA=SfcArea/snowMass
            GrainDiam=6./(917.0*SSA)*1000.

            print(GrainDiam*2.*np.pi/(wavelen * 1E-6))

            CRRELPD.WritePolyDataToVtk(outpath+shape+'.vtk')

        if shape == 'sphere':
            points,triangles,colors =DrawShapes.MakeSphere(L, center, nTheta, nPhi)

            CRRELPD= DrawShapes.CreateCRRELPolyDataFromPoints(points,triangles,colors)
            # print('sphere bounds = ', CRRELPD.xBounds,CRRELPD.yBounds,CRRELPD.zBounds)
            CRRELPD.WritePolyDataToVtk(outpath+shape+'.vtk')


        if shape == 'column':
            if LoA[sdx]> 1:
                a=2*L/LoA[sdx]
            else:
                L=L/2.*LoA[sdx]
            points,triangles,colors = DrawShapes.DrawHexColumn(a,center,2*L,points=None,triangles=None,theta=0,phi=0)

            CRRELPD= DrawShapes.CreateCRRELPolyDataFromPoints(points,triangles,colors)
            # print('column bounds = ', CRRELPD.xBounds,CRRELPD.yBounds,CRRELPD.zBounds)
            CRRELPD.WritePolyDataToVtk(outpath+shape+"_"+str(LoA[sdx])+'.vtk')

        CRRELPD.AssignMaterial('ice',filePath=MATERIAL_PATH)
        
        nIce,kIce,Nc=CRRELPD.GetRefractiveIndex(wavelen,units=wavelenUnits)
        
        normalsMesh=CRRELPD.GetNormalsMesh()
        obbTree=CRRELPD.GetObbTree()

        if voxelize == True:
            CRRELPD.Voxelize()

        for ndx in range(numPhotons):
            x1=np.random.uniform(CRRELPD.xBounds[0],CRRELPD.xBounds[1])
            y1=np.random.uniform(CRRELPD.yBounds[0],CRRELPD.yBounds[1])
            z1=np.random.uniform(CRRELPD.zBounds[0],CRRELPD.zBounds[1])

            x2=np.random.uniform(CRRELPD.xBounds[0],CRRELPD.xBounds[1])
            y2=np.random.uniform(CRRELPD.yBounds[0],CRRELPD.yBounds[1])
            z2=np.random.uniform(CRRELPD.zBounds[0],CRRELPD.zBounds[1])
            
            # hxy=np.hypot(CRRELPD.xBounds[1]-CRRELPD.xBounds[0],CRRELPD.yBounds[1]-CRRELPD.yBounds[0])
            # rmax=np.hypot(hxy,CRRELPD.zBounds[1]-CRRELPD.zBounds[0])

            # rs=np.random.uniform(0,rmax)
            # Thetas=np.random.uniform(*ThetaBounds)
            # Azis=np.random.uniform(*AziBounds)

            # x2,y2,z2= sph2cart(Azis,Thetas,rs)
            # x2,y2,z2 = x2+x1,y2+y1,z2+z1
            # x2=CRRELPD.xBounds[1]

            axis=np.random.randint(0,6)

            if axis == 0:
                p11=[CRRELPD.xBounds[0],y1,z1]
                p22=[CRRELPD.xBounds[1],y2,z2]

            elif axis == 1:
                p11=[x1,CRRELPD.yBounds[0],z1]
                p22=[x2,CRRELPD.yBounds[1],z2]

            elif axis == 2:
                p11=[x1,y1,CRRELPD.zBounds[0]]
                p22=[x2,y2,CRRELPD.zBounds[1]]

            elif axis == 3:
                p11=[CRRELPD.xBounds[1],y1,z1]
                p22=[CRRELPD.xBounds[0],y2,z2]

            elif axis == 4:
                p11=[x1,CRRELPD.yBounds[1],z1]
                p22=[x2,CRRELPD.yBounds[0],z2]

            elif axis == 5:
                p11=[x1,y1,CRRELPD.zBounds[1]]
                p22=[x2,y2,CRRELPD.zBounds[0]]

            p11=[x1,y1,z1]
            p22=[x2,y2,z2]
            
            # p11 = srcPts[ndx,:]
            # p22 = tgtPts[ndx,:]

            # inter=np.random.randint(1,3)

            # polar=complex(0,inter)

            weights,COSPHIS,intersections,dummy=RTcode.ParticlePhaseFunction(CRRELPD,p11,p22,normalsMesh,obbTree,nIce,kIce,absorb=True)


            if dummy == True:
                continue

            #for i in range(len(intersections[:-1])):
            #    RenderFunctions.addLine(renderer, intersections[i], intersections[i+1])

            if len(COSPHIS) == 0: ## THIS TOTALLY MISSED ALL ICE! ##
                continue

            for cdx,c in enumerate(COSPHIS):
                # index=np.argmin(np.abs(binCenters-c))
                index = np.where((c>bins[1:]) & (c<bins[:-1]))[0]
                UnPol[index]+=weights[cdx]
                # if inter == 1:
                #     PhaseHPol[index]+=weights[cdx]
                # else:
                #     PhaseVPol[index]+=weights[cdx]

            # print(np.shape(intersections))
            # for i in range(len(intersections[:-1])):
            #     RenderFunctions.addLine(renderer, intersections[i], intersections[i+1])
                #

        # thetas=np.arccos(binCenters)
        N = np.sum(UnPol)
        dOmega = np.sin(thetaCenters)*dtheta*2*np.pi
        pf=4.*np.pi*UnPol[:]/(N*dOmega)
        # pf=4.*np.pi*UnPol[:]/(numSpheres*numPhotons*np.sin(thetas[:])*dtheta)
        # UnPol=4.*np.pi*UnPol[:]/(numSpheres*numPhotons*np.sin(thetas[:])*dtheta)

    #    VPol=4.*np.pi*PhaseVPol[:]/(numSpheres*numPhotons*np.sin(thetas[:])*dtheta)
    #    HPol=4.*np.pi*PhaseHPol[:]/(numSpheres*numPhotons*np.sin(thetas[:])*dtheta)
    
        asymmparam = 0.5*dtheta*np.sum(pf*np.sin(thetaCenters)*np.cos(thetaCenters))

        ax=plt.subplot(2,2,counters[countIdx])
        ax.plot(np.degrees(thetaCenters),pf,label="$\lambda$ = %.2f $\mu$m"%(wavelen/1000.))
        ax.set_yscale('log')

        if wdx == 0:
            ax.grid()

        ax.set_xlim(0,180)
        ax.set_ylim(0.01,10000)

        if wdx == 0:
            ax.text(5,1300,labels[sdx],ha='left')
            ax.text(5,500,"g = %.2f"%asymmparam,ha='left')

        if sdx == 0 and wdx == 1:
            #ax.set_title("$\lambda$ = %.2f $\mu$m"%(wavelen/1000.),loc='left')
            ax.legend()


        if counters[countIdx] not in [3,4]:
            ax.set_xticklabels([])

        if counters[countIdx] in [3,4]:
            ax.set_xlabel(r"$\Phi$")

        if counters[countIdx] in [1,3]:
            ax.set_ylabel("Phase Function")

        countIdx+=1


plt.show()
