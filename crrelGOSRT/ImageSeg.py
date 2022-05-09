""" This file contains the main functions for generating a mesh from binarized microCT
    image stacks.

    Notes
    -----
      First Created by Ted Letcher January 2021
      Updated by Julie Parno September 2021


    Functions Included
    -----------------------------
    ImagestoArray  - Converts binarized MicroCT image stacks to numpy array
    GrainSeg  - Performs snow grain segmentation using watershed segmentation currently
    MeshGen  - Generates a mesh of the snow grains, can do both individual grains and full sample mesh


    Requirements
    -------------
      VTK
      PyVista
      PyMeshFix
      SciPy
      scikit-image
      Binarized microCT image stack


    Last modified:
    ---------------
      September 2021
"""

import vtk
import pandas as pd
import numpy as np
import pyvista as pv
import pymeshfix as mf
from matplotlib import pyplot as plt
import glob
from matplotlib.image import imread
from crrelGOSRT import CRRELPolyData
from crrelGOSRT import RenderFunctions
from scipy import signal, ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.measure import marching_cubes_lewiner
from skimage.filters import gaussian
from datetime import datetime, timedelta
import os
import shutil
from pathlib import Path



def ImagesToArray(path,outpath,XYstart,XYend,depthTop,Ztop,voxelRes,thresh=0.9,savePropsToCsv=True,saveMeshParams=True,imgsfx='png'):

    """
    This function reads in a microCT binarized image stack and generates arrays necessary for grain segmentation and mesh generation

    Note that depths are measured from ground (d=0)

    INPUTS:
        path: main path for microCT data
        outpath: Output Path for .vtk files
        XYstart: The starting point in XY for the mesh subset within a sample image (in plane view) in millimeters, assuming the the left most pixel is 0.
        XYend: The ending point in XY for the mesh subset within a sample image (in plane view) in millimeters, assuming the the left most pixel is 0.
        depthTop: Top snow depth of scanned sample (in mm) --> Usually corresponds to file name in "subpath" variable
           * Note that depthTop is critically important in getting the depth-oriented sample correct!
        Ztop: Top depth selected for mesh sample subset (must be within the sample depth)
        voxelRes: voxel resolution in mm from microCT log
        thresh: threshold for snow/air boundary in binarized microCT images, recommend 0.9
        savePropsToCsv: option to save sample properties to CSV file
        saveMeshParams: option to save mesh parameter information

    RETURNS:
        SNOW: binary 3D array where 1 is snow and 0 is air
        grid: list of numpy arrays that represent coordinate matrices for X,Y,Z axes of sample array

    """
    # begin_time = datetime.now()

    # Flag to indicate, that depths come from a text file, otherwise, they are numbered according to PNG filenames.
    FromTextFile=False

    ## File definitions to find text file describing sample depths
    txtFilePfx='batman1'  ## need to change if going to use this feature

    # Text file describing sample depths
    textpath= os.path.join(path,txtFilePfx)

    # Check to see if output path exists, and if not, create it
    if not os.path.exists(outpath):
        print('Outpath does not exist, creating new directory at ' + str(outpath))
        os.makedirs(outpath)

    # Bottom most sample depth (example: subsamples of 5.5 mm height, Ztop-5.5) - Currently set to automatically generate a cube
    Zbot=Ztop-(XYend-XYstart)
    # Sample depth range
    Zrange=[Zbot,Ztop]

    # Saves mesh parameters for reviewing/testing purposes
    if saveMeshParams==True:
        with open(os.path.join(outpath,'MeshParameters.txt'), 'w') as file:
           file.write("File Created on %s \n"%datetime.now().strftime('%c'))
           file.write("XYStart, XYEnd =  %.1f, %.1f \n"%(XYstart,XYend))
           file.write("Voxel Resolution= %.6f (mm) \n"%voxelRes)
           file.write("Binary image segmentation threshold = %.1f \n"%thresh)


    # # If there is a text file describing sample depths, read that in
    if FromTextFile == True:
        textfile=glob.glob(textpath)[0]
        data=pd.read_csv(textfile,header=41)

    # Read in full stack of plane view images
    pngfiles=sorted(glob.glob(os.path.join(path,'*.%s'%imgsfx)))

    # Calculate and define depths
    Z=[]
    finalpngs=[]
    for png in pngfiles:
        if FromTextFile == True:
            bmpmatch=png.split('/')[-1].split('.')[0]+'.bmp'
            datamatch=data[data['File name'] == bmpmatch]
            depth=float(datamatch['Z position'][:])
        else:
            try:
                depthNum =float(png.split('.%s'%imgsfx)[0].split('_')[-1])
                depth = depthNum*voxelRes
            except:
                print("WARNING --- %s does not match the correct image format!"%os.path.basename(png))
                print("Skipping this file!")

        # pull just PNGs associated with the subsample
        if Zrange[0] <= depthTop-depth <= Zrange[1]:
            Z.append(depthTop-depth)
            finalpngs.append(png)

    Z=Z[::-1]
    finalpngs=finalpngs[::-1]
    # print(finalpngs[0],finalpngs[-1])

    # Narrow image field for analysis by masking image with previously defined XYstart and XYend points,
    # to get rid of unwanted parts of image that are not snow
    for idx, i in enumerate(finalpngs):
        img = imread(finalpngs[idx])

        # if idx == 0:
        #     plt.figure()
        #     ax=plt.subplot(1,1,1)
        #     ax.imshow(img)

        startX,startY=[int(XYstart/voxelRes)]*2
        endX,endY=[int(XYend/voxelRes)]*2

        img_mask=img[startX:endX+1,startY:endY+1,0].squeeze()

        img_mask=np.ma.masked_invalid(np.ma.masked_less(img_mask,thresh).filled(0.0)/img_mask).filled(0.0)
        if idx == 0:
            # plt.figure()
            # ax=plt.subplot(1,1,1)
            # ax.imshow(img_mask[::-1],cmap='binary_r')
            X=XYstart+np.arange(img_mask.shape[0])*voxelRes
            Y=XYstart+np.arange(img_mask.shape[1])*voxelRes
            SNOW=np.zeros([img_mask.shape[0],img_mask.shape[1],len(finalpngs)])

        SNOW[:,:,idx]=img_mask[:]

    np.save(os.path.join(outpath,'microCT_arr.npy'),SNOW)

    # create grid from new selected image field, where Z is the bottom image of the stack (lowest depth, highest voxel #)
    X,Y,Z=np.meshgrid(X,Y,Z)
    print("Array Size:",np.shape(SNOW))
    grid = [X,Y,Z]
    origin_mm = np.array([XYstart, XYstart, np.min(Z)])

    return SNOW,grid



def GrainSeg(SNOW,voxelRes,minGrainSize,outpath,thresh=0.9,saveMeshParams=True):
    """
    This function reads in a binarized snow/air array and runs a watershed segmentation
    to separate out individual snow grains

    INPUTS:
        SNOW: binary 3D array where 1 is snow and 0 is air
        voxelRes: voxel resolution in mm from microCT log
        minGrainSize: This sets the minimum grainsize (in mm) for the peak-local-max function for watershed segmentation, generally 0.3 - 0.8mm, depending on sample
        outpath: Output Path for .vtk files
        thresh: threshold for snow/air boundary in binarized microCT images, recommend 0.9
        saveMeshParams: option to save mesh parameter information

    RETURNS:
        grain_labels: numpy array where 0 indicates air and all individual snow grains are marked with a unique number
        grains: array that lists grain numbers in increasing order
        properties: list of region properties for each grain (watershed), output by watershed segmentation algorithm

    """
    # create arrays where non-snow pixels are 0 (SNOW_IN) or snow pixels are set to 0 (SNOW_OUT)
    SNOW_IN=np.ma.masked_less(SNOW,thresh)  # anything less than threshold (0.9) is masked
    SNOW_OUT=np.ma.masked_array(np.ones_like(SNOW),mask=~SNOW_IN.mask).filled(0.0) # creates an array of ones and then masks and sets snow pixels to 0
    SNOW_IN=SNOW_IN.filled(0.0) # sets non-snow pixels to 0

    # Calculate density of snow sample before meshing
    grain_pixcount = np.count_nonzero(SNOW_IN)*voxelRes**3
    snow_mass = grain_pixcount/(1000**3)*917
    totalvol = ((np.shape(SNOW)[0]*np.shape(SNOW)[1]*np.shape(SNOW)[2])*voxelRes**3)/1000**3
    snow_density = snow_mass/totalvol

    # Calculating euclidean distances. *note distance_in is the only one used
    distance_in = ndi.distance_transform_edt(SNOW_IN)
    # distance_out = ndi.distance_transform_edt(SNOW_OUT)
    # distance = distance_out - distance_in

    # Find peaks (local maxima) in an image, output is a coordinate list, must be separated by at least the min_dist calculated below
    min_dist = int(minGrainSize/voxelRes)
    local_maxi = peak_local_max(distance_in, min_distance=min_dist, exclude_border=False, indices=False, labels=SNOW_IN)

    # mark local maxima and fill "watershed basins" in image
    markers = ndi.label(local_maxi)[0]
    grain_labels = watershed(-distance_in, markers, mask=SNOW_IN) # give each grain a label number
    properties = regionprops(grain_labels) # measures and records properties of each "watershed"

    # Plot a grain separation example for the last image in the stack
    plt.figure(figsize=(12,9))
    ax=plt.subplot(1,1,1)
    ZZ=plt.pcolormesh(grain_labels[:,:,-1])
    plt.colorbar(ZZ)
    plt.title('Min Grain Size set at ' + str(minGrainSize) + ' mm')
    plt.savefig(os.path.join(outpath,('grainsep_'+str(minGrainSize)+'.png')))

    # Plot approximate grain size distribution
    radius = np.ones((len(properties),1))
    plt.figure(figsize=(12,9))
    ax=plt.subplot(1,1,1)
    for idx, i in enumerate(properties):
        radius[idx] = properties[idx].equivalent_diameter/2.0*voxelRes
    ax.hist(radius, bins=20)
    plt.xlabel('Grain radius (mm)')
    plt.ylabel('Number of grains')
    plt.title('Total number of grains =' + str(len(properties)))
    plt.savefig(os.path.join(outpath,('grainsize_distr_'+str(minGrainSize)+'.png')))


    print('Total number of grains = ' + str(len(properties)))
    print('Mean grain radius = ' + str(np.mean(radius)))
    print('Median grain radius = ' + str(np.median(radius)))
    print('Snow density before meshing = %.2f kg/m^3' % snow_density)
    print('Sample volume before meshing = %.2f mm^3' %(totalvol*1000**3))

    if saveMeshParams==True:
        with open(os.path.join(outpath,'MeshParameters.txt'), 'a') as file:
           file.write('\n')
           file.write("Grain segmentation method = Watershed segmentation \n")
           file.write("Minimum Grain Size Set To: %.2f (mm) \n"%minGrainSize)
           file.write('Total number of grains = %i \n' %len(properties))
           file.write('Mean grain radius = %.2f \n' %(np.mean(radius)))
           file.write('Median grain radius = %.2f \n' %(np.median(radius)))
           file.write('Snow density before meshing = %.2f kg/m^3 \n' % snow_density)
           file.write('Sample volume before meshing = %.2f mm^3 \n' % (totalvol*1000**3))

    grains=np.arange(np.min(grain_labels),np.max(grain_labels)) # rearrange grains in min-max order

    return grains,grain_labels,properties



def MeshGen(grains,grain_labels,properties,voxelRes,grid,allowedBorder,
            minpoints,decimate_val,outpath,fullMeshName,saveMeshParams=True,
            savePropsToCsv=True,check=True,create_individual=True):

    """
    This function generates a sample mesh and individual grain meshes, if selected, via
    a contour/isoline method and Marching Cubes (Lewiner et al., 2003). The function writes
    out mesh VTK file(s) to specified directory.

    INPUTS:
        grains: array that lists grain numbers in increasing order
        grain_labels: numpy array where 0 indicates air and all individual snow grains are marked with a unique number
        properties: list of region properties for each grain (watershed), output by watershed segmentation algorithm
        voxelRes: voxel resolution in mm from microCT log
        grid: list of numpy arrays that represent coordinate matrices for X,Y,Z axes of sample array
        allowedBorder: Number of points allowed to be on a mesh border. This can eliminate "flat" grain boundaries that are up against the mesh boundary.
            * Larger number = less restrictive (if you want everything set to 1E6 or more)
        minpoints: Minimum number of points allowed for each grain
        decimate_val: The decimal percentage of mesh triangles to eliminate in decimation (recommended: 0.9)
        outpath: Output Path for .vtk files
        fullMeshName: filename for full sample mesh (*.vtk)
        saveMeshParams: option to save mesh parameter information
        savePropsToCsv: option to save sample properties to CSV file
        check: If True, will stop after first grain to allow for visual inspection prior to running the whole code
            If True, it will:
                - return a vtk window of a single grain for visual inspection, it will also
                - plot up a 2D image slice showing the grain separation by coloring each grain differently
                - Ideally, you want the boundaries to mostly fall along grain-necks and separate areas that look distinct
                - Note that in this figure, the air space will plot as dark purple (0).
        create_individual: option to save individual grain VTK files in addition to the full mesh, generally recommended if space allows

    """

    # Unpack grid
    X,Y,Z = grid[0],grid[1],grid[2]

    # Define origin of grid
    origin_mm = np.array([np.min(X),np.min(Y),np.min(Z)])

    # If option selected, start creating data dictionary to write out properties to a CSV
    if savePropsToCsv == True:
        dataDict={}
        dataDict['Name']=[]
        dataDict['Center x (mm)'] = []
        dataDict['Center y (mm)'] = []
        dataDict['Center z (mm)'] = []
        dataDict['Radius (mm)'] = []
        dataDict['Volume (mm^3)'] = []

    # Create directory for grains with no border for phase function calculation
    # and create empty list to populate with borderless grains
    GrainPath = os.path.join(outpath,'GRAINS','')
    # Check to see if grain path exists, and if not, create it
    if not os.path.exists(GrainPath):
        os.makedirs(GrainPath)
    borderless = []

    print("Creating Volume, this will take a while")
    fulltime1=datetime.now()
    begin_time = datetime.now()

    if saveMeshParams==True:
        with open(os.path.join(outpath,'MeshParameters.txt'), 'a') as file:
           file.write('\n')
           file.write("Points allowed on border = %i  \n"%allowedBorder)
           file.write("Minimum number of points allowed per grain = %i \n"%minpoints)
           file.write("Decimate value for contour method = %.1f \n"%decimate_val)

    ### Here's where we start the main mesh-building loop! ###
    newPoly=True
    for gdx, grain in enumerate(grains[1:]):
        time1=datetime.now()
        if gdx == 0:
            print("On Grain %i of %i"%(grain,len(grains)))
        else:
            print("On Grain %i of %i.  Estimated time remaining = %.1f s"%(grain,len(grains),total_seconds*(len(grains)-grain)))


        grainExample=np.ma.masked_not_equal(grain_labels,grain)

        # selects out the X,Y,Z values associated with the specific grain that the loop is currently working on
        x=np.ma.masked_array(X,mask=grainExample.mask).compressed()
        y=np.ma.masked_array(Y,mask=grainExample.mask).compressed()
        z=np.ma.masked_array(Z,mask=grainExample.mask).compressed()


        # Get the points as a 2D NumPy array (N by 3)
        points = np.c_[x.reshape(-1), y.reshape(-1), z.reshape(-1)]

        # Calculates number of points of grain that are on the sample border
        xborder=len(np.ma.masked_not_equal(x,np.min(X)).compressed())+len(np.ma.masked_not_equal(x,np.max(X)).compressed())
        yborder=len(np.ma.masked_not_equal(y,np.min(Y)).compressed())+len(np.ma.masked_not_equal(y,np.max(Y)).compressed())
        zborder=len(np.ma.masked_not_equal(z,np.min(Z)).compressed())+len(np.ma.masked_not_equal(z,np.max(Z)).compressed())

        # if the border points exceed the allowedBorder set at top of script, skip it
        total_border=xborder+yborder+zborder
        if total_border > allowedBorder:
            time2=datetime.now()
            total_seconds=(time2-time1).total_seconds()
            print("Border grain, skipping! %i"%total_border)
            continue

        if total_border == 0:
            borderless.append(grain)

        if len(x) < minpoints:
            print("Too few points, assuming grain is too small, skipping!")
            continue

        if savePropsToCsv == True:
            #print("Centroid: {0}".format(properties[gdx].centroid))
            #print("Radius: {0}".format(properties[gdx].equivalent_diameter/2.0*voxelRes))
            #print("Area: {0}".format(properties[gdx].area*voxelRes**3))

            dataDict['Name'].append('Grain_%i'%grain)
            dataDict['Center x (mm)'].append(np.min(X) + properties[gdx].centroid[0]*voxelRes)
            dataDict['Center y (mm)'].append(np.min(Y) + properties[gdx].centroid[1]*voxelRes)
            dataDict['Center z (mm)'].append(np.min(Z) + properties[gdx].centroid[2]*voxelRes)
            dataDict['Volume (mm^3)'].append(properties[gdx].area*voxelRes**3)
            dataDict['Radius (mm)'].append(properties[gdx].equivalent_diameter/2.0*voxelRes)

            if check == True:
                x1=np.min(X) + properties[gdx].centroid[0]*voxelRes
                y1=np.min(Y) + properties[gdx].centroid[1]*voxelRes
                z1=np.min(Z) + properties[gdx].centroid[2]*voxelRes
                print(x1,y1,z1)


            ## Mesh generation via contour method and Marching Cubes
            grain_select = (grain_labels==grain)  # boolean array where grain is True, all else is False

            # Extract the subset of the large image that corresponds to this grain, this should help with code speed
            nz_inds = np.nonzero(grain_select)
            buffer = 3
            imin, imax = np.max([0,np.min(nz_inds[0])-buffer]), np.min([grain_select.shape[0],np.max(nz_inds[0])+buffer])
            jmin, jmax = np.max([0,np.min(nz_inds[1])-buffer]), np.min([grain_select.shape[1],np.max(nz_inds[1])+buffer])
            kmin, kmax = np.max([0,np.min(nz_inds[2])-buffer]), np.min([grain_select.shape[2],np.max(nz_inds[2])+buffer])

            grain_select = grain_select[imin:imax,jmin:jmax,kmin:kmax]

            # Smooth the grain_select indicator to get a smooth level set
            num_refine = 3
            smooth_length = 2

            # refine_time = datetime.now()
            grain_select = np.repeat(np.repeat(np.repeat(grain_select,num_refine,axis=0),num_refine,axis=1),num_refine,axis=2).astype(np.float)
            # refine_end_time = (datetime.now() - refine_time)
            # print('Refine time: ' + str(refine_end_time))
            # smooth_time = datetime.now()
            grain_select = gaussian(grain_select,smooth_length) # second input is number of pixels it is smoothing over
            # smooth_end_time = (datetime.now() - smooth_time)
            # print('Smooth time: ' + str(smooth_end_time))

            # Compute the isosurface at the boundary
            iso_val =  0.4*(np.max(grain_select)+np.min(grain_select)) # 0 outside, 1 inside.  0.5 Should be close to original edge, but this should be played with

            # Set the values on the boundary to be slightly less than the iso value
            # This ensures that the isosurface will contain the boundary near the grain
            grain_select[[0,-1],:,:] = 0.999*iso_val
            grain_select[:,[0,-1],:] = 0.999*iso_val
            grain_select[:,:,[0,-1]] = 0.999*iso_val

            # # plot a slice
            # zind = 13
            # plt.figure()
            # plt.imshow(grain_select[:,:,zind*num_refine],cmap='gray')
            # plt.contour(grain_select[:,:,zind*num_refine], levels=[iso_val],colors='r')

            # Apply marching cubes method to create mesh and convert back to full sample coordinates in mm
            verts,faces,_,_ = marching_cubes_lewiner(grain_select,iso_val)
            verts += num_refine*np.array([imin,jmin,kmin]) # Convert from subset indices to big indices
            verts_mm = (verts-0.5)*voxelRes/num_refine + origin_mm # Convert from big indices to mm.  The -0.5 accounts for the fact that the point cloud is in the middle of a cell.
            verts_mm = verts_mm[:,[1,0,2]]  # things got flipped somewhere so flip back to match delaunay method mesh orientation

            # run through meshfix repair just in case
            meshfix = mf.MeshFix(verts_mm,faces)
            meshfix.repair()
            mesh = meshfix.mesh

            # reduce number of faces (triangles) for computational speed and output file size
            mesh.decimate(decimate_val,inplace=True)

            # save mesh
            # mesh.save(os.path.join(path,'mc_mesh_%i.vtk')%grain)


        # This section writes VTK file for individual grains, if option is selected, and combines grains into full sample mesh
        # Define min/max bounds of each dimension
        xBounds=[np.min(x),np.max(x)]
        yBounds=[np.min(y),np.max(y)]
        zBounds=[np.min(z),np.max(z)]

        # creates new polygon if this is first time through the loop
        if newPoly == True:
            polyShell=CRRELPolyData._CRRELPolyData(mesh,xBounds,yBounds,zBounds,voxelRes,100.,description='REAL Snow Mesh')

            if create_individual == True:
                print(outpath + 'Grain_{}.vtk'.format(grain))
                polyShell.WritePolyDataToVtk(os.path.join(outpath, 'Grain_{}.vtk'.format(grain)))  # write VTK file for individual grain
            newPoly=False
        else:
            poly1=CRRELPolyData._CRRELPolyData(mesh,xBounds,yBounds,zBounds,voxelRes,100.,description='REAL Snow Mesh')

            if create_individual == True:
                poly1.WritePolyDataToVtk(os.path.join(outpath, 'Grain_{}.vtk'.format(grain)))  # write VTK file for individual grain

            polyShell.appendPoly(poly1)

        if check == True:
            break

        time2=datetime.now()
        total_seconds=(time2-time1).total_seconds()
        # print('Grain mesh total execution time = ' + str(time2-time1))
        # print('Refining took ' + str((refine_end_time.total_seconds()/(total_seconds))*100) + ' % of the meshing time')
        # print('Smoothing took ' + str((smooth_end_time.total_seconds()/(total_seconds))*100) + ' % of the meshing time')

    # get bounds of full mesh
    xBounds=[np.min(X),np.max(X)]
    yBounds=[np.min(Y),np.max(Y)]
    zBounds=[np.min(Z),np.max(Z)]

    # repair full mesh
    meshfix = mf.MeshFix(pv.PolyData(polyShell.GetPolyData()))
    meshfix.repair()
    ## Shell = polydata of repaired mesh!
    shell =meshfix.mesh  #this is vtk poly data!
    CRRELPolyData._CRRELPolyData(shell,xBounds,yBounds,zBounds,voxelRes,100.,description='REAL Snow Mesh')
    #REBUILD ONE GIANT MESH!! ##


    # Set 3D rendering options for plot
    camera = vtk.vtkCamera()
    camera.SetPosition(np.min(x),np.min(y),np.max(Z)+2*voxelRes)
    camera.SetFocalPoint(np.min(x),np.min(y),np.max(z))
    camera.SetViewUp(0,0,0)

    renderer=RenderFunctions.Render3DMesh(polyShell)


    if check == True:
        RenderFunctions.addPoint(renderer, (x1,y1,z1), radius=voxelRes, color=[1.0, 0.0, 0.0], opacity=1.0)

    RenderFunctions.ShowRender(renderer,camera=camera)

    if check == False:
        polyShell.WritePolyDataToVtk(os.path.join(outpath,fullMeshName))  # write out full sample mesh

    fulltime2=datetime.now()
    print("Finished, total time = %.1f seconds"%((fulltime2-fulltime1).total_seconds()))
    plt.show()

    # copy over VTK files for first 30 borderless grains to GRAINS folder
    for i in range (0,30):
        shutil.copy(os.path.join(outpath,'Grain_{}.vtk'.format(borderless[i])),os.path.join(GrainPath,'Grain_{}.vtk'.format(borderless[i])))

    # write properties to CSV
    dataFrame=pd.DataFrame.from_dict(dataDict)
    dataFrame.to_csv(outpath / 'Mesh_Properties.csv')

    end_time = (datetime.now() - begin_time)
    print('Mesh Generation Execution Time: ' + str(end_time))
    if saveMeshParams==True:
        with open(os.path.join(outpath,'MeshParameters.txt'), 'a') as file:
           file.write('\n')
           file.write('Mesh Generation Execution Time: ' + str(end_time) + ' \n')
