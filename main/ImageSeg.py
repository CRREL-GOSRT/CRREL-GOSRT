""" This file is the main script for generating a mesh from binarized microCT
    image stacks. The output is a VTK 3D mesh file, options for creating meshes
    for individual grains or the full sample.


    Notes
    -----
      First Created by Ted Letcher January 2021
      Updated by Julie Parno May 2021
      
      
    Example for generating mesh
    -----------------------------     

      import ImageSeg
      
      path='F:\\Snow_Optics\\MicroCT_Data\\1Feb_UVD_Pit1'
      subpath = 'Sample_1_20-13cm\\1Feb_UVD_Pit1_1_1\\1Feb_UVD_Pit1_1_1_Rec\\VOI\\Snow'
      outpath = 'contour\\1Feb_UVD_Pit1\\1Feb_UVD_Pit1_1_1\\'  # Output Path for .vtk files ( set equal to subpath if you just want it there)
      XYstart = 3.0
      XYend = 12.0
      depthTop = 200
      Ztop = 195.0  # Top most sample depth
      allowedBorder = 10000000  # Number of points allowed to be on a mesh border
      minpoints = 25  # Minimum number of points allowed for each grain
      minGrainSize = 0.3 # This sets the minimum grainsize (in mm) for the peak-local-max function
      decimate = 0.9

      ImageSeg.meshgen(path,subpath,outpath,XYstart,XYend,depthTop,Ztop,allowedBorder,minpoints,minGrainSize,decimate,check=False)
    
    
    
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
      July 2021
"""

# import sys
# sys.path.append('C:\\Users\\RDCRLJTP\\Documents\\Projects\\Snow_Optics\\Code\\CRREL-GOSRT\\main')
import vtk
import pandas as pd
import numpy as np
import pyvista as pv
import pymeshfix as mf
from matplotlib import pyplot as plt
import glob
from matplotlib.image import imread
import CRRELPolyData
import RenderFunctions
from scipy import signal, ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.measure import marching_cubes_lewiner
from skimage.filters import gaussian
from datetime import datetime, timedelta
import os
from pathlib import Path
    
def ImagesToArray(path,outpath,XYstart,XYend,depthTop,Ztop,voxelRes,thresh=0.9,check=True,savePropsToCsv=True,saveMeshParams=True):
    
    """
    This function generates a mesh from binarized microCT
    image stacks. The output is a VTK 3D mesh file, options for creating meshes
    for individual grains or the full sample.
    
    Note that depths are measured from ground (d=0)
    
    INPUTS:
        path: main path for microCT data
        subpath: Sub-folder with microCT images in it.
        outpath: Output Path for .vtk files ( set equal to subpath if you just want it there)
        XYstart: The starting point in XY for the mesh subset within a sample image (in plane view) in millimeters, assuming the the left most pixel is 0.
        XYend: The ending point in XY for the mesh subset within a sample image (in plane view) in millimeters, assuming the the left most pixel is 0.
        depthTop: Top snow depth of scanned sample (in mm) --> Usually corresponds to file name in "subpath" variable
           * Note that depthTop is critically important in getting the depth-oriented sample correct!
        Ztop: Top depth selected for mesh sample subset (must be within the sample depth)
        allowedBorder: Number of points allowed to be on a mesh border. This can eliminate "flat" grain boundaries that are up against the mesh boundary.
            * Larger number = less restrictive (if you want everything set to 1E6 or more)
        minpoints: Minimum number of points allowed for each grain
        minGrainSize: This sets the minimum grainsize (in mm) for the peak-local-max function for watershed segmentation, generally 0.3 - 0.8mm, depending on sample 
        decimate_val: The decimal percentage of mesh triangles to eliminate in decimation (recommended: 0.9)
        check: If True, will stop after first grain to allow for visual inspection prior to running the whole code
            If True, it will:
                - return a vtk window of a single grain for visual inspection, it will also
                - plot up a 2D image slice showing the grain separation by coloring each grain differently
                - Ideally, you want the boundaries to mostly fall along grain-necks and separate areas that look distinct
                - Note that in this figure, the air space will plot as dark purple (0).
        
    """
    # begin_time = datetime.now()
    
    ## Flag to indicate, that depths come from a text file, otherwise, they are numbered according to PNG filenames.
    FromTextFile=False                                                                                                                             
    
    ## Properties related to file structure / resolution.
    csvIndexStart=2                     
    txtFilePfx='batman1'
    
    alpha=voxelRes
    binary_path=''
    
    # Text file describing sample depths
    # textpath= os.path.join(folder,binary_path.split('/')[0],txtFilePfx)
    
    # # this sets the output path!
    # path= os.path.join(path,'VTK',outpath)
    # #path = os.path.dirname(path)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    
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
           
           
    # If there is a text file describing sample depths, read that in
    if FromTextFile == True:
        textfile=glob.glob(textpath)[0]
        data=pd.read_csv(textfile,header=41)                                   
    
    # Read in full stack of plane view images
    pngfiles=sorted(glob.glob(os.path.join(path,'*.png')))
    
    # Calculate and define depths
    Z=[]
    finalpngs=[]
    for png in pngfiles:
        if FromTextFile == True:    
            bmpmatch=png.split('/')[-1].split('.')[0]+'.bmp'
            datamatch=data[data['File name'] == bmpmatch]
            depth=float(datamatch['Z position'][:])                                            
        else:
            depthNum =float(png.split('.png')[0].split('_')[-1])
            depth = depthNum*voxelRes                             
    
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
    origin_mm = np.array([XYstart, XYstart, np.min(Z)])
    
    return SNOW,X,Y,Z,origin_mm
    
def GrainSeg(SNOW,voxelRes,minGrainSize,outpath,thresh=0.9,saveMeshParams=True):
    
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
  
def MeshGen(grains,grain_labels,properties,voxelRes,X,Y,Z,origin_mm,allowedBorder,minpoints,decimate_val,outpath,fullMeshName,saveMeshParams=True,savePropsToCsv=True,check=True,create_individual=True):
    
    # If option selected, start creating data dictionary to write out properties to a CSV
    if savePropsToCsv == True:
        dataDict={}
        dataDict['Name']=[]
        dataDict['Center x (mm)'] = []
        dataDict['Center y (mm)'] = []
        dataDict['Center z (mm)'] = []
        dataDict['Radius (mm)'] = []
        dataDict['Volume (mm^3)'] = []
        
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
    for gdx, grain in enumerate(grains[1:]):
        time1=datetime.now()
        newPoly=True
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
    
    #%% ### Mesh generation via contour method and Marching Cubes
         
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
        
        
    #%% This section writes VTK file for individual grains, if option is selected, and combines grains into full sample mesh
        # Define min/max bounds of each dimension
        xBounds=[np.min(x),np.max(x)]
        yBounds=[np.min(y),np.max(y)]
        zBounds=[np.min(z),np.max(z)]
    
        # creates new polygon if this is first time through the loop
        if newPoly == True:
            polyShell=CRRELPolyData._CRRELPolyData(mesh,xBounds,yBounds,zBounds,voxelRes,100.,description='REAL Snow Mesh')
    
            if create_individual == True:
                print(os.path.join(outpath,'Grain_%i.vtk'%grain))
                polyShell.WritePolyDataToVtk(outpath / 'Grain_%i.vtk'%grain)  # write VTK file for individual grain
            newPoly=False
        else:
            poly1=CRRELPolyData._CRRELPolyData(mesh,xBounds,yBounds,zBounds,voxelRes,100.,description='REAL Snow Mesh')
    
            if create_individual == True:
                poly1.WritePolyDataToVtk(outpath / 'Grain_%i.vtk'%grain)  # write VTK file for individual grain
    
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
        polyShell.WritePolyDataToVtk(outpath / fullMeshName)  # write out full sample mesh
    
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


#%% Plots

def slice_plot(path,outpath,pltidx,voxelRes,save=True):
    outpath = os.path.join(path,'VTK',outpath)
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(os.path.join(outpath,'CRREL_MESH.vtk'))
    reader.Update()
    shell = reader.GetOutput()
    
    snowbin = np.load(os.path.join(outpath,'microCT_arr.npy'))
    
    ## Plot a slice of the mesh to compare with microCT scan binary image
    m = pv.PolyData(shell)
    plt.figure(figsize=(12,9))
    ax=plt.subplot(1,1,1)
    slice_depth = m.bounds[4]+(pltidx)*voxelRes      
    ax.imshow(snowbin[:,:,pltidx][::-1],cmap='binary_r',extent=[m.bounds[0],m.bounds[1],m.bounds[2],m.bounds[3]])
    slices = m.slice(normal='z',origin=(m.center[0],m.center[1],slice_depth))
    pts = slices.points
    ax.plot(pts[:,0],pts[:,1],'.')
    plt.title('Mesh vs MicroCT at %.4f mm depth' % slice_depth)
    if save==True:
        plt.savefig(os.path.join(outpath,('mesh_compare_%.4f.png' % slice_depth)))
