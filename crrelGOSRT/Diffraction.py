import sys
import numpy as np
import glob as glob
import os

from scipy.signal import argrelextrema

import numpy as np
from matplotlib import pyplot as plt


class RenderedDiffractionParticle:
    def __init__(self,particle,**kwargs):
        import pyvista as pv
        from scipy.interpolate import griddata
        """
        This is a method to hold all relevent information and functions required to compute diffraction around a "real" particle
        One key note is that the user must have pyvista installed (https://docs.pyvista.org) in order to handle to loading an manipulation 
        of rendered snow grains in .vtk format.  This function is called as part of the CRREL-GOSRT "slab" model if diffraction is indicated in the 
        namelist, and if the diffraction shape is provided as a .vtk file.  Otherwise, the model will compute diffraction using an idealized particle.

        Inputs: particle (string): Full path to a specified .vtk grain particle.
       

        kwargs (optional keyword arguments): 
                nx - number of x-gridcells to use when generating a 2D uniform grid to store the shadow information
                ny - number of y-gridcells to use when generating a 2D uniform grid to store the shadow information
                ** Note: The larger nx and ny are, the better resolved the particle will be, but it will be slower to compute diffraction and take up more memory.'

                normal - tuple specifying the direction of the projection along the particle. the default (0,0,1) value indicates a vertically oriented vector along the "z" direction
                CellSamples - number of samples to draw from each mesh cell in the projected shadow to convert the projected shadow to a 2D uniform grid.
                    see: LoadGrain function for more information.

        Returns: Object class containing the grain mesh and other accompanying information.

        """

        DefaultKeywords = { 'normal': (0,0,1),'CellSamples':5,'nx':150,'ny':150}
        kwargs = { **DefaultKeywords , **kwargs }
        self.ShapeDictionary=kwargs
        self.Shape = particle

        assert os.path.isfile(particle), "The file '%s' does not exist, please choose a valid particle file"%particle

        mesh = pv.read(particle)
        bounds = mesh.bounds 
        r = (np.mean([(bounds[1]-bounds[0]),(bounds[3]-bounds[2]),(bounds[5]-bounds[4])]))/2.
        self.GrainSize_meters = r / 1000.

        ## set up grid for approximating the shape for the diffraction pattern.
        self.xx = np.linspace(1.01*- self.GrainSize_meters ,1.01* self.GrainSize_meters ,self.ShapeDictionary['nx'])
        self.yy = np.linspace(1.01*- self.GrainSize_meters ,1.01* self.GrainSize_meters ,self.ShapeDictionary['ny'])
        self.dx = self.xx[1] - self.xx[0]
        self.dy = self.yy[1] - self.yy[0]

        self.nx = self.ShapeDictionary['nx'] 
        self.ny = self.ShapeDictionary['ny']

        self.mesh = mesh
        self.mesh_bounds = bounds

        self.Shadow = self.LoadGrain()
        self.ShadowArea = np.sum(self.Shadow)*(self.dx)*(self.dy)

    def __repr__(self):
        return """Idealized Diffraction Shape Class: %s with effective diameter %.2f (mm)
                Projected Shadow Area %.2f (mm^2)
                Additional Keywords: %s

            """%(self.Shape,self.GrainSize_meters*2000.,self.ShadowArea*1000.*1000.,self.ShapeDictionary)
            
    def PlotMesh(self):
        """Simple helper diagnostic function to plot the mesh!"""
        from pyvista import Plotter as pvPlotter
        ##Render the result
        p = pvPlotter()
        p.add_mesh(self.mesh, show_edges=True, opacity=0.5, color="w", lighting=False, label="Grain Particle")
        p.add_mesh(self.mesh_ProjectedShadow , color="maroon", label="Shadow Projection")
        p.add_legend()
        p.show()
        sys.exit()

    def LoadGrain(self):
        r"""
        This is a somewhat awkward (and a little slow) helper function 
        that tries to take the triangulated cells that make up the mesh of the projected shadow area
        and grid them onto a uniform grid made of up equal-area squares, matching the data stored when running idealized shapes.
        This is performed by randomly sampling points within each cell, and comparing the sample positions against a uniform 2D grid.

        Returns array sized nx,ny with 1.0s and 0.0s, where values of 1.0 indicate that this gridcell is within the projected 
        shadow area of the grain mesh.

        Example:
           /\               | 0| 0| 0|
          /. \              | 1| 0| 0|
         /   .\   --------> | 0| 0| 1|
        /    . \            | 0| 0| 1|
       /________\           |_0|_0|_0|

       Inputs: No inputs, all information comes from the parent class (self)

       Returns: 2D gridded array of size nx,ny.
        
        """
        origin = self.mesh.center
        origin[-1]= self.mesh_bounds[5]
        projected = self.mesh.project_points_to_plane(origin=origin, normal=self.ShapeDictionary['normal'])

        self.mesh_ProjectedShadow = projected

        nCells = projected.n_faces  ## Get the number of cells along projected shadow.
        ALL_POINTS=[[],[]]
        for i in range(nCells):
            c_cell = projected.GetCell(i)
            bounds = c_cell.GetBounds()
            xbounds = [bounds[0],bounds[1]]
            ybounds = [bounds[2],bounds[3]]

            x,y = [np.random.uniform(*xbounds,self.ShapeDictionary['CellSamples']),np.random.uniform(*ybounds,self.ShapeDictionary['CellSamples'])]
            x,y = (x- origin[0])/1000.,(y- origin[1])/1000.
            x,y=list(x),list(y)

            ALL_POINTS[0]+=x
            ALL_POINTS[1]+=y

        ALLX = np.array(ALL_POINTS[0])
        ALLY = np.array(ALL_POINTS[1])
        grid_z1 = np.zeros([len(self.xx),len(self.yy)])

        for ii in range(len(self.xx)):
            for jj in range(len(self.yy)):
                center = (self.xx[ii],self.yy[jj])
                xBounds = [center[0]-self.dx,center[0]+self.dx]
                yBounds = [center[1]-self.dy,center[1]+self.dy]
                inX = np.where((ALLX > xBounds[0]) & (ALLX < xBounds[1]))
                if len(inX[0]) == 0:
                    continue
                YLEFT = ALLY[inX]
                inY = np.where((YLEFT > yBounds[0]) & (YLEFT < yBounds[1]))
                if len(inY[0]) == 0:
                    continue

                grid_z1[ii,jj]=1.0

        return grid_z1

    def GetDiffractionPhaseFunction(self,wavelen,NumSamples,scale = 30,PzScale = 1000, verbose = False, print_freq = 1000,maxTheta=5):
            """
            This Function gets uses the projected shadow area of the grain mapped to a uniform 2D grid 
            to compute far-field (Fraunhofer) diffraction around the geometric shadow area.  
            This function computes the diffracted intensity at a random sampling of points on an observation plane perpendicular to the 
            incident ray at distance (PzScale*Grain_Diameter) away from the particle for a specified wavelength.

            Inputs:
                wavelen (float): wavelength of incident radiation in nanometers.
                NumSamples (int): Number of random samples used to compute scattering angles
                scale (optional - float): scaling constant to specify the boundaries for the random x/y sampling on the observation plane.
                    i.e., the observation plane is defined by a sqaure with size: (scale*grain_size) x (scale*grain_size)
                    A larger value here allows for wider scattering angles.

                PzScale (optional - float): A scaling factor determining the distance from the apature to the observation plane:
                    i.e., Pz = PzScale*grain_size
                                                            (scale x grain_size)
                                                            ___________________
                  /\                                       |                   |
                 /  \                                      |     x             | 
                |    |                                     |                   |
               /     /  ---------------------------------> |                   | (scale x grain_size)
               \    |        (PzScale x grain_size)        |                   |
                \   |                                      |                   |  
                 \_/                                       |___________________|


                verbose (optional - bool): Flag to turn on additional output to the terminal
                print_freq (optional - int): only used if "verbose" = True.  How often to update the user on sampling progress.
                maxTheta (optional - float): maximum scattering angle used when interpolating the random samples to a uniform scattering angle array. 
                          Angle is assumed to be in degrees.

            Returns: 
                IterpTheta (array): Array of scattering angles (size = 1000) from 0 - maxTheta (radians)
                IterpIntensity (array): Array of intensity at each scattering angle (size 1000)

            """

            P0 = [0,0,0] ## For simplicity, set to be the origin. ##
            wave_num = 2. * np.pi / (wavelen * 1e-9)  ## units = 1/m
            Mesh_xx,Mesh_yy=np.meshgrid(self.xx,self.yy) ##Mesh-grid the xx and yy coordinates
            wv_meters = wavelen * 1e-9 ### Convert input wavelength to meters.

            ## initialize a list for scattering angles AND a list for intensity
            DiffractedIntensity = []
            DiffractedScatteringAngle = []
            ## loop through all the samples!

            if verbose == True: ## Verbose!
                print("----------------------------------------------------------------------------------------")
                print("Computing scattering angle PDF for diffraction around a %s with an effective radius of %.2f mm"%(self.Shape,self.GrainSize_meters*1000.))
                print("Computing using %i samples, this may take a few moments."%NumSamples)

            for ndx in range(NumSamples):

                if verbose == True:
                    if ndx%print_freq == 0:
                        print("On sample: %i of %i"%(ndx,NumSamples))


                I0 = 1.0 ## initial intensity ==>

                x = np.random.uniform(-scale*self.GrainSize_meters,scale*self.GrainSize_meters) ### Pick a random x sample for this point! 
                y = np.random.uniform(-scale*self.GrainSize_meters,scale*self.GrainSize_meters) ### Pick a random y sample for this point! 
                Pz = self.GrainSize_meters*PzScale ## Distance from shadow / apature to observation plane.

                ## some geometry here...
                mag1 = np.sqrt(x**2+y**2.+Pz**2)
                mag2 = np.sqrt(x**2.+y**2.)
                mag3 = np.sqrt(P0[0]**2.+P0[1]**2+Pz**2.)

                dot1 = np.array([x/mag1,y/mag1,Pz/mag1])
                dot2 = np.array([P0[0]/mag3,P0[1]/mag3,Pz/mag3])

                cos_theta = np.dot(dot1,dot2) ## this is the scattering direction.

                theta = np.arccos(cos_theta) ## get scattering angle in radians
                phi = np.arccos(x/mag2) ## get azimuthal angle
            
                ## Compute diffracted intensity.
                ADD = np.exp(-1j*wave_num*(Mesh_xx*np.cos(phi)+Mesh_yy*np.sin(phi))*np.sin(theta))*self.dx*self.dy
                ADD = np.ma.masked_where(self.Shadow < 1, ADD).filled(0.0) ## mask out all values not within the shadow area
                DFF = np.sum(ADD) ## Integrate.

                I0 = I0 * self.ShadowArea / (wv_meters)
                u0 = np.sqrt(I0)

                up = -(1j*u0)/(mag1*wv_meters)*np.exp(-1j*wave_num*Pz)*DFF
                I = np.abs(up)**2. # this is the diffracted intensity!
                #########
            
                DiffractedIntensity.append(I) ##W/m2 (assuming I0 = 1.)
                DiffractedScatteringAngle.append(theta) ## Radians ##

            

            ## Sort by theta.
            DiffractedIntensity = [x for _, x in sorted(zip(DiffractedScatteringAngle, DiffractedIntensity))]
            DiffractedScatteringAngle = sorted(DiffractedScatteringAngle) ## Radians


            ## Generate PDF of scattering directions on a uniform array. -> Ensures proper converstion into a probability distribution functions.
            IterpTheta = np.linspace(0.0,maxTheta*np.pi/180.,1000) ## Radians
            IterpIntensity = np.interp(IterpTheta,DiffractedScatteringAngle,DiffractedIntensity)
            

            if verbose == True:
                print("Finished determining probability distribution function for a %s with a %.2f mm effective radius at wavelength = %.1f nm"%(self.Shape,self.GrainSize_meters*1000.,wavelen))
                print("----------------------------------------------------------------------------------------")

            # KEEP FOR NOW FOR POSSIBLE DIAGNOSTICS, BUT DELETE LATER! ###
            # plt.figure()
            # ax=plt.subplot(111)
            # plt.plot(np.array(DiffractedScatteringAngle )*180./np.pi,np.array(DiffractedIntensity),color='k',zorder=8)
            # ax1=ax.twinx()
            # plt.hist(choice*180./np.pi,edgecolor='k',bins=50)
            # plt.show()
            # sys.exit()
        
            return np.array(IterpTheta), np.array(IterpIntensity)

class IdealizedDiffractionParticle:
    def __init__(self,shape,diameter,**kwargs):
        """
        This is a method to hold all relevent information and functions required to compute diffraction around idealized particles with
        specified 2D shapes and sizes. Currently allowd shapes are:

            -   Circles (sphere/disk)
            -   Squares
            -   Triangles
            -   Hexagons
            -   needles

            **Note that needles have additional keyword information: needle_aspect and needle_cap_ratio, that specify the shape of the needle.**
        
        Inputs: 
            shape (string): string specifing which shape to use, must by in the list of allowed shapes.
            diameter (float): number specifying the size (equivalent diameter) of the particle in mm.

            kwargs (optional keyword arguments): 
                nx - number of x-gridcells to use when generating a 2D uniform grid to store the shadow information
                ny - number of y-gridcells to use when generating a 2D uniform grid to store the shadow information
                ** Note: The larger nx and ny are, the better resolved the particle will be, but it will be slower to compute diffraction and take up more memory.

                needle_aspect - only affects needles; specifies the aspect ratio of the needle.  A value approaching 1 will be a "wide" square needle, 
                                whereas a value approaching 0 will be a narrow needle. Must be less than 1.
                needle_cap_ratio - only affects needles; specifies how large the pointy caps are compared to the needle.  Values approaching 0.5 yield diamonds, 
                                   whereas values approaching 0 yield nearly rectangular particles.  Must be less than 0.5.

                /\  <---- needle "cap"             
               /  \  
              |    |  <---------- General shape of needle.
              |    | 
               \  / <---- needle "cap"  
                \/
            
        Returns: Object class containing the grain mesh and other accompanying information.
        """
        
        ## First load particle! ##

        self.AllowedShapes = ['sphere','circle','disk','square','triangle','hexagon','needle'] ## set a list of allowed shapes.

        assert shape.lower() in self.AllowedShapes, "The shape %s is not specified in the allowed shapes.  Please see help(IdealizedDiffractionPattern) for more information"%(shape.lower())

        ## handle keyword arguments for specific shapes related to defaults.
        DefaultKeywords = { 'needle_aspect': 0.06, 'needle_cap_ratio': 0.06,'nx':150,'ny':150} 
        kwargs = { **DefaultKeywords , **kwargs }

        self.ShapeDictionary=kwargs

        self.GrainSize_meters = 0.5*diameter/1000. ## Grain size here is converted from effective diameter to radius.
        self.Shape = shape

        ## set up grid for approximating the shape for the diffraction pattern.
        self.xx = np.linspace(1.01*- self.GrainSize_meters ,1.01* self.GrainSize_meters ,self.ShapeDictionary['nx'])
        self.yy = np.linspace(1.01*- self.GrainSize_meters ,1.01* self.GrainSize_meters ,self.ShapeDictionary['ny'])
        self.dx = self.xx[1] - self.xx[0]
        self.dy = self.yy[1] - self.yy[0]

        self.nx = self.ShapeDictionary['nx']
        self.ny = self.ShapeDictionary['ny']
        
    
        SHAPE,Area = self.GetIdealShape()

        self.Shadow = SHAPE
        self.ShadowArea = Area ## meters^2


    def __repr__(self):
        return """Idealized Diffraction Shape Class: %s with effective diameter %.2f (mm)
                  Projected Shadow Area %.2f (mm^2)
                  Additional Keywords: %s

                """%(self.Shape,self.GrainSize_meters*2000.,self.ShadowArea*1000.*1000.,self.ShapeDictionary)

    
    def GetIdealShape(self):
        
        """Doc String here """
        from crrelGOSRT.Utilities import isInside


        ShapeArray = np.zeros([self.nx,self.ny])
        if self.Shape.lower() in ['circle','sphere','disk']:
            for idx, i in enumerate(self.xx):
                for jdx, j in enumerate(self.yy):
                    r = np.sqrt(i**2.+j**2.)
                    if r < self.GrainSize_meters:
                        ShapeArray[idx,jdx] = 1.0

        elif self.Shape.lower() in ['triangle']:
            for idx, i in enumerate(self.xx):
                for jdx, j in enumerate(self.yy):
                    inside = isInside(-self.GrainSize_meters, -self.GrainSize_meters, self.GrainSize_meters, 0, -self.GrainSize_meters, self.GrainSize_meters, self.xx[idx], self.yy[jdx])
                    if inside == True:
                        ShapeArray[idx,jdx] = 1.0

        elif self.Shape.lower() in ['hexagon']:
            for idx, i in enumerate(self.xx):
                for jdx, j in enumerate(self.yy):
                    angle = 30.
                    for t in range(6):
                        x1 = 0.0
                        y1 = 0.0
                        x2 = self.GrainSize_meters*np.cos(angle/180.*np.pi)
                        y2 = self.GrainSize_meters*np.sin(angle/180.*np.pi)
                        x3 = self.GrainSize_meters*np.cos((angle+60.)/180.*np.pi)
                        y3 = self.GrainSize_meters*np.sin((angle+60.)/180.*np.pi)
                        inside = isInside(x1, y1, x2, y2, x3, y3, self.xx[idx], self.yy[jdx])
                        if inside == True:
                            ShapeArray[idx,jdx] = 1.0
                        angle+=60.

        elif self.Shape.lower() in ['needle']:
            aspect_ratio = self.ShapeDictionary['needle_aspect']
            cap_ratio = self.ShapeDictionary['needle_cap_ratio']

            if cap_ratio > 0.5: ## too large.
                cap_ratio = 0.5

            if aspect_ratio > 1.0: ## too large.
                aspect_ratio = 1.0

            for idx, i in enumerate(self.xx):
                for jdx, j in enumerate(self.yy):
                    if -aspect_ratio*self.GrainSize_meters <= self.yy[jdx] <= aspect_ratio*self.GrainSize_meters:
                        if -(1.-2.*cap_ratio)*self.GrainSize_meters <= self.xx[idx] <= (1.-2.*cap_ratio)*self.GrainSize_meters:
                            ShapeArray[idx,jdx] = 1.0
                        
                    x1,y1 = -self.GrainSize_meters*(1.-2.*cap_ratio),aspect_ratio*self.GrainSize_meters 
                    x2,y2 = -self.GrainSize_meters,0.0
                    x3,y3 = -self.GrainSize_meters*(1.-2.*cap_ratio),-aspect_ratio*self.GrainSize_meters 

                    inside = isInside(x1, y1, x2, y2, x3, y3, self.xx[idx], self.yy[jdx])
                    if inside == True:
                        ShapeArray[idx,jdx] = 1.0

                    x1,y1 = self.GrainSize_meters*(1.-2.*cap_ratio),aspect_ratio*self.GrainSize_meters 
                    x2,y2 = self.GrainSize_meters,0.0
                    x3,y3 = self.GrainSize_meters*(1.-2.*cap_ratio),-aspect_ratio*self.GrainSize_meters 

                    inside = isInside(x1, y1, x2, y2, x3, y3, self.xx[idx], self.yy[jdx])
                    if inside == True:
                        ShapeArray[idx,jdx] = 1.0

        elif self.Shape.lower() in ['square','cube']:
            ShapeArray[:] = 1.0

        Area = np.sum(ShapeArray)*(self.dx)*(self.dy)

        return ShapeArray,Area


    def GetDiffractionPhaseFunction(self,wavelen,NumSamples,scale = 30,PzScale = 1000, verbose = False, print_freq = 1000,maxTheta=5):
        """
            This Function gets uses the projected shadow area of the grain mapped to a uniform 2D grid 
            to compute far-field (Fraunhofer) diffraction around the geometric shadow area.  
            This function computes the diffracted intensity at a random sampling of points on an observation plane perpendicular to the 
            incident ray at distance (PzScale*Grain_Diameter) away from the particle for a specified wavelength.

            Inputs:
                wavelen (float): wavelength of incident radiation in nanometers.
                NumSamples (int): Number of random samples used to compute scattering angles
                scale (optional - float): scaling constant to specify the boundaries for the random x/y sampling on the observation plane.
                    i.e., the observation plane is defined by a sqaure with size: (scale*grain_size) x (scale*grain_size)
                    A larger value here allows for wider scattering angles.

                PzScale (optional - float): A scaling factor determining the distance from the apature to the observation plane:
                    i.e., Pz = PzScale*grain_size
                                                            (scale x grain_size)
                                                            ___________________
                  /\                                       |                   |
                 /  \                                      |     x             | 
                |    |                                     |                   |
               /     /  ---------------------------------> |                   | (scale x grain_size)
               \    |        (PzScale x grain_size)        |                   |
                \   |                                      |                   |  
                 \_/                                       |___________________|


                verbose (optional - bool): Flag to turn on additional output to the terminal
                print_freq (optional - int): only used if "verbose" = True.  How often to update the user on sampling progress.
                maxTheta (optional - float): maximum scattering angle used when interpolating the random samples to a uniform scattering angle array. 
                          Angle is assumed to be in degrees.

            Returns: 
                IterpTheta (array): Array of scattering angles (size = 1000) from 0 - maxTheta (radians)
                IterpIntensity (array): Array of intensity at each scattering angle (size 1000)

            """

        P0 = [0,0,0] ## For simplicity, set to be the origin. ##
        wave_num = 2. * np.pi / (wavelen * 1e-9)  ## units = 1/m
        Mesh_xx,Mesh_yy=np.meshgrid(self.xx,self.yy) ##Mesh-grid the xx and yy coordinates
        wv_meters = wavelen * 1e-9 ### Convert input wavelength to meters.

        ## initialize a list for scattering angles AND a list for intensity
        DiffractedIntensity = []
        DiffractedScatteringAngle = []
        ## loop through all the samples!

        if verbose == True:
            print("----------------------------------------------------------------------------------------")
            print("Computing scattering angle PDF for diffraction around a %s with an effective radius of %.2f mm"%(self.Shape,self.GrainSize_meters*1000.))
            print("Computing using %i samples, this may take a few moments."%NumSamples)

        for ndx in range(NumSamples):

            if verbose == True:
                 if ndx%print_freq == 0:
                     print("On sample: %i of %i"%(ndx,NumSamples))


            I0 = 1.0

            x = np.random.uniform(-scale*self.GrainSize_meters,scale*self.GrainSize_meters) ### Pick a random x sample for this point! 
            y = np.random.uniform(-scale*self.GrainSize_meters,scale*self.GrainSize_meters) ### Pick a random y sample for this point! 
            Pz = self.GrainSize_meters*PzScale ## Distance from shadow / apature to observation plane.

            mag1 = np.sqrt(x**2+y**2.+Pz**2)
            mag2 = np.sqrt(x**2.+y**2.)
            mag3 = np.sqrt(P0[0]**2.+P0[1]**2+Pz**2.)

            dot1 = np.array([x/mag1,y/mag1,Pz/mag1])
            dot2 = np.array([P0[0]/mag3,P0[1]/mag3,Pz/mag3])

            cos_theta = np.dot(dot1,dot2) ## this is the scattering direction.

            theta = np.arccos(cos_theta)
            phi = np.arccos(x/mag2)
           
            ADD = np.exp(-1j*wave_num*(Mesh_xx*np.cos(phi)+Mesh_yy*np.sin(phi))*np.sin(theta))*self.dx*self.dy
            ADD = np.ma.masked_where(self.Shadow < 1, ADD).filled(0.0)
            DFF = np.sum(ADD)

            I0 = I0 * self.ShadowArea / (wv_meters)

            u0 = np.sqrt(I0)


            up = -(1j*u0)/(mag1*wv_meters)*np.exp(-1j*wave_num*Pz)*DFF
            I = np.abs(up)**2.
        
            DiffractedIntensity.append(I) ##W/m2 (assuming I0 = 1.)
            DiffractedScatteringAngle.append(theta) ## Radians ##

        

        DiffractedIntensity = [x for _, x in sorted(zip(DiffractedScatteringAngle, DiffractedIntensity))]
        DiffractedScatteringAngle = sorted(DiffractedScatteringAngle) ## Radians

        ## Generate PDF of scattering directions (cos-theta.)

        IterpTheta = np.linspace(0.0,maxTheta*np.pi/180.,2000) ## Radians
        IterpIntensity = np.interp(IterpTheta,DiffractedScatteringAngle,DiffractedIntensity)
        


        if verbose == True:
            print("Finished determining probability distribution function for a %s with a %.2f mm effective radius at wavelength = %.1f nm"%(self.Shape,self.GrainSize_meters*1000.,wavelen))
            print("----------------------------------------------------------------------------------------")

        # KEEP FOR NOW FOR POSSIBLE DIAGNOSTICS, BUT DELETE LATER! ###
        # plt.figure()
        # ax=plt.subplot(111)
        # plt.plot(np.array(DiffractedScatteringAngle )*180./np.pi,np.array(DiffractedIntensity),color='k',zorder=8)
        # ax1=ax.twinx()
        # plt.hist(choice*180./np.pi,edgecolor='k',bins=50)
        # plt.show()
        # sys.exit()
        return np.array(IterpTheta), np.array(IterpIntensity)