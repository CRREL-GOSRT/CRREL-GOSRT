
class _CRRELPolyData:
    """This is a private class contatining the shell data! required for holding poly data information!
       You should not attempt to call this function externally.

       Note that this class is unique to the CRREL RTM, and is used throughout the larger models in place of
       standard vtk PolyData.
    """
    def __init__(self,mesh,xBounds,yBounds,zBounds,resolution,density,units='mm',description='unknown'):
        import vtk
        self.xBounds=xBounds
        self.yBounds=yBounds
        self.zBounds=zBounds

        triFilter=vtk.vtkTriangleFilter()
        triFilter.SetInputDataObject(mesh)
        triFilter.Update()

        self.PolyData=triFilter.GetOutput()

        self.isCRRELPolyData=True
        self.description=description

        # Create OBB (oriented bounding box) tree
        self.obbTree = vtk.vtkOBBTree()
        self.obbTree.SetDataSet(self.PolyData)
        self.obbTree.BuildLocator()

        # Create a new 'vtkPolyDataNormals' and connect to triangulated surface mesh
        normalsCalcMesh = vtk.vtkPolyDataNormals()
        normalsCalcMesh.SetInputData(self.PolyData)
        normalsCalcMesh.ComputePointNormalsOff()
        normalsCalcMesh.ComputeCellNormalsOn()
        normalsCalcMesh.Update()
        self.normalsMesh = normalsCalcMesh.GetOutput().GetCellData().GetNormals()
        normalsCalcMesh=None ## Free up just a bit of memory.

        self.Units='mm'

        self.Resolution=resolution
        self.Density=density
        self.NumCells=mesh.GetNumberOfCells()
        self._RefracData=None
        self.Material = None
        self.BoundaryCondition='open'

    def __repr__(self):
        """Returns basic information on the class"""
        return '<%s | Units = %s | Density = %.1f kg m^3 | Number of Cells = %i' % (
            self.__class__.__name__, self.Units,self.Density,self.NumCells)

    def __len__(self):
        """Length returns number of cells in polymesh"""
        return self.NumCells

    def GetPolyData(self):
        return self.PolyData

    def SetPolyData(self,PolyData):
        self.PolyData=PolyData

    def GetObbTree(self):
        return self.obbTree

    def GetNormalsMesh(self):
        return self.normalsMesh

    def GetRefractiveIndex(self,wavelength,units='nm'):
        import numpy as np
        import re
        if isinstance(self._RefracData,type(None)):
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
        k=np.exp(np.interp(np.log(wavelength),np.log(self._RefracData.wave.values),np.log(self._RefracData.Im.values)))
        RI=np.interp(np.log(wavelength),np.log(self._RefracData.wave.values),self._RefracData.Re.values)
        abso=(4.*np.pi*k/wavelength)*1e-3 # absorption coefficient
        n=RI ## real part of refractive index

        return n,abso,RI+k*1.j

    def AssignMaterial(self,material,filePath):
        import glob as glob
        import os
        from pandas import read_csv

        files=glob.glob(os.path.join(filePath,'*.csv'))
        allowed=[os.path.split(f)[1].split('_')[0] for f in files]
        if material.lower() not in allowed:
            print('%s is not allowe, defaulting to %s '%(material.lower(),allowed[0]))
            print("Allowed Materials:")
            for i in allowed:
                print(" -- %s"%i)
            material=allowed[0]


        RIfile=glob.glob(os.path.join(filePath,'*'+material.lower()+'*.csv'))[0]
        self.RIFile=RIfile

        ##open Refractive index file
        RIdata=read_csv(RIfile)
        self._RefracData=read_csv(RIfile)
        ## convert from um to meters
        self._RefracData.wave=self._RefracData.wave*1E-6
        self.Material=material


    def Voxelize(self,scale=120):
        import pyvista as pv
        Mesh=pv.wrap(self.GetPolyData())
        voxels = pv.voxelize(Mesh, density=Mesh.length/scale,check_surface=False)
        self.SetPolyData(voxels.extract_geometry())

    def appendPoly(self,other):
        import vtk
         # Append the two meshes

        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(self.GetPolyData())
        appendFilter.AddInputData(other.GetPolyData())

        appendFilter.Update()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendFilter.GetOutputPort())
        cleanFilter.Update()

        self.SetPolyData(cleanFilter.GetOutput())
        appendFilter=None
        cleanFilter=None

    def RotateMesh(self,angle,axis='x'):
        import pyvista as pv
        Mesh=pv.wrap(self.GetPolyData())
        if axis.lower() not in ['x','y','z']:
            print('WARNING --> axis %s not in x,y,z, defaulting to x-axis'%axis)
            axis='x'


        if axis.lower() == 'x':
            Mesh.rotate_x(angle)
        elif axis.lower() == 'y':
            Mesh.rotate_y(angle)
        elif axis.lower() == 'z':
            Mesh.rotate_z(angle)

        self.SetPolyData(Mesh.clean())

    def WritePolyDataToVtk(self,FileName):
         import vtk
         writer = vtk.vtkPolyDataWriter()
         writer.SetInputData(self.GetPolyData())
         writer.SetFileName(FileName)
         writer.Write()
