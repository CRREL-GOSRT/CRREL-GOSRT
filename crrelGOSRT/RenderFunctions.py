import vtk

def addPoint(renderer, p, radius=1.0, color=[0.0, 0.0, 0.0], opacity=1.0):
    point = vtk.vtkSphereSource()
    point.SetCenter(p)
    point.SetRadius(radius)
    point.SetPhiResolution(100)
    point.SetThetaResolution(100)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(point.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)

    renderer.AddActor(actor)


def addLine(renderer, p1, p2, color=[0.0, 0.0, 1.0], lineWidth=4.0):
    line = vtk.vtkLineSource()
    line.SetPoint1(p1)
    line.SetPoint2(p2)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(lineWidth)

    renderer.AddActor(actor)


def RenderSun(CRRELSunObject, bcolor=(0,0,0), renderer= None):
    mapperSun = vtk.vtkPolyDataMapper()
    mapperSun.SetInputData(CRRELSunObject.GetPolyData())

    if isinstance(renderer, type(None)):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(*bcolor) # Background color black
    # Create actor
    actor= vtk.vtkActor()
    actor.SetMapper(mapperSun)
    actor.GetMapper().ScalarVisibilityOff()
    actor.GetProperty().SetColor(CRRELSunObject.CentralColor)  #set color to yellow
    actor.GetProperty().SetEdgeColor(CRRELSunObject.CentralColor)  #render edges as
    renderer.AddActor(actor)

    return renderer

def SolarIntersection(CRRELSunObject,CRRELPolyData,bcolor=(0,0,0),renderer=None):
    ## initialize renderer
    if isinstance(renderer, type(None)):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(*bcolor) # Background color black

    for idx in range(CRRELSunObject.CellCenters.GetNumberOfPoints()):
        # Get coordinates of sun's cell center
        pointSun = CRRELSunObject.CellCenters.GetPoint(idx)
        # Get normal vector at that cell
        normalSun = CRRELSunObject.normalsMesh.GetTuple(idx)
        # Calculate the 'target' of the ray based on 'RayCastLength'
        pointRayTarget = np.array(pointSun)+100000*np.array(normalSun)

        ## Cast ray here! Ted Stopped Octover 30, 2020
        pointSun
        pointRayTarget




def Render3DMesh(CRRELPolyData,bcolor=(0,0,0),opacity=0.95,orientation=True,BoundBox=True,renderer=None):
    ## initialize renderer
    if isinstance(renderer, type(None)):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(*bcolor) # Background color black

    if orientation == True:
        x1=CRRELPolyData.xBounds[0]
        y1=CRRELPolyData.yBounds[0]
        z1=CRRELPolyData.zBounds[1]
        x2,y2,z2=x1+1,y1+1,z1+1

        addLine(renderer, [x1,y1,z1], [x2,y1,z1], color=[1.0, 0.0, 0.0], lineWidth=4.0)
        addLine(renderer, [x1,y1,z1], [x1,y2,z1], color=[0.0, 1.0, 0.0], lineWidth=4.0)
        addLine(renderer, [x1,y1,z1], [x1,y1,z2], color=[0.0, 0.0,1.0], lineWidth=4.0)



    if BoundBox == True:
        ## add more lines. -- >x!
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        ## add more lines. -- >y!
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]],
                [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        ## add more lines. -- >z!
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[0],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[0],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)
        addLine(renderer, [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[0]],
                [CRRELPolyData.xBounds[1],CRRELPolyData.yBounds[1],CRRELPolyData.zBounds[1]], color=[0.4, 0.4, 0.4], lineWidth=2.0)

    # Add actors and render
    # Create a mapper and actor for mesh dataset
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(CRRELPolyData.GetPolyData())
    #normalsMesh=shell.compute_normals()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    renderer.AddActor(actor)

    return renderer


def ShowRender(renderer,camera=None,size=(800,800)):

    if camera == None:
        camera = vtk.vtkCamera()
        camera.SetPosition(0,-10,10)
        camera.SetFocalPoint(0,0,0)
        camera.SetViewUp(0,0,0)

    #    camera.ParallelProjectionOn()

    renderer.SetActiveCamera(camera)

    # Render
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(*size)
    renderWindow.Render()

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderWindowInteractor.Initialize()
    renderWindowInteractor.Start()
